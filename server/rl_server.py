#!/usr/bin/env python3
"""Local RL backend for BREACH bots.

Run it next to the game (python3 server/rl_server.py) and every hostile in
the game is driven by the policies served here: one net for melee hostiles,
one for shooters. Each net has a shared hidden layer and four heads: move
(stop or one of 8 directions around the player), fire, aim lead and a state
value. The game streams transitions (obs, actions, reward, next obs) on
POST /transitions about once a second and every batch is applied at once as
an advantage actor-critic step, so a bot's action is corrected within a few
seconds, not once per round. GET /policy returns the current weights, GET
/stats how much has been learned, GET/POST /save keeps the player's run.
"""
import json
import os
import signal
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TR_FILE = os.path.join(DATA, "transitions.jsonl")
POLICY_FILES = {"melee": os.path.join(DATA, "policy_melee.json"), "ranged": os.path.join(DATA, "policy_ranged.json")}
SAVE_FILE = os.path.join(DATA, "save.json")
PORT = int(os.environ.get("RL_PORT", "8790"))

OBS, HID = 36, 64
HEADS = {"m": 9, "f": 2, "a": 7}
GAMMA = 0.96
LR = 1e-3
ENTROPY = 0.02
VALUE_COEF = 0.5
REPLAY = 20000
REPLAY_BATCH = 256
REPLAY_STEPS = 2
PERSIST_EVERY = 30.0

os.makedirs(DATA, exist_ok=True)
lock = threading.Lock()
replay = {"melee": deque(maxlen=REPLAY), "ranged": deque(maxlen=REPLAY)}
stats = {"transitions": 0, "reward": deque(maxlen=2000), "hits": deque(maxlen=500)}


class Policy:
    """Actor-critic MLP: one hidden layer, three softmax heads, one value head."""

    def __init__(self, name):
        self.name = name
        rng = np.random.default_rng(0)
        self.p = {
            "W1": rng.normal(0, 0.2, (OBS, HID)),
            "b1": np.zeros(HID),
            "Wv": rng.normal(0, 0.1, (HID, 1)),
            "bv": np.zeros(1),
        }
        for h, n in HEADS.items():
            self.p["W" + h] = rng.normal(0, 0.1, (HID, n))
            self.p["b" + h] = np.zeros(n)
        self.version = 0
        self.updates = 0
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        pre = X @ self.p["W1"] + self.p["b1"]
        h = np.maximum(pre, 0)
        out = {}
        for k in HEADS:
            z = h @ self.p["W" + k] + self.p["b" + k]
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            out[k] = e / e.sum(axis=1, keepdims=True)
        val = (h @ self.p["Wv"] + self.p["bv"])[:, 0]
        return h, out, val

    def value(self, X):
        return self.forward(X)[2]

    def step(self, X, acts, R, X2, done, use_aim):
        n = len(X)
        h, probs, val = self.forward(X)
        target = R + GAMMA * (1 - done) * self.value(X2)
        adv = target - val
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-6) if n > 1 else adv
        grads = {}
        dh = np.zeros_like(h)
        for k in HEADS:
            p = probs[k]
            A = acts[k]
            onehot = np.zeros_like(p)
            onehot[np.arange(n), A] = 1
            logp = np.log(p + 1e-8)
            ent = -(p * logp).sum(axis=1, keepdims=True)
            mask = use_aim if k == "a" else np.ones(n)
            dz = (p - onehot) * adv_n[:, None] + ENTROPY * p * (logp + ent)
            dz *= mask[:, None] / n
            grads["W" + k] = h.T @ dz
            grads["b" + k] = dz.sum(axis=0)
            dh += dz @ self.p["W" + k].T
        dv = (-VALUE_COEF * adv / n)[:, None]
        grads["Wv"] = h.T @ dv
        grads["bv"] = dv.sum(axis=0)
        dh += dv @ self.p["Wv"].T
        dh[h <= 0] = 0
        grads["W1"] = X.T @ dh
        grads["b1"] = dh.sum(axis=0)
        self.adam(grads)
        self.updates += 1
        return float((adv**2).mean())

    def adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k, g in grads.items():
            g = np.clip(g, -5, 5)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1**self.t)
            vh = self.v[k] / (1 - b2**self.t)
            self.p[k] -= LR * mh / (np.sqrt(vh) + eps)

    def to_json(self):
        d = {k: v.tolist() for k, v in self.p.items()}
        d["version"] = self.updates
        d["obs"] = OBS
        return d

    def load(self, d):
        for k in self.p:
            if k in d:
                self.p[k] = np.array(d[k], dtype=np.float64)
        self.updates = int(d.get("version", 0))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}

    def save(self):
        path = POLICY_FILES[self.name]
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f)
        os.replace(tmp, path)


policies = {"melee": Policy("melee"), "ranged": Policy("ranged")}


def class_of(tr):
    return "ranged" if tr.get("ranged") else "melee"


def total_updates():
    return sum(p.updates for p in policies.values())


def policy_json():
    return {
        "version": total_updates(),
        "obs": OBS,
        "melee": policies["melee"].to_json(),
        "ranged": policies["ranged"].to_json(),
    }


def valid(tr):
    o = tr.get("o")
    if not isinstance(o, list) or len(o) != OBS:
        return False
    o2 = tr.get("o2")
    if o2 is not None and (not isinstance(o2, list) or len(o2) != OBS):
        return False
    for k, n in HEADS.items():
        a = tr.get(k)
        if not isinstance(a, int) or not 0 <= a < n:
            return False
    r = tr.get("r")
    if not isinstance(r, (int, float)) or r != r:
        return False
    return all(isinstance(x, (int, float)) and x == x for x in o) and (o2 is None or all(isinstance(x, (int, float)) and x == x for x in o2))


def tensors(batch):
    X = np.array([t["o"] for t in batch], dtype=np.float64)
    X2 = np.array([t["o2"] if t["o2"] is not None else t["o"] for t in batch], dtype=np.float64)
    done = np.array([0.0 if t["o2"] is not None else 1.0 for t in batch])
    R = np.array([t["r"] for t in batch], dtype=np.float64)
    acts = {k: np.array([t[k] for t in batch], dtype=np.int64) for k in HEADS}
    use_aim = np.array([1.0 if t.get("ranged") else 0.0 for t in batch])
    return X, acts, R, X2, done, use_aim


def learn(fresh):
    """One on-policy step on the batch just received, then a couple of replay steps."""
    by_class = {"melee": [], "ranged": []}
    for t in fresh:
        by_class[class_of(t)].append(t)
    out = {}
    rng = np.random.default_rng()
    with lock:
        for name, batch in by_class.items():
            if not batch:
                continue
            pol = policies[name]
            err = pol.step(*tensors(batch))
            buf = replay[name]
            buf.extend(batch)
            if len(buf) >= REPLAY_BATCH:
                idx = rng.choice(len(buf), size=(REPLAY_STEPS, REPLAY_BATCH), replace=True)
                for row in idx:
                    pol.step(*tensors([buf[i] for i in row]))
            out[name] = {"updates": pol.updates, "td_error": err, "batch": len(batch)}
    return out


def persist_all():
    with lock:
        for p in policies.values():
            p.save()


def persister():
    while True:
        time.sleep(PERSIST_EVERY)
        persist_all()


def shutdown(signum, frame):
    persist_all()
    print(f"[exit] saved melee u{policies['melee'].updates}, ranged u{policies['ranged'].updates}", flush=True)
    sys.exit(0)


class Handler(BaseHTTPRequestHandler):
    def cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def reply(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/policy":
            with lock:
                self.reply(200, policy_json())
        elif self.path == "/save":
            with lock:
                self.reply(200, read_save())
        elif self.path == "/stats":
            with lock:
                rw = list(stats["reward"])
                hits = list(stats["hits"])
                self.reply(
                    200,
                    {
                        "version": total_updates(),
                        "updates": {name: p.updates for name, p in policies.items()},
                        "transitions": stats["transitions"],
                        "replay": {name: len(b) for name, b in replay.items()},
                        "mean_reward_last_2000": float(np.mean(rw)) if rw else None,
                        "ranged_hit_rate_last_500": float(np.mean(hits)) if hits else None,
                    },
                )
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/transitions", "/save"):
            self.reply(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            self.reply(400, {"error": "bad json"})
            return
        if self.path == "/save":
            if not isinstance(body, dict):
                body = {}
            with lock:
                write_save(body)
            self.reply(200, body)
            return
        trs = [t for t in body.get("transitions", []) if isinstance(t, dict) and valid(t)]
        if trs:
            now = time.time()
            with lock:
                with open(TR_FILE, "a") as f:
                    for t in trs:
                        t["at"] = now
                        f.write(json.dumps(t) + "\n")
                stats["transitions"] += len(trs)
                for t in trs:
                    stats["reward"].append(t["r"])
                    if t.get("ranged") and t["f"] == 1:
                        stats["hits"].append(1.0 if t["r"] > 0.3 else 0.0)
            res = learn(trs)
        else:
            res = {}
        self.reply(200, {"accepted": len(trs), "learned": res})

    def log_message(self, fmt, *args):
        line = fmt % args
        if "/policy" in line or "/save" in line or "/stats" in line or "/transitions" in line:
            return
        sys.stderr.write("%s %s\n" % (self.address_string(), line))


def read_save():
    if not os.path.exists(SAVE_FILE):
        return {}
    try:
        with open(SAVE_FILE) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}


def write_save(d):
    tmp = SAVE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, SAVE_FILE)


def load_state():
    for name, path in POLICY_FILES.items():
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            if d.get("obs") == OBS:
                policies[name].load(d)
            else:
                print(f"[boot] {path} has obs={d.get('obs')}, current build wants {OBS}; starting {name} fresh", flush=True)
    if os.path.exists(TR_FILE):
        with open(TR_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except ValueError:
                    continue
                if valid(t):
                    replay[class_of(t)].append(t)
                    stats["transitions"] += 1
    print(f"[boot] melee u{policies['melee'].updates}, ranged u{policies['ranged'].updates}, replay melee={len(replay['melee'])} ranged={len(replay['ranged'])}", flush=True)


def reporter():
    while True:
        time.sleep(30)
        with lock:
            rw = list(stats["reward"])
            hits = list(stats["hits"])
        print(f"[learn] melee u{policies['melee'].updates} ranged u{policies['ranged'].updates} transitions={stats['transitions']} mean_r={np.mean(rw) if rw else 0:.3f} hit_rate={np.mean(hits) if hits else 0:.2f}", flush=True)


if __name__ == "__main__":
    load_state()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    threading.Thread(target=persister, daemon=True).start()
    threading.Thread(target=reporter, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[boot] listening on http://localhost:{PORT}", flush=True)
    srv.serve_forever()
