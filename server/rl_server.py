#!/usr/bin/env python3
"""Local RL backend for BREACH bots.

Run it next to the game (python3 server/rl_server.py) and the game will hand
every second bot to the policies served here: one net for melee hostiles,
one for shooters, each trained only on its own class. Episodes arrive on
POST /episodes, land in server/data/episodes.jsonl, and a background thread
retrains both with REINFORCE every TRAIN_EVERY seconds. GET /policy returns
the current weights, GET /stats shows how much has been collected. GET/POST /save
keeps the player's run (wave, score, ammo) so a page reload resumes it.
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EP_FILE = os.path.join(DATA, "episodes.jsonl")
POLICY_FILES = {"melee": os.path.join(DATA, "policy_melee.json"), "ranged": os.path.join(DATA, "policy_ranged.json")}
SAVE_FILE = os.path.join(DATA, "save.json")
PORT = int(os.environ.get("RL_PORT", "8790"))

OBS, HID, ACT = 24, 48, 6
GAMMA = 0.96
LR = 2e-3
EPOCHS = 4
BATCH = 256
ENTROPY = 0.03
TRAIN_EVERY = 20.0
WINDOW = 3000
MIN_EPISODES = 8

os.makedirs(DATA, exist_ok=True)
lock = threading.Lock()
episodes = []
new_since_train = 0
last_train = {"at": None, "melee": None, "ranged": None}


class Policy:
    """Two-layer MLP with a softmax head, trained by hand-rolled Adam."""

    def __init__(self, name):
        self.name = name
        rng = np.random.default_rng(0)
        self.W1 = rng.normal(0, 0.3, (OBS, HID))
        self.b1 = np.zeros(HID)
        self.W2 = rng.normal(0, 0.1, (HID, ACT))
        self.b2 = np.zeros(ACT)
        self.version = 0
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        pre = X @ self.W1 + self.b1
        h = np.maximum(pre, 0)
        logits = h @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return h, p

    def step(self, X, A, adv):
        n = len(X)
        h, p = self.forward(X)
        onehot = np.zeros_like(p)
        onehot[np.arange(n), A] = 1
        logp = np.log(p + 1e-8)
        entropy = -(p * logp).sum(axis=1, keepdims=True)
        loss = -(logp[np.arange(n), A] * adv).mean() - ENTROPY * entropy.mean()
        d_logits = (p - onehot) * adv[:, None] / n
        d_logits += ENTROPY * p * (logp + entropy) / n
        gW2 = h.T @ d_logits
        gb2 = d_logits.sum(axis=0)
        dh = d_logits @ self.W2.T
        dh[h <= 0] = 0
        gW1 = X.T @ dh
        gb1 = dh.sum(axis=0)
        self.adam([gW1, gb1, gW2, gb2])
        return float(loss)

    def adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params(), grads)):
            g = np.clip(g, -5, 5)
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1**self.t)
            vh = self.v[i] / (1 - b2**self.t)
            p -= LR * mh / (np.sqrt(vh) + eps)

    def to_json(self):
        return {
            "version": self.version,
            "obs": OBS,
            "actions": ACT,
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }

    def load(self, d):
        self.W1 = np.array(d["W1"])
        self.b1 = np.array(d["b1"])
        self.W2 = np.array(d["W2"])
        self.b2 = np.array(d["b2"])
        self.version = int(d.get("version", 0))
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]

    def save(self):
        path = POLICY_FILES[self.name]
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f)
        os.replace(tmp, path)


policies = {"melee": Policy("melee"), "ranged": Policy("ranged")}


def class_of(ep):
    return "ranged" if ep.get("ranged") else "melee"


def policy_json():
    return {
        "version": sum(p.version for p in policies.values()),
        "obs": OBS,
        "actions": ACT,
        "melee": policies["melee"].to_json(),
        "ranged": policies["ranged"].to_json(),
    }


def valid(ep):
    steps = ep.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    for st in steps:
        o = st.get("o")
        if not isinstance(o, list) or len(o) != OBS:
            return False
        a = st.get("a")
        if not isinstance(a, int) or not 0 <= a < ACT:
            return False
        if not isinstance(st.get("r"), (int, float)):
            return False
    return True


def returns_of(ep):
    G = 0.0
    out = []
    for st in reversed(ep["steps"]):
        G = st["r"] + GAMMA * G
        out.append(G)
    out.reverse()
    return out


def train_once():
    global new_since_train
    with lock:
        new_since_train = 0
        by_class = {name: [e for e in episodes if class_of(e) == name][-WINDOW:] for name in policies}
    for name, batch in by_class.items():
        if len(batch) >= MIN_EPISODES:
            train_class(policies[name], batch)


def train_class(policy, batch):
    X, A, G = [], [], []
    for ep in batch:
        rets = returns_of(ep)
        for st, g in zip(ep["steps"], rets):
            X.append(st["o"])
            A.append(st["a"])
            G.append(g)
    X = np.array(X, dtype=np.float64)
    A = np.array(A, dtype=np.int64)
    G = np.array(G, dtype=np.float64)
    adv = (G - G.mean()) / (G.std() + 1e-6)
    rng = np.random.default_rng()
    loss = None
    with lock:
        for _ in range(EPOCHS):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), BATCH):
                sel = idx[s : s + BATCH]
                loss = policy.step(X[sel], A[sel], adv[sel])
        policy.version += 1
        policy.save()
        last_train["at"] = time.time()
        last_train[policy.name] = {"version": policy.version, "episodes": len(batch), "steps": len(X), "loss": loss, "mean_return": float(G.mean())}
    print(f"[train] {policy.name} v{policy.version} episodes={len(batch)} steps={len(X)} loss={loss:.4f} mean_return={G.mean():.3f}", flush=True)


def trainer():
    while True:
        time.sleep(TRAIN_EVERY)
        try:
            if new_since_train > 0:
                train_once()
        except Exception as ex:
            print(f"[train] failed: {ex!r}", flush=True)


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
                tail = episodes[-100:]
                mean_ret = float(np.mean([sum(s["r"] for s in e["steps"]) for e in tail])) if tail else None
                self.reply(
                    200,
                    {
                        "version": sum(p.version for p in policies.values()),
                        "versions": {name: p.version for name, p in policies.items()},
                        "episodes": len(episodes),
                        "episodes_by_class": {name: sum(1 for e in episodes if class_of(e) == name) for name in policies},
                        "steps": sum(len(e["steps"]) for e in episodes),
                        "pending": new_since_train,
                        "mean_return_last_100": mean_ret,
                        "last_train": last_train,
                    },
                )
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        global new_since_train
        if self.path not in ("/episodes", "/save"):
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
        eps = [e for e in body.get("episodes", []) if isinstance(e, dict) and valid(e)]
        with lock:
            with open(EP_FILE, "a") as f:
                for e in eps:
                    e["received"] = time.time()
                    f.write(json.dumps(e) + "\n")
            episodes.extend(eps)
            new_since_train += len(eps)
        self.reply(200, {"accepted": len(eps), "total": len(episodes)})

    def log_message(self, fmt, *args):
        if "/policy" in fmt % args or "/save" in fmt % args or "/stats" in fmt % args:
            return
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


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
    global new_since_train
    for name, path in POLICY_FILES.items():
        if os.path.exists(path):
            with open(path) as f:
                policies[name].load(json.load(f))
    if os.path.exists(EP_FILE):
        with open(EP_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except ValueError:
                    continue
                if valid(ep):
                    episodes.append(ep)
    new_since_train = 0
    print(f"[boot] melee v{policies['melee'].version}, ranged v{policies['ranged'].version}, {len(episodes)} episodes on disk", flush=True)


if __name__ == "__main__":
    load_state()
    threading.Thread(target=trainer, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[boot] listening on http://localhost:{PORT}", flush=True)
    srv.serve_forever()
