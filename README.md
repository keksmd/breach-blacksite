# BREACH — Blacksite (fork)

**Live:** https://keksmd.github.io/breach-blacksite/

Fork of `alesha-pro/bench-portal @ 2fa5c82` → `games/breach-blacksite-astra`.

Static Three.js horde-survival FPS. No build step: `index.html` + prebuilt bundle in `assets/`.
Upstream ships only build output, so tuning happens directly in `assets/index-DjtD87GC.js`
(game logic lives in the tail of the file) and in `assets/index-E51VA0F4.css` / `index.html`
(both unminified-friendly).

## Run

    python3 -m http.server 8788

Then open http://localhost:8788

## RL bots (local only)

    pip3 install numpy
    python3 server/rl_server.py

Backend listens on http://localhost:8790 (`RL_PORT` to change). When it is up, the game
hands every second enemy to a small policy network (MLP 10-32-6) and streams each bot's
trajectory (observation, action, reward) to `POST /episodes`. Episodes land in
`server/data/episodes.jsonl`; a trainer thread reruns REINFORCE every 20 s on the last
3000 episodes and bumps `server/data/policy.json`. The game refetches `/policy` every 30 s,
so bots improve between waves without a reload. `GET /stats` shows episodes, steps,
policy version and mean return of the last 100 episodes. On GitHub Pages the backend is
absent, `/policy` fails within 1.2 s and all bots stay scripted.

Rewards: +0.1 per damage point dealt, -0.03 per damage point taken, -0.01 per decision,
-2 on death, +5 to every living bot when the player dies. `__BREACH__.state.rl` exposes
ready/version/bots/decisions/queued/sent.

## Debug hook

The bundle exposes `window.__BREACH__.state` — mode, wave, hostiles, score, health,
weapon, ammo, fps, drawCalls, player position and live target list.
