# BREACH — Blacksite (fork)

**Live:** https://keksmd.github.io/breach-blacksite/

Fork of `alesha-pro/bench-portal @ 2fa5c82` → `games/breach-blacksite-astra`.

Static Three.js horde-survival FPS. No build step: `index.html` + prebuilt bundle in `assets/`.
Upstream ships only build output, so tuning happens directly in `assets/index-e23056ba.js`
(game logic lives in the tail of the file) and in `assets/index-8d3db6dc.css` / `index.html`
(both unminified-friendly).

## Run

    python3 -m http.server 8788

Then open http://localhost:8788

## RL bots (local only)

    pip3 install numpy
    python3 server/rl_server.py

Backend listens on http://localhost:8790 (`RL_PORT` to change). When it is up, the game
hands every second enemy to a policy network and streams each bot's trajectory
(observation, action, reward) to `POST /episodes`. There are two nets, MLP 24-48-6 each:
one for melee hostiles, one for shooters, trained only on episodes of their own class.
Episodes land in `server/data/episodes.jsonl`; a trainer thread reruns REINFORCE every
20 s on the last 3000 episodes per class (needs at least 8) and bumps
`server/data/policy_melee.json` / `policy_ranged.json`. Both files plus the episode log
are reloaded on backend start, so RL progress survives restarts the same way the run
save does. The game refetches `/policy` every 30 s, so bots improve between waves without
a reload. `GET /stats` shows episodes and versions per class, steps and mean return of
the last 100 episodes. On GitHub Pages the backend is absent, `/policy` fails within
1.2 s and all bots stay scripted.

Observation (24 floats, every 0.5 s):

    0  distance to player / 24        12 player health / 100
    1  line of sight (0/1)            13 player reloading (0/1)
    2  own health fraction            14 player aiming down sights (0..1)
    3  heavy (0/1)                    15 player weapon index / 3
    4  staggered (0/1)                16 time since player was hurt / 3 (capped)
    5  attack cooldown (-1..1)        17 allies within 6 m / 5
    6  ranged and aiming (0/1)        18 nearest ally distance / 10
    7  player speed / 8               19 hostiles alive / 20
    8  player facing dot (-1..1)      20 wave / 10
    9  player facing cross (left/right) 21 cover 1 m to the left (0/1)
    10 player airborne (0/1)          22 cover 1 m to the right (0/1)
    11 player sliding (0/1)           23 previous action / 5

Actions: 0 push, 1 strafe left, 2 strafe right, 3 back off, 4 hold (fire), 5 flank.
Each decision samples the policy with 15 % uniform exploration, so even a trained policy
keeps trying other moves.

Rewards: +0.1 per damage point dealt, -0.03 per damage point taken, -0.01 per decision,
-2 on death, +5 to every living bot when the player dies. Class shaping on top: melee
gets +0.05 per metre closed (-0.05 per metre lost, capped at 1 m per tick); shooters get
+0.02 per tick when they hold line of sight at 7.5..15 m and -0.02 when closer than 6 m.

What you see in game when the backend is up: a green line under the score (`RL v3 · 4 BOTS ·
120 EP`), a green cube above every RL-driven hostile, and those hostiles strafing, backing
off and holding instead of running the scripted line. `__BREACH__.state.rl` exposes
ready/version/bots/decisions/queued/sent.

The same backend keeps your run: wave, score, kills, health, weapon and ammo are posted to
`POST /save` every 4 s while playing. Reload the page and the menu button reads
`RESUME WAVE 05 · 01234`; DEPLOY continues from the start of that wave. Dying or choosing
RESTART OPERATION clears the save.

## Debug hook

The bundle exposes `window.__BREACH__.state` — mode, wave, hostiles, score, health,
weapon, ammo, fps, drawCalls, player position and live target list.
