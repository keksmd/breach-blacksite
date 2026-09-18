# BREACH — Blacksite (fork)

**Live:** https://keksmd.github.io/breach-blacksite/

Fork of `alesha-pro/bench-portal @ 2fa5c82` → `games/breach-blacksite-astra`.

Static Three.js horde-survival FPS. No build step: `index.html` + prebuilt bundle in `assets/`.
Upstream ships only build output, so tuning happens directly in `assets/index-5b07d2b6.js`
(game logic lives in the tail of the file) and in `assets/index-49044fd1.css` / `index.html`
(both unminified-friendly).

## Run

    python3 -m http.server 8788

Then open http://localhost:8788

## RL bots (local only)

    pip3 install numpy
    python3 server/rl_server.py

Backend listens on http://localhost:8790 (`RL_PORT` to change). When it is up, every
hostile (screamers excepted) is driven end to end by a policy network: where to run, when
to swing or pull the trigger, and how far to lead the shot. The scripted AI only runs when
the backend is absent (GitHub Pages: `/policy` fails within 1.2 s, bots stay scripted).

Two nets, one for melee hostiles and one for shooters, each MLP 36-64 with four heads:

    move  9   stop, or one of 8 directions in 45 degree steps around the player
    fire  2   melee: swing now; ranged: start the 0.72 s windup
    aim   7   lead the shot by 0..0.9 s of player velocity, sampled at windup start
    value 1   critic (state value) used for the advantage

Hits are geometry, not dice. When the windup ends the bot fires one ray from its muzzle
toward the aimed point (where the target was at windup start plus the chosen lead), with a
fixed angular spread of 0.02 rad (heavies 0.026). The ray stops at the first box collider
it crosses and at the first body it meets, using the same head / torso / legs spheres the
player's bullets use. So there is no range cap and no distance coefficient: a 1 degree
wobble is 0.2 m at 10 m and 1 m at 50 m, cover blocks the bullet, a crouched player behind
a 1.35 m barrier is not visible to a muzzle at 1.3 m, and a teammate standing in the line
of fire eats the shot (no damage, no reward). Tracers end where the ray stopped, so a shot
into a wall shows as a shot into a wall. A melee swing further than 1.9 m is a whiff.

Learning is online. The game posts transitions (obs, actions, reward, next obs) every
second; the backend applies each batch immediately as an advantage actor-critic step
(plus two minibatches from a 20000-transition replay buffer per class) and the game
refetches weights every 3 s. A bot's action is therefore corrected within a few seconds,
not once per round. The HUD line `RL u184 · 12 BOTS · 9310 STEPS` counts gradient updates
and transitions received. Everything persists in `server/data/` (`policy_melee.json`,
`policy_ranged.json` every 30 s, `transitions.jsonl` append-only) and is reloaded on
start, so RL progress survives restarts the same way the run save does. Several tabs can
feed the same backend at once. `GET /stats` shows updates per class, transitions, replay
size, mean reward and the shooters' hit rate.

Observation (36 floats, every 0.5 s; directions are in the bot's frame where index 0 points
at the player):

    0  distance to player / 24        18 nearest ally distance / 10
    1  line of sight (0/1)            19 hostiles alive / 20
    2  own health fraction            20 wave / 10
    3  heavy (0/1)                    21 cover 1 m to the left (0/1)
    4  staggered (0/1)                22 cover 1 m to the right (0/1)
    5  attack cooldown (-1..1)        23 previous move / 8
    6  winding up a shot (0/1)        24 player velocity along the bot-player axis / 8
    7  player speed / 8               25 player velocity across it / 8
    8  player facing dot (-1..1)      26 flow field direction, along
    9  player facing cross            27 flow field direction, across
    10 player airborne (0/1)          28..35 free distance in 8 directions (0..3 m)
    11 player sliding (0/1)
    12 player health / 100
    13 player reloading (0/1)
    14 player aiming down sights
    15 player weapon index / 3
    16 time since player hurt / 3
    17 allies within 6 m / 5

Rewards go to the bot that earned them; the two nets are shared per class, so every
bot's transitions train the same net, but credit is per transition:

    +1 +0.1/dmg   landed a hit (melee swing or shot that connects)
    +0.05/m       ...times the distance the shot travelled, so a 20 m hit pays +1 extra
    -0.03/dmg     damage taken

Nothing else: no per-tick cost, no miss penalty, no death penalty, no class shaping, no
kill or team bonus. Each head samples with 10 % uniform exploration. Weights persist in
`server/data/` across restarts; do not delete them.

## Team deathmatch (local only)

The menu's second button, TEAM DEATHMATCH 10v10, splits the map: ALPHA spawns along the
north edge (z > 0), BRAVO along the south (z < 0). You are on ALPHA with 9 bots; BRAVO
fields 10. Each team is 4 melee, 2 heavies, 4 shooters, all driven by the same RL nets as
survival. Blue marker cube = ALPHA, green = BRAVO.

Bots pick the nearest living enemy across both teams (the player counts for BRAVO) and
fight it with the same melee / ranged code that survival uses against you; the target's
feature slots in the observation (health, velocity, facing, reload...) are filled from
whatever they are hunting, so the trained weights apply unchanged. Each team gets its own
flow field, a multi-source BFS from every living enemy, so bots without line of sight
path towards the closest one instead of hugging walls. Bot models face their own target,
not you.

A round ends when one side has nobody left. You respawn at an ALPHA spawn while any ALPHA
bot is alive; once the last one falls, your death ends the round. Five seconds later the
next round starts with fresh teams. HUD shows `ALPHA n · m BRAVO` and the round number.
Bot-on-bot kills do not score or drop pickups; only your own kills do. Team mode does
not touch the survival save.

Stats are equal in team mode: every bot has 100 hp and runs at your walking speed (5.1
m/s), shooters deal the MK18's 29 per hit and DMRs the MK14's 92, headshots are x2.5 for
both sides instead of the survival mode's one-shot kill on bots.
A round also ends after 240 s of sim time as a draw, so camping shooters cannot stall it.

## Collision

Bodies are circles sliding against axis-aligned boxes (`Ba`). A box pushes you out through the
face nearest to where you already are, never through the face your velocity points at, so a
body that ended up overlapping a box (stepping off a crate edge, crouching next to a barrier,
a knockback) is nudged out by a few centimetres instead of being thrown across it. A body
that is inside a box on both axes leaves along the axis of least penetration. Standing room on
top of a box matches the body radius, so you can sit on a crate edge without being pushed off.
Bots that make no headway on their chosen direction (a wall or a corner) fall back to the flow
field direction, then to sliding along the wall either way, so they do not park in corners.

## Self-play (headless training)

    pip3 install playwright
    python3 server/selfplay.py --games 16

Opens N headless Chrome tabs (Google Chrome via `channel=chrome`, or `playwright install
chromium`) at `/?auto=tdm`. Auto mode fields 10 bots per side, drops the player, skips
rendering and the pause-on-blur handlers, and starts the first round once the policy has
loaded. Every tab posts its transitions to the same backend, so the nets train on bot vs
bot at roughly 30 transitions/s per tab. The script prints rounds, wins, transition
count, update counts and mean reward every `--report` seconds; `--minutes` bounds the run,
0 runs until Ctrl-C. Needs the static server (`--port`, default 4178) and
`server/rl_server.py` already running.

The same backend keeps your run: wave, score, kills, health, weapon and ammo are posted to
`POST /save` every 4 s while playing. Reload the page and the menu button reads
`RESUME WAVE 05 · 01234`; DEPLOY continues from the start of that wave. Dying or choosing
RESTART OPERATION clears the save. `__BREACH__.state.rl` exposes
ready/version/bots/decisions/queued/sent.

## Debug hook

The bundle exposes `window.__BREACH__.state` — mode, wave, hostiles, score, health,
weapon, ammo, fps, drawCalls, player position and live target list.
