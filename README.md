# BREACH — Blacksite (fork)

**Live:** https://keksmd.github.io/breach-blacksite/

Fork of `alesha-pro/bench-portal @ 2fa5c82` → `games/breach-blacksite-astra`.

Static Three.js horde-survival FPS. No build step: `index.html` + prebuilt bundle in `assets/`.
Upstream ships only build output, so tuning happens directly in `assets/index-5d21b869.js`
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

The net is a 41 -> 64 -> 64 MLP with tanh hidden layers, three softmax heads (move,
fire, aim lead) and a value head. Head logits are soft-clipped to +-6 (`6 * tanh(z / 6)`)
so no head can collapse to a one-hot policy, and AdamW weight decay (1e-4) keeps the
weights bounded. The first net (one ReLU layer, plain Adam) drifted after ~700k updates
into |W| ~ 60, pre-activations ~ 1500, 63 of 64 units always on and entropy 0.000 on every
head, i.e. a deterministic linear policy whose only exploration was the 10 % epsilon;
those weights sit in `server/data/backup-relu1-obs41/` together with their transitions
and are not loaded. `/policy` carries `arch: "tanh2"` and the game refuses any other.

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
    +0.02/tick    team mode only: alive, ramping from 0 over the bot's first 30 s
                  (0.04 per second at full ramp, so a minute alive is worth ~2 hits)

Nothing else: no miss penalty, no death penalty, no class shaping, no
kill or team bonus. Each head samples with 10 % uniform exploration. Weights persist in
`server/data/` across restarts; do not delete them.

## Team deathmatch (local only)

The menu's second button, TEAM DEATHMATCH 10v10, gives each team one spawn: ALPHA in the
south-east pocket behind the MAINTENANCE block (the DANGER / LIVE POWER sign, around
x 30 z 28, spread 2.5 x 3 m), BRAVO in the gut behind BAY 03: the 3 m corridor between
the bay's west face and the map wall (around x -33 z -29, spread 0.8 x 3 m, the only way
out is south). The pair was picked by sweeping the collider map with the bots' own
line-of-sight test: no point of one spawn sees any point of the other, BRAVO walks about
20 m out of the gut before ALPHA's spawn comes into view and ALPHA about 55 m before it
sees BRAVO's. You are on ALPHA with 9 bots, start in that pocket facing west;
BRAVO fields 10. 3 of every 10 bots per team carry only the knife and run at 6.9 m/s
(between your walk and sprint); the rest are soldiers with a rifle drawn at random from
your own weapon table (MK18, M590, MK14, P226), all driven by the same RL nets as
survival (knife bots by the melee net). No markers over heads: team colour is on the
shoulder and neck chevrons, blue = ALPHA, green = BRAVO.

A small round minimap sits under the BREACH title, north up, centred on you with a 16 m
radius: box colliders above 1 m as grey blocks, nearby zone names (OPS, BAY 03, SECTOR 07, MAINT, POWER, WEST LANE, YARD,
LOGISTICS, SOUTH LOT), you as the gold arrow, living ALPHA bots as blue dots, and BRAVO
only as ALPHA's team memory: bright green when seen in the last second, dim green for the
remembered position up to 12 s. Your own line of sight feeds that memory too, so what you
see, your bots know. In survival it shows every hostile in red (screamers orange).

Bots only know what they have seen. Every 0.35 s each bot casts line-of-sight rays to
every enemy within 48 m; the ones it can see are its candidates, and it hunts the nearest
of them with the same melee / ranged code that survival uses against you (the target's
feature slots in the observation are filled from whatever it is hunting, so the trained
weights apply unchanged). Every sighting is written to a team memory (last seen x/z per
enemy, 12 s expiry, cleared at round start). A bot that sees nobody heads for the nearest
remembered position as a ghost target: it walks there, its observation says no line of
sight, and it cannot shoot or stab a ghost. With no memory at all a bot keeps walking its
own random heading, re-rolled only when the 6 m waypoint ahead is inside a wall; in ZONE
CONTROL the empty-memory fallback is the zone centre instead. Each team's flow field, a
multi-source BFS, is seeded from that memory (or the zone), never from live enemy
positions, so nothing is sensed through walls, and nobody is sent to the enemy spawn.
Bot models face their own target, not you.

Team-mode bots are steered by the policy alone: no flank arcs, no hold band, no
break-off retreat. The frame the policy moves in is the direction to whatever `tmTarget`
returned (visible enemy, remembered position via the flow field, zone centre, or own
heading); the net picks one of 8 directions or hold, plus fire and aim, every 0.5 s. What
remains outside the net is physics and rules: line of sight, the reaction delay before the
first shot, the 36 m fire range, magazine and reload, gravity and hops over knee-high
boxes.

A round ends when one side has nobody left or the 240 s clock runs out. Dead is dead: when
you go down you drop to the floor, lose movement, fire, aim and reload, and watch the rest
of the round (the HUD switches to `DOWN · ALPHA n · m BRAVO`). Five seconds after the round
ends the next one starts with fresh teams and you back on your feet. The kill feed reads
`ALPHA DOWN` / `BRAVO DOWN` for every kill; survival scoring text does not appear in team
modes. Bot-on-bot kills do not score or drop pickups; only your own kills do. Team mode
does not touch the survival save.

## Zone control (local only)

The third menu button, ZONE CONTROL 10v10, plays the same teams and spawns with one
objective: a 7 m circle at x -3 z -1 in the middle of the yard (translucent cylinder with two
rings; on the minimap a dashed circle in the colour of whoever leads). Each team has its own
percentage. While at least one member of a team stands inside, that team's counter climbs
at 100 % per 60 s; if both teams are inside, both climb. The first to 100 % takes the
round. Nobody stays dead: bots redeploy at their own spawn 5 s after dying, and so do you.
Those 5 s you are out of the game: on the floor, no movement, no fire, no aim, HUD counting
`REDEPLOY IN 5s`, and you cannot hold the zone while down. The zone sits 38 m of walking
from either spawn
(the centre was swept for equal path length once BRAVO moved into the gut), so the first
contact is at the ring, not in a corridor, and a respawned bot needs a real walk back. RL bots
earn +0.05 per decision while inside the circle on top of the usual hit / damage rewards.
`?auto=zone` runs it headless, `__BREACH__.start(2)` from script, and
`__BREACH__.state.team` reports `mode`, `cap` (both percentages) and `queue` (bots waiting
to respawn).

The nets see the zone. Five observation slots (37-41, zeros outside zone mode) carry: in
the circle or not, the direction to the zone centre in the bot's movement frame (forward
and lateral components), the distance over 40 m, and own minus enemy percentage. Rewards
on top of the usual hit / damage ones: +0.05 per decision inside the circle, and at the
end of the round +2 to every bot on the side that reached 100 % and -2 to the other side.
The observation grew from 36 to 41; `rl_server.py` pads older weights with zero rows and
older transitions with zeros, so the survival and deathmatch training carries over
unchanged (the pre-zone weights are kept in `server/data/backup-obs36/`).

Stats are the same on both sides in team mode. Bots have 100 hp, run at your walking speed
(5.1 m/s) and shoot with the weapon table's numbers: the same damage, pellet count, angular
spread and range as the gun in your hands, headshots x2.5 for everyone (survival's one-shot
kill on bots is off). All gun damage in team mode, yours and theirs, is scaled by 0.6
(knife stays 40): MK18 17 per round, MK14 55, P226 22, M590 11 per pellet before the
shotgun curve, so an MK18 burst no longer kills from full health and only the MK14
headshot is still a one-shot. Shotgun falloff is the same curve for both: each pellet's
damage is scaled by `2.15 - dist / 13`, clamped to 0.22..2.15, so a pellet does 2.15x at
the muzzle, 1x at 15 m, and 0.22x from 25 m out, on top of the 0.048 rad spread that
scatters the pattern over 3 m at 30 m and the 37 m ray limit.

A bot's trigger pull is the 0.72 s windup, then a burst at the weapon's own fire interval
(4 rounds for the MK18, 2 for the MK14 / P226, 1 shell for the M590), then a 1.7-2.9 s
pause; bots aim at the chest, so their headshots come from spread the same way yours do.
Before the windup can start the bot must have had continuous line of sight to its target
for a reaction time drawn per sighting from 0.3..0.6 s, so the first bullet comes 1.0..1.3
s after you show yourself, and the target must be inside 36 m (about half the map; the
ray is capped there too, so no bot fires across the whole arena). If line of sight is lost
while winding up the bot holds fire; if it is lost mid-burst the burst stops. Bots carry
magazines from the same table (30 / 8 / 16 / 24) and reload with the same timings (1.85 /
2.65 / 2.25 / 2.1 s) when empty, or when under 30 % and out of sight; a reloading bot
cannot start a windup and shows as reloading in its enemies' observation (slot 13).
Reserve ammo is infinite for bots. Both sides carry a knife: `F` swings it (40 damage,
1.9 m reach, 0.8 s cooldown; the old inspect animation is gone), and a bot inside 1.9 m of
its target stabs instead of shooting. In team mode observation slot 3 holds the bot's own
weapon index / 3 instead of the heavy flag.

Bots have vertical physics in team mode: gravity 17, jump 6.1 m/s (same as you), landing
on box tops, and their box collisions are filtered by feet height like yours. A bot hops
over a box up to 1 m above its feet that blocks its path (the 0.7 m posts; barriers and
crates are as unjumpable for them as for you), and a moving bot that was hit in the last
1.5 s hops randomly (0.7 per second) as a dodge. Half the bots per round are crouchers:
while they hold position, wind up or burst with the target in sight they crouch to 1.25 / 1.7
of their height,
which lowers the head / torso / legs spheres and the muzzle the same way (their model is
squashed on Y so you can read it). No RL head drives jump or crouch yet; that would need
an observation change and a policy migration.
A round also ends after 240 s of sim time as a draw, so camping shooters cannot stall it.

## Collision

Bodies are circles sliding against axis-aligned boxes (`Ba`). A box pushes you out through the
face nearest to where you already are, never through the face your velocity points at, so a
body that ended up overlapping a box (stepping off a crate edge, crouching next to a barrier,
a knockback) is nudged out by a few centimetres instead of being thrown across it. A body
that is inside a box on both axes leaves along the axis of least penetration. Standing room on
top of a box matches the body radius, so you can sit on a crate edge without being pushed off.
Push-outs land 2 mm clear of the inflated box, never on its edge, because a body parked exactly
on the edge reads as inside on the next frame and gets shoved along the other axis (that was
the sideways lurch when walking into a crate face). After the per-box pass a body that is still
inside some box (the 0.2 m slot between two crates, a corner shared by three) is moved to the
nearest spot that is clear of every box (`unwedge`), so it can never be wedged for good.
Landing uses the feet height from before gravity was applied for the frame, so a fast fall on a
slow frame (dt is clamped to 50 ms; at 6 m/s that is 30 cm) cannot pass through a crate top and
leave you inside the box below it.
Bots that make no headway on their chosen direction (a wall or a corner) fall back to the flow
field direction, then to sliding along the wall either way, so they do not park in corners.

## Self-play (headless training)

    pip3 install playwright
    python3 server/selfplay.py --games 16

Opens N headless Chrome tabs (Google Chrome via `channel=chrome`, or `playwright install
chromium`) at `/?auto=tdm` (`--mode zone` opens `/?auto=zone`). Auto mode fields 10 bots
per side, drops the player, skips
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
weapon, ammo, fps, drawCalls, player position and live target list. `__BREACH__.player` is
the live player object (`pos` is the eye, feet are `pos.y - height`) and `__BREACH__.colliders`
the box list, `__BREACH__.enemies` the live bot list, `__BREACH__.weapons` the weapon table
and `__BREACH__.shoot(bot, target)` fires one probe ray from a bot's muzzle at a target's
chest with the bot's weapon spread (returns victim / head / dist), so collision and hit
cases can be reproduced from the console by teleporting. `__BREACH__.start(team)` starts a
round from script (1 = team deathmatch, 2 = zone control, 0 = survival). Query flags for harnesses: `?nopause` keeps
the sim running when the tab loses focus or the pointer lock, `?norender` skips drawing
(headless Chrome on swiftshader otherwise runs the sim 10x slow).
