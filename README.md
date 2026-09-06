# BREACH — Blacksite (fork)

**Live:** (not deployed yet)

Fork of `alesha-pro/bench-portal @ 2fa5c82` → `games/breach-blacksite-astra`.

Static Three.js horde-survival FPS. No build step: `index.html` + prebuilt bundle in `assets/`.
Upstream ships only build output, so tuning happens directly in `assets/index-DjtD87GC.js`
(game logic lives in the tail of the file) and in `assets/index-E51VA0F4.css` / `index.html`
(both unminified-friendly).

## Run

    python3 -m http.server 8788

Then open http://localhost:8788

## Debug hook

The bundle exposes `window.__BREACH__.state` — mode, wave, hostiles, score, health,
weapon, ammo, fps, drawCalls, player position and live target list.
