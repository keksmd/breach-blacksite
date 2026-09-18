"""Headless self-play: N browser tabs run team deathmatch bot-vs-bot against the RL backend.

Usage: python3 server/selfplay.py --games 16 --port 4178 --minutes 0 --mode tdm|zone
Requires: pip3 install playwright; Google Chrome installed (channel=chrome) or `playwright install chromium`.
The static site must already be served on --port and rl_server.py on 8790.
"""
import argparse
import asyncio
import json
import sys
import time
import urllib.request

from playwright.async_api import async_playwright

ARGS = [
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--autoplay-policy=no-user-gesture-required",
    "--mute-audio",
]


def stats():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8790/stats", timeout=3) as r:
            return json.load(r)
    except Exception as e:
        return {"error": str(e)}


async def probe(page, i):
    try:
        return await page.evaluate(
            "() => { const s = __BREACH__.state; return {r: s.team.round, w: s.team.wins, a: s.team.alive, cap: s.team.cap, u: s.rl.version, ready: s.rl.ready && s.team.auto, t: s.time}; }"
        )
    except Exception as e:
        return {"err": str(e)[:60]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--port", type=int, default=4178)
    ap.add_argument("--minutes", type=float, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("--channel", default="chrome")
    ap.add_argument("--report", type=int, default=60)
    ap.add_argument("--mode", default="tdm", help="tdm or zone")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/?auto={a.mode}"
    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch(headless=True, channel=a.channel, args=ARGS)
        except Exception as e:
            print(f"chrome channel failed ({e}); trying bundled chromium", file=sys.stderr)
            browser = await pw.chromium.launch(headless=True, args=ARGS)
        pages = []
        for i in range(a.games):
            ctx = await browser.new_context(viewport={"width": 640, "height": 360})
            page = await ctx.new_page()
            page.on("pageerror", lambda e, i=i: print(f"[g{i}] pageerror {e}", file=sys.stderr))
            await page.goto(url, wait_until="commit", timeout=120000)
            pages.append(page)
            await asyncio.sleep(6)
        print(f"{a.games} games launched -> {url}", flush=True)
        t0 = time.time()
        base = stats().get("transitions", 0)
        try:
            while a.minutes <= 0 or time.time() - t0 < a.minutes * 60:
                await asyncio.sleep(a.report)
                st = stats()
                rows = await asyncio.gather(*(probe(p, i) for i, p in enumerate(pages)))
                rounds = sum(r.get("r", 0) for r in rows if "r" in r)
                wins = [sum(r["w"][0] for r in rows if "w" in r), sum(r["w"][1] for r in rows if "w" in r)]
                sim = [round(r.get("t", 0)) for r in rows if "t" in r]
                dead = [i for i, r in enumerate(rows) if "err" in r or not r.get("ready")]
                el = int(time.time() - t0)
                print(
                    f"[{el:5d}s] rounds={rounds} ALPHA:BRAVO={wins[0]}:{wins[1]} "
                    f"transitions={st.get('transitions')} (+{st.get('transitions', 0) - base}) "
                    f"updates={st.get('updates')} reward2k={round(st.get('mean_reward_last_2000', 0), 4)} "
                    f"hit500={st.get('ranged_hit_rate_last_500')} sim_s={min(sim) if sim else '?'}..{max(sim) if sim else '?'} "
                    f"stuck={dead}",
                    flush=True,
                )
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
