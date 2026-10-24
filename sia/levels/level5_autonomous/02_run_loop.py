"""Level 5 · The loop runs itself.

Every cycle: read production signal -> check the judge still agrees with people -> grow the
dataset -> propose a change (GEPA, the analyst, or OpenClaw editing itself) -> gate -> ship or
reject -> watch the shipped version and roll back if it does worse live.

For the demo, --simulate sends traffic and has the simulated reviewers label half of it before
each cycle. --drift mixes in (from cycle 2) a new kind of question (see scenario/cases.yaml, split: drift) that
the current instructions don't handle, so you can watch the loop notice, fix, and ship.

    python levels/level5_autonomous/02_run_loop.py --simulate 16 --drift --cycles 4
    python levels/level5_autonomous/02_run_loop.py --improver openclaw --simulate 16 --drift
    python levels/level5_autonomous/02_run_loop.py --interval 1800            # run forever, every 30 min
    python levels/level5_autonomous/02_run_loop.py --no-auto-promote          # stop at "awaiting approval"
"""

import argparse
import time

from sia import tracking, traffic
from sia.loop import LoopConfig, cycle

parser = argparse.ArgumentParser()
parser.add_argument("--improver", default="gepa", choices=["gepa", "analyst", "openclaw"])
parser.add_argument("--cycles", type=int, default=1)
parser.add_argument("--interval", type=int, default=0, help="seconds between cycles; 0 = back to back")
parser.add_argument("--simulate", type=int, default=0, help="send N simulated questions before each cycle")
parser.add_argument("--drift", action="store_true", help="include the new 'drift' questions in the traffic")
parser.add_argument("--drift-from", type=int, default=2, help="first cycle in which drift questions appear")
parser.add_argument("--window-hours", type=float, default=24)
parser.add_argument("--no-auto-promote", action="store_true")
args = parser.parse_args()

cfg = LoopConfig(improver=args.improver, window_hours=args.window_hours, auto_promote=not args.no_auto_promote)
tracking.setup()
i = 0
while args.interval or i < args.cycles:
    i += 1
    print(f"\n======== cycle {i} ========")
    if args.simulate:
        splits = ("traffic", "drift") if args.drift and i >= args.drift_from else ("traffic",)
        ids = traffic.send(splits, n=args.simulate, seed=i)
        print(f"[traffic] {len(ids)} questions ({'+'.join(splits)}); "
              f"{traffic.label(ids, rate=0.5, seed=i)} labeled by reviewers")
    print("=>", cycle(cfg))
    if args.interval:
        time.sleep(args.interval)
