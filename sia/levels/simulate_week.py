"""Between levels: send another "week" of production traffic and have reviewers label some of it.

    python levels/simulate_week.py [--n 24] [--drift] [--label-rate 0.5]

--drift mixes in questions about the new support_process policy (escalations, discounts).
"""

import argparse

from sia import tracking, traffic

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=24)
parser.add_argument("--drift", action="store_true")
parser.add_argument("--label-rate", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

tracking.setup()
ids = traffic.send(("traffic", "drift") if args.drift else ("traffic",), n=args.n, seed=args.seed)
n = traffic.label(ids, rate=args.label_rate, seed=args.seed)
print(f"{len(ids)} production traces, {n} labeled by reviewers")
