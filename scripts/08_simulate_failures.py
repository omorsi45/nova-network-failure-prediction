from __future__ import annotations

import argparse
from pathlib import Path

from nova.common.logging import setup_logging
from nova.simulate.local_simulator import SimConfig, generate


def main() -> int:
    setup_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/raw")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--failure_rate_per_day", type=float, default=1.5)
    args = ap.parse_args()

    cfg = SimConfig(seed=args.seed, nodes=args.nodes, days=args.days, failure_rate_per_day=args.failure_rate_per_day)
    paths = generate(Path(args.out), cfg)
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
