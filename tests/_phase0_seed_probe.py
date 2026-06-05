"""Phase-0 probe: run main() with RNG seeded BEFORE construction.

Master calls set_seed() inside BenchmarkRunner.run(), i.e. AFTER the policy
networks are built in __init__, so weight init uses torch's process-random
default seed and runs are not reproducible. This wrapper pre-seeds the process
(mimicking the proposed Phase-1 fix) WITHOUT modifying any tracked file, to
verify that pre-seeding is sufficient for bitwise run-to-run reproducibility.
"""

from __future__ import annotations

import sys

from src.config.seeding import set_seed


def run(argv: list[str]) -> None:
    set_seed(42, deterministic=True)
    sys.argv = ["main.py"] + argv
    import main

    main.main()


if __name__ == "__main__":
    run(sys.argv[1:])
