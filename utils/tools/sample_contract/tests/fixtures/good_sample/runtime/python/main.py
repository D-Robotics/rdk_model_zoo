"""Fixture entrypoint for the checker positive case.

``build_parser`` is side-effect free: importing this module performs no SDK
load, no download, and no board access.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the fixture command line parser."""

    parser = argparse.ArgumentParser(description="good_sample fixture")
    parser.add_argument(
        "--target",
        choices=("auto", "x5", "s100"),
        default="auto",
        help="Execution target.",
    )
    parser.add_argument(
        "--top-k",
        "--topk",
        dest="top_k",
        type=int,
        default=5,
        help="Number of results (default: 5).",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Scheduling priority (default: 0).",
    )
    parser.add_argument(
        "--bpu-cores",
        nargs="+",
        type=int,
        default=[0],
        help="BPU core indexes (default: 0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional score threshold.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the fixture (no-op for checker purposes)."""

    build_parser().parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
