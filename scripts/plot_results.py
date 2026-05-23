from __future__ import annotations

import argparse
from pathlib import Path

from dqf.plotting import generate_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--figures-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_all_figures(Path(args.summary), Path(args.figures_dir))


if __name__ == "__main__":
    main()
