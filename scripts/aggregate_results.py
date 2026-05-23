from __future__ import annotations

import argparse
from pathlib import Path

from dqf.aggregate import aggregate_jsonl_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = []
    for input_arg in args.input:
        input_path = Path(input_arg)
        if input_path.is_dir():
            input_paths.extend(sorted(input_path.glob("*.jsonl")))
        else:
            input_paths.append(input_path)
    aggregate_jsonl_to_csv(input_paths, Path(args.output))


if __name__ == "__main__":
    main()
