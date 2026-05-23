from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def recommend_precision_policy(
    summary_path: Path,
    output_path: Path,
    *,
    acceptance_floor: float,
) -> Path:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["task"], []).append(row)

    policy = []
    for task, group in sorted(grouped.items()):
        candidates = [
            row for row in group if float(row["acceptance_rate_mean"]) >= acceptance_floor
        ]
        if not candidates:
            candidates = group
        selected = max(candidates, key=lambda row: float(row["tokens_per_second_mean"]))
        policy.append(
            {
                "task": task,
                "draft_precision": selected["draft_precision"],
                "quant_method": selected["quant_method"],
                "acceptance_rate_mean": float(selected["acceptance_rate_mean"]),
                "tokens_per_second_mean": float(selected["tokens_per_second_mean"]),
                "acceptance_floor": acceptance_floor,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"policy": policy}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend draft precision by task.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--acceptance-floor", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recommend_precision_policy(
        args.summary,
        args.output,
        acceptance_floor=args.acceptance_floor,
    )


if __name__ == "__main__":
    main()
