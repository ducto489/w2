from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def recommend_gamma_policy(
    summary_path: Path,
    output_path: Path,
    *,
    acceptance_floor: float,
) -> Path:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["task"], row["draft_precision"], row["quant_method"]),
            [],
        ).append(row)

    policy = []
    for key, group in sorted(grouped.items()):
        candidates = [
            row for row in group if float(row["acceptance_rate_mean"]) >= acceptance_floor
        ]
        if not candidates:
            candidates = group
        selected = max(candidates, key=lambda row: float(row["tokens_per_second_mean"]))
        task, draft_precision, quant_method = key
        policy.append(
            {
                "task": task,
                "draft_precision": draft_precision,
                "quant_method": quant_method,
                "gamma": int(selected["gamma"]),
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
    parser = argparse.ArgumentParser(description="Recommend gamma by task and draft precision.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--acceptance-floor", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recommend_gamma_policy(
        args.summary,
        args.output,
        acceptance_floor=args.acceptance_floor,
    )


if __name__ == "__main__":
    main()
