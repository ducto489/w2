from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from dqf.task_quality import extract_gsm8k_answer


def build_gsm8k_fixture_rows(
    dataset_rows: Iterable[dict],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, str]]:
    rows = list(dataset_rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected = rows[:limit]
    fixture_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected):
        answer = extract_gsm8k_answer(str(row["answer"]))
        if answer is None:
            raise ValueError(f"could not extract GSM8K answer from sampled row {index}")
        fixture_rows.append(
            {
                "prompt_id": f"gsm8k-{index}",
                "prompt": str(row["question"]).strip(),
                "answer": answer,
            }
        )
    return fixture_rows


def write_gsm8k_fixtures(
    rows: list[dict[str, str]],
    prompts_path: Path,
    references_path: Path,
) -> None:
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    references_path.parent.mkdir(parents=True, exist_ok=True)

    with prompts_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row["prompt"].replace("\n", " ").strip() + "\n")

    with references_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {"prompt_id": row["prompt_id"], "answer": row["answer"]},
                    ensure_ascii=True,
                )
                + "\n"
            )


def load_gsm8k_dataset(split: str) -> object:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required to build GSM8K fixtures; install it with "
            "`uv pip install datasets` in the active environment"
        ) from exc
    return load_dataset("gsm8k", "main", split=split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic GSM8K prompt and reference fixtures."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=Path("configs/prompts_gsm8k_200.txt"),
    )
    parser.add_argument(
        "--references-output",
        type=Path,
        default=Path("configs/references_gsm8k_200.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_gsm8k_fixture_rows(
        load_gsm8k_dataset(args.split),
        limit=args.limit,
        seed=args.seed,
    )
    write_gsm8k_fixtures(rows, args.prompts_output, args.references_output)


if __name__ == "__main__":
    main()
