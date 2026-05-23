from __future__ import annotations

import argparse
import json
from pathlib import Path

from dqf.logging_utils import load_jsonl, write_jsonl


def _load_baseline_tokens(path: Path) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(prompt_id): [int(token) for token in tokens] for prompt_id, tokens in payload.items()}


def evaluate_quality_file(raw_path: Path, baseline_path: Path) -> int:
    baseline_by_prompt = _load_baseline_tokens(baseline_path)
    records = load_jsonl(raw_path)
    updated = 0

    for record in records:
        prompt_id = record["prompt_id"]
        generated_tokens = [int(token) for token in record.get("generated_tokens", [])]
        baseline_tokens = baseline_by_prompt.get(prompt_id)
        if baseline_tokens is None:
            raise KeyError(f"missing baseline tokens for prompt_id={prompt_id}")
        record["quality_score"] = 1.0 if generated_tokens == baseline_tokens else 0.0
        updated += 1

    write_jsonl(raw_path, records)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a raw JSONL result file.")
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to a JSON file mapping prompt_id to target-only generated token IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_quality_file(Path(args.input), Path(args.baseline))


if __name__ == "__main__":
    main()
