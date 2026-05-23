from __future__ import annotations

import argparse
import json
from pathlib import Path

from dqf.logging_utils import load_jsonl, write_jsonl
from dqf.task_quality import extract_gsm8k_answer, normalize_numeric_answer


def _load_references(path: Path) -> dict[str, str]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {
            str(prompt_id): normalize_numeric_answer(answer)
            for prompt_id, answer in payload.items()
        }

    references: dict[str, str] = {}
    for record in load_jsonl(path):
        prompt_id = str(record["prompt_id"])
        if "answer" not in record:
            raise KeyError(f"missing answer for prompt_id={prompt_id}")
        references[prompt_id] = normalize_numeric_answer(record["answer"])
    return references


def evaluate_gsm8k_file(raw_path: Path, references_path: Path) -> int:
    references = _load_references(references_path)
    records = load_jsonl(raw_path)
    updated = 0

    for record in records:
        prompt_id = str(record["prompt_id"])
        if "generated_text" not in record:
            raise ValueError(
                f"missing generated_text for prompt_id={prompt_id}; "
                "run scripts/evaluate_semantic_quality.py first or log decoded text"
            )
        reference_answer = references.get(prompt_id)
        if reference_answer is None:
            raise KeyError(f"missing GSM8K reference answer for prompt_id={prompt_id}")

        task_answer = extract_gsm8k_answer(str(record["generated_text"]))
        record["task_answer"] = task_answer
        record["reference_answer"] = reference_answer
        record["task_exact_match"] = 1.0 if task_answer == reference_answer else 0.0
        updated += 1

    write_jsonl(raw_path, records)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill GSM8K exact-match task quality into raw JSONL records."
    )
    parser.add_argument("--input", required=True, help="Path to a raw JSONL result file.")
    parser.add_argument(
        "--references",
        required=True,
        help="JSON or JSONL mapping prompt_id to GSM8K reference answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_gsm8k_file(Path(args.input), Path(args.references))


if __name__ == "__main__":
    main()
