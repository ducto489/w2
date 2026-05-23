from __future__ import annotations

import argparse
from pathlib import Path

from dqf.logging_utils import load_jsonl, write_jsonl
from dqf.semantic_quality import compute_semantic_quality
from scripts.evaluate_quality import _load_baseline_tokens


def evaluate_semantic_quality_file(raw_path: Path, baseline_path: Path, tokenizer: object) -> int:
    baseline_by_prompt = _load_baseline_tokens(baseline_path)
    records = load_jsonl(raw_path)
    updated = 0

    for record in records:
        prompt_id = record["prompt_id"]
        generated_tokens = [int(token) for token in record.get("generated_tokens", [])]
        baseline_tokens = baseline_by_prompt.get(prompt_id)
        if baseline_tokens is None:
            raise KeyError(f"missing baseline tokens for prompt_id={prompt_id}")

        generated_text = _decode_tokens(tokenizer, generated_tokens)
        baseline_text = _decode_tokens(tokenizer, baseline_tokens)
        record["generated_text"] = generated_text
        record["baseline_text"] = baseline_text
        record.update(compute_semantic_quality(generated_text, baseline_text))
        updated += 1

    write_jsonl(raw_path, records)
    return updated


def _decode_tokens(tokenizer: object, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer must provide a decode method")
    return str(decode(token_ids, skip_special_tokens=True))


def load_tokenizer(model_name: str) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill decoded-text semantic quality metrics into raw JSONL records."
    )
    parser.add_argument("--input", required=True, help="Path to a raw JSONL result file.")
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to a JSON file mapping prompt_id to target-only generated token IDs.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer name used to decode generated and baseline token IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    evaluate_semantic_quality_file(Path(args.input), Path(args.baseline), tokenizer)


if __name__ == "__main__":
    main()
