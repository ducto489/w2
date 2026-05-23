import json
from pathlib import Path

from dqf.logging_utils import load_jsonl
from dqf.semantic_quality import compute_semantic_quality
from scripts.evaluate_semantic_quality import evaluate_semantic_quality_file


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids)


def test_compute_semantic_quality_normalizes_case_and_punctuation():
    metrics = compute_semantic_quality("Hello, world!", "hello world")

    assert metrics["semantic_exact_match"] == 0.0
    assert metrics["semantic_normalized_match"] == 1.0
    assert metrics["semantic_token_f1"] == 1.0
    assert metrics["semantic_length_ratio"] == 1.0


def test_compute_semantic_quality_scores_partial_token_overlap():
    metrics = compute_semantic_quality("draft quantization helps sometimes", "draft quantization hurts")

    assert metrics["semantic_exact_match"] == 0.0
    assert metrics["semantic_normalized_match"] == 0.0
    assert round(metrics["semantic_token_f1"], 4) == 0.5714
    assert round(metrics["semantic_length_ratio"], 4) == 1.3333


def test_evaluate_semantic_quality_file_backfills_text_metrics(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    baseline_path = tmp_path / "baseline.json"
    raw_path.write_text(
        json.dumps(
            {
                "prompt_id": "chat-0",
                "generated_tokens": [1, 2, 3],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_path.write_text(json.dumps({"chat-0": [1, 2, 4]}), encoding="utf-8")

    updated = evaluate_semantic_quality_file(raw_path, baseline_path, FakeTokenizer())

    assert updated == 1
    records = load_jsonl(raw_path)
    assert records[0]["generated_text"] == "1 2 3"
    assert records[0]["baseline_text"] == "1 2 4"
    assert round(records[0]["semantic_token_f1"], 4) == 0.6667
    assert records[0]["semantic_length_ratio"] == 1.0
