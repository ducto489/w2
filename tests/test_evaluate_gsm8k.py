import json
from pathlib import Path

from dqf.logging_utils import load_jsonl
from scripts.evaluate_gsm8k import evaluate_gsm8k_file


def test_evaluate_gsm8k_file_backfills_task_exact_match(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    references_path = tmp_path / "references.jsonl"
    raw_records = [
        {"prompt_id": "gsm8k-0", "generated_text": "Reasoning... #### 42"},
        {"prompt_id": "gsm8k-1", "generated_text": "The answer is 19."},
    ]
    reference_records = [
        {"prompt_id": "gsm8k-0", "answer": "42"},
        {"prompt_id": "gsm8k-1", "answer": "18"},
    ]
    raw_path.write_text(
        "\n".join(json.dumps(record) for record in raw_records) + "\n",
        encoding="utf-8",
    )
    references_path.write_text(
        "\n".join(json.dumps(record) for record in reference_records) + "\n",
        encoding="utf-8",
    )

    updated = evaluate_gsm8k_file(raw_path, references_path)

    assert updated == 2
    records = load_jsonl(raw_path)
    assert records[0]["task_answer"] == "42"
    assert records[0]["reference_answer"] == "42"
    assert records[0]["task_exact_match"] == 1.0
    assert records[1]["task_answer"] == "19"
    assert records[1]["reference_answer"] == "18"
    assert records[1]["task_exact_match"] == 0.0


def test_evaluate_gsm8k_file_raises_for_missing_generated_text(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    references_path = tmp_path / "references.jsonl"
    raw_path.write_text(json.dumps({"prompt_id": "gsm8k-0"}) + "\n", encoding="utf-8")
    references_path.write_text(
        json.dumps({"prompt_id": "gsm8k-0", "answer": "42"}) + "\n",
        encoding="utf-8",
    )

    try:
        evaluate_gsm8k_file(raw_path, references_path)
    except ValueError as exc:
        assert "generated_text" in str(exc)
    else:
        raise AssertionError("expected ValueError")
