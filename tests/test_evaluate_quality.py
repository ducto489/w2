import json
from pathlib import Path

from dqf.logging_utils import load_jsonl
from scripts.evaluate_quality import evaluate_quality_file


def test_evaluate_quality_file_scores_match_against_baseline_tokens(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    baseline_path = tmp_path / "baseline.json"
    raw_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_id": "chat-0",
                        "generated_tokens": [1, 2, 3],
                        "quality_score": None,
                    }
                ),
                json.dumps(
                    {
                        "prompt_id": "chat-1",
                        "generated_tokens": [7, 8, 9],
                        "quality_score": None,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(
            {
                "chat-0": [1, 2, 3],
                "chat-1": [7, 8, 0],
            }
        ),
        encoding="utf-8",
    )

    updated = evaluate_quality_file(raw_path, baseline_path)

    assert updated == 2
    records = load_jsonl(raw_path)
    assert records[0]["quality_score"] == 1.0
    assert records[1]["quality_score"] == 0.0
