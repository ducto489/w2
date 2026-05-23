from pathlib import Path

from scripts.build_gsm8k_fixtures import (
    build_gsm8k_fixture_rows,
    write_gsm8k_fixtures,
)


def test_build_gsm8k_fixture_rows_samples_deterministically_and_extracts_references():
    dataset_rows = [
        {"question": "q0", "answer": "work #### 10"},
        {"question": "q1", "answer": "work #### 11"},
        {"question": "q2", "answer": "work #### 12"},
    ]

    rows = build_gsm8k_fixture_rows(dataset_rows, limit=2, seed=7)

    assert rows == [
        {"prompt_id": "gsm8k-0", "prompt": "q2", "answer": "12"},
        {"prompt_id": "gsm8k-1", "prompt": "q0", "answer": "10"},
    ]


def test_write_gsm8k_fixtures_writes_prompt_txt_and_reference_jsonl(tmp_path: Path):
    rows = [
        {"prompt_id": "gsm8k-0", "prompt": "q0", "answer": "10"},
        {"prompt_id": "gsm8k-1", "prompt": "q1", "answer": "11"},
    ]
    prompts_path = tmp_path / "prompts.txt"
    references_path = tmp_path / "references.jsonl"

    write_gsm8k_fixtures(rows, prompts_path, references_path)

    assert prompts_path.read_text(encoding="utf-8") == "q0\nq1\n"
    assert references_path.read_text(encoding="utf-8") == (
        '{"prompt_id": "gsm8k-0", "answer": "10"}\n'
        '{"prompt_id": "gsm8k-1", "answer": "11"}\n'
    )
