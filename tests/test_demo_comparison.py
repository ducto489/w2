import csv
import sys
from pathlib import Path

from dqf.demo_comparison import build_demo_comparison_csv
from scripts.build_demo_comparison import parse_args


FIELDNAMES = [
    "task",
    "draft_precision",
    "quant_method",
    "gamma",
    "num_prompts",
    "acceptance_rate_mean",
    "position_1_acceptance",
    "position_2_acceptance",
    "position_3_acceptance",
    "position_4_acceptance",
    "draft_time_ms_mean",
    "total_wall_time_ms_mean",
    "tokens_per_second_mean",
    "quality_mean",
    "semantic_token_f1_mean",
    "semantic_length_ratio_mean",
    "draft_model_generate_ms_per_call",
]


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_build_demo_comparison_csv_labels_hf_and_vllm_rows(tmp_path: Path):
    hf_path = tmp_path / "hf.csv"
    vllm_path = tmp_path / "vllm.csv"
    output_path = tmp_path / "demo.csv"
    write_summary(
        hf_path,
        [
            {
                "task": "chat",
                "draft_precision": "bf16",
                "quant_method": "none",
                "gamma": 4,
                "num_prompts": 10,
                "acceptance_rate_mean": 0.5,
                "position_1_acceptance": 0.8,
                "position_2_acceptance": 0.6,
                "position_3_acceptance": 0.4,
                "position_4_acceptance": 0.2,
                "draft_time_ms_mean": 100.0,
                "total_wall_time_ms_mean": 200.0,
                "tokens_per_second_mean": 8.0,
                "quality_mean": 1.0,
                "semantic_token_f1_mean": 0.95,
                "semantic_length_ratio_mean": 1.0,
                "draft_model_generate_ms_per_call": "",
            },
            {
                "task": "chat",
                "draft_precision": "int4",
                "quant_method": "bnb",
                "gamma": 4,
                "num_prompts": 10,
                "acceptance_rate_mean": 0.4,
                "position_1_acceptance": 0.7,
                "position_2_acceptance": 0.5,
                "position_3_acceptance": 0.3,
                "position_4_acceptance": 0.1,
                "draft_time_ms_mean": 300.0,
                "total_wall_time_ms_mean": 400.0,
                "tokens_per_second_mean": 4.0,
                "quality_mean": 1.0,
                "semantic_token_f1_mean": 0.85,
                "semantic_length_ratio_mean": 0.9,
                "draft_model_generate_ms_per_call": "",
            },
        ],
    )
    write_summary(
        vllm_path,
        [
            {
                "task": "chat",
                "draft_precision": "int4",
                "quant_method": "gptq_marlin",
                "gamma": 4,
                "num_prompts": 10,
                "acceptance_rate_mean": 0.45,
                "position_1_acceptance": 0.75,
                "position_2_acceptance": 0.55,
                "position_3_acceptance": 0.35,
                "position_4_acceptance": 0.15,
                "draft_time_ms_mean": 80.0,
                "total_wall_time_ms_mean": 180.0,
                "tokens_per_second_mean": 12.0,
                "quality_mean": 1.0,
                "semantic_token_f1_mean": 0.9,
                "semantic_length_ratio_mean": 1.1,
                "draft_model_generate_ms_per_call": 20.0,
            }
        ],
    )

    build_demo_comparison_csv(
        hf_summary_paths=[hf_path],
        vllm_summary_paths=[vllm_path],
        output_path=output_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["variant"] for row in rows] == ["HF BF16", "HF bnb INT4", "vLLM GPTQ INT4"]
    assert [row["backend"] for row in rows] == ["hf", "hf", "vllm"]
    assert rows[0]["display_order"] == "1"
    assert rows[1]["display_order"] == "3"
    assert rows[2]["display_order"] == "4"
    assert rows[2]["semantic_token_f1_mean"] == "0.9"
    assert rows[2]["semantic_length_ratio_mean"] == "1.1"
    assert rows[2]["draft_backend_ms_per_call"] == "20.0"


def test_demo_comparison_cli_overrides_default_summary_paths(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_demo_comparison.py",
            "--hf-summary",
            "new_hf_chat.csv",
            "--hf-summary",
            "new_hf_reasoning.csv",
            "--vllm-summary",
            "new_vllm.csv",
        ],
    )

    args = parse_args()

    assert args.hf_summary == [Path("new_hf_chat.csv"), Path("new_hf_reasoning.csv")]
    assert args.vllm_summary == [Path("new_vllm.csv")]
