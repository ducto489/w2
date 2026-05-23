import csv
import json
from pathlib import Path

from dqf.aggregate import aggregate_jsonl_to_csv


def test_aggregate_jsonl_to_csv_summarizes_expected_columns(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    summary_path = tmp_path / "summary.csv"
    records = [
        {
            "task": "chat",
            "draft_precision": "bf16",
            "quant_method": "bnb",
            "gamma": 4,
            "acceptance_rate": 0.5,
            "mean_accepted_length": 2.0,
            "position_acceptance": [1.0, 1.0, 0.0, 0.0],
            "draft_time_ms": 10.0,
            "verify_time_ms": 15.0,
            "total_wall_time_ms": 25.0,
            "tokens_per_second": 20.0,
            "quality_score": 1.0,
            "task_exact_match": 1.0,
            "semantic_token_f1": 0.5,
            "semantic_length_ratio": 1.0,
            "notes": "",
        },
        {
            "task": "chat",
            "draft_precision": "bf16",
            "quant_method": "bnb",
            "gamma": 4,
            "acceptance_rate": 0.75,
            "mean_accepted_length": 3.0,
            "position_acceptance": [1.0, 1.0, 1.0, 0.0],
            "draft_time_ms": 20.0,
            "verify_time_ms": 20.0,
            "total_wall_time_ms": 40.0,
            "tokens_per_second": 25.0,
            "quality_score": 1.0,
            "task_exact_match": 0.0,
            "semantic_token_f1": 1.0,
            "semantic_length_ratio": 0.5,
            "notes": "",
        },
    ]
    raw_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    aggregate_jsonl_to_csv([raw_path], summary_path)

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["task"] == "chat"
    assert row["draft_precision"] == "bf16"
    assert row["num_prompts"] == "2"
    assert row["acceptance_rate_mean"] == "0.625"
    assert row["acceptance_rate_ci_low"] != ""
    assert row["acceptance_rate_ci_high"] != ""
    assert float(row["acceptance_rate_ci_low"]) <= 0.625 <= float(row["acceptance_rate_ci_high"])
    assert row["position_3_acceptance"] == "0.5"
    assert row["position_3_acceptance_ci_low"] != ""
    assert row["position_3_acceptance_ci_high"] != ""
    assert row["semantic_token_f1_mean"] == "0.75"
    assert row["semantic_length_ratio_mean"] == "0.75"
    assert row["task_exact_match_mean"] == "0.5"
    assert row["draft_cost_share_mean"] == "0.45"


def test_aggregate_jsonl_to_csv_summarizes_optional_profile_columns(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    summary_path = tmp_path / "summary.csv"
    records = [
        {
            "task": "chat",
            "draft_precision": "int8",
            "quant_method": "bnb",
            "gamma": 4,
            "acceptance_rate": 0.5,
            "mean_accepted_length": 2.0,
            "position_acceptance": [1.0, 1.0, 0.0, 0.0],
            "draft_time_ms": 100.0,
            "verify_time_ms": 50.0,
            "total_wall_time_ms": 160.0,
            "tokens_per_second": 10.0,
            "quality_score": 1.0,
            "notes": "",
            "draft_profile": {
                "forward_calls": 4.0,
                "extend_cache_calls": 3.0,
                "cache_reset_calls": 1.0,
                "input_tokens_total": 7.0,
                "model_forward_time_ms": 80.0,
            },
            "target_profile": {
                "forward_calls": 2.0,
                "model_forward_time_ms": 30.0,
            },
        }
    ]
    raw_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    aggregate_jsonl_to_csv([raw_path], summary_path)

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["draft_forward_calls_mean"] == "4.0"
    assert row["draft_model_forward_time_ms_mean"] == "80.0"
    assert row["draft_model_forward_ms_per_call"] == "20.0"
    assert row["draft_input_tokens_per_forward"] == "1.75"
    assert row["draft_extend_cache_ratio"] == "0.75"
    assert row["draft_cache_reset_calls_mean"] == "1.0"
    assert row["target_model_forward_ms_per_call"] == "15.0"


def test_aggregate_jsonl_to_csv_summarizes_vllm_draft_profile_columns(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    summary_path = tmp_path / "summary.csv"
    records = [
        {
            "task": "chat",
            "draft_precision": "int4",
            "quant_method": "gptq_marlin",
            "gamma": 4,
            "acceptance_rate": 0.25,
            "mean_accepted_length": 1.0,
            "position_acceptance": [0.5, 0.25, 0.0, 0.0],
            "draft_time_ms": 100.0,
            "verify_time_ms": 50.0,
            "total_wall_time_ms": 160.0,
            "tokens_per_second": 10.0,
            "quality_score": 1.0,
            "notes": "",
            "draft_profile": {
                "propose_calls": 4.0,
                "generated_tokens": 16.0,
                "model_generate_time_ms": 80.0,
            },
        }
    ]
    raw_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    aggregate_jsonl_to_csv([raw_path], summary_path)

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["draft_propose_calls_mean"] == "4.0"
    assert row["draft_model_generate_time_ms_mean"] == "80.0"
    assert row["draft_model_generate_ms_per_call"] == "20.0"
    assert row["draft_generated_tokens_per_call"] == "4.0"
