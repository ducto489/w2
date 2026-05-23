import csv
import json
from pathlib import Path

from scripts.benchmark_vllm_draft import (
    build_latency_record,
    normalize_quantization_backend,
    load_prompt_sets,
    summarize_latency_records,
    write_latency_summary_csv,
)


def test_load_prompt_sets_preserves_task_and_prompt_ids(tmp_path: Path):
    chat_path = tmp_path / "chat.txt"
    reasoning_path = tmp_path / "reasoning.txt"
    chat_path.write_text("hello\n\nworld\n", encoding="utf-8")
    reasoning_path.write_text("think\n", encoding="utf-8")

    prompts = load_prompt_sets(
        [
            ("chat", chat_path),
            ("reasoning", reasoning_path),
        ]
    )

    assert [prompt.prompt_id for prompt in prompts] == ["chat-0", "chat-2", "reasoning-0"]
    assert [prompt.task for prompt in prompts] == ["chat", "chat", "reasoning"]
    assert [prompt.prompt for prompt in prompts] == ["hello", "world", "think"]


def test_normalize_quantization_backend_treats_none_as_unquantized():
    assert normalize_quantization_backend("none") is None
    assert normalize_quantization_backend(None) is None
    assert normalize_quantization_backend("gptq_marlin") == "gptq_marlin"


def test_build_latency_record_preserves_requested_and_actual_batch_size():
    prompt = load_prompt_sets.__annotations__  # keep imports simple for this module
    from dqf.tasks import PromptExample

    record = build_latency_record(
        prompt=PromptExample(prompt_id="chat-0", task="chat", prompt="hi"),
        request_output=object(),
        model_name="model",
        quantization_backend="none",
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        requested_batch_size=32,
        actual_batch_size=2,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        batch_output_tokens=0,
        latency_ms=100.0,
        tokens_per_second=0.0,
    )

    assert prompt is not None
    assert record["batch_size"] == 32
    assert record["actual_batch_size"] == 2


def test_summarize_latency_records_groups_by_task_and_backend():
    records = [
        {
            "task": "chat",
            "backend": "vllm",
            "quantization_backend": "gptq_marlin",
            "batch_size": 1,
            "latency_ms": 100.0,
            "tokens_per_second": 20.0,
        },
        {
            "task": "chat",
            "backend": "vllm",
            "quantization_backend": "gptq_marlin",
            "batch_size": 1,
            "latency_ms": 300.0,
            "tokens_per_second": 10.0,
        },
        {
            "task": "reasoning",
            "backend": "vllm",
            "quantization_backend": "gptq_marlin",
            "batch_size": 4,
            "latency_ms": 200.0,
            "tokens_per_second": 40.0,
        },
    ]

    summaries = summarize_latency_records(records)

    assert summaries == [
        {
            "task": "chat",
            "backend": "vllm",
            "quantization_backend": "gptq_marlin",
            "batch_size": 1,
            "num_prompts": 2,
            "latency_ms_mean": 200.0,
            "latency_ms_p50": 200.0,
            "latency_ms_p95": 290.0,
            "tokens_per_second_mean": 15.0,
        },
        {
            "task": "reasoning",
            "backend": "vllm",
            "quantization_backend": "gptq_marlin",
            "batch_size": 4,
            "num_prompts": 1,
            "latency_ms_mean": 200.0,
            "latency_ms_p50": 200.0,
            "latency_ms_p95": 200.0,
            "tokens_per_second_mean": 40.0,
        },
    ]


def test_write_latency_summary_csv_round_trips_raw_jsonl(tmp_path: Path):
    raw_path = tmp_path / "raw" / "vllm_gptq_marlin.jsonl"
    summary_path = tmp_path / "summary.csv"
    records = [
        {
            "task": "chat",
            "backend": "vllm",
            "quantization_backend": "gptq_marlin",
            "batch_size": 1,
            "latency_ms": 100.0,
            "tokens_per_second": 20.0,
        },
        {
            "task": "chat",
            "backend": "vllm",
            "quantization_backend": "gptq_marlin",
            "batch_size": 1,
            "latency_ms": 300.0,
            "tokens_per_second": 10.0,
        },
    ]
    raw_path.parent.mkdir()
    raw_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    write_latency_summary_csv([raw_path], summary_path)

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["task"] == "chat"
    assert row["backend"] == "vllm"
    assert row["quantization_backend"] == "gptq_marlin"
    assert row["batch_size"] == "1"
    assert row["num_prompts"] == "2"
    assert row["latency_ms_mean"] == "200.0"
    assert row["latency_ms_p95"] == "290.0"
    assert row["tokens_per_second_mean"] == "15.0"
