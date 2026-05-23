from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from dqf.logging_utils import load_jsonl, write_jsonl
from dqf.tasks import PromptExample, load_prompt_examples


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
DEFAULT_QUANTIZATION = "gptq_marlin"
DEFAULT_OUTPUT_DIR = Path("results/raw_vllm_draft_latency_2026-05-16")
DEFAULT_SUMMARY_PATH = Path("results/processed/summary_vllm_draft_latency_2026-05-16.csv")
SUMMARY_FIELDS = [
    "task",
    "backend",
    "quantization_backend",
    "batch_size",
    "num_prompts",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p95",
    "tokens_per_second_mean",
]


def load_prompt_sets(prompt_sets: list[tuple[str, Path]]) -> list[PromptExample]:
    prompts: list[PromptExample] = []
    for task, prompt_path in prompt_sets:
        prompts.extend(load_prompt_examples(prompt_path, task))
    return prompts


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = rank - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def summarize_latency_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["task"]),
            str(record["backend"]),
            str(record["quantization_backend"]),
            int(record.get("batch_size", 1)),
        )
        grouped.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        task, backend, quantization_backend, batch_size = key
        latencies = [float(record["latency_ms"]) for record in group]
        token_rates = [float(record["tokens_per_second"]) for record in group]
        rows.append(
            {
                "task": task,
                "backend": backend,
                "quantization_backend": quantization_backend,
                "batch_size": batch_size,
                "num_prompts": len(group),
                "latency_ms_mean": mean(latencies) if latencies else 0.0,
                "latency_ms_p50": percentile(latencies, 0.50),
                "latency_ms_p95": percentile(latencies, 0.95),
                "tokens_per_second_mean": mean(token_rates) if token_rates else 0.0,
            }
        )
    return rows


def write_latency_summary_csv(input_paths: list[Path], output_path: Path) -> Path:
    records: list[dict[str, Any]] = []
    for input_path in input_paths:
        records.extend(load_jsonl(input_path))
    rows = summarize_latency_records(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def synchronize_cuda_if_available() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def output_token_ids(request_output: object) -> list[int]:
    outputs = getattr(request_output, "outputs", None)
    if not outputs:
        return []
    token_ids = getattr(outputs[0], "token_ids", None)
    if token_ids is None:
        return []
    return [int(token_id) for token_id in token_ids]


def output_text(request_output: object) -> str:
    outputs = getattr(request_output, "outputs", None)
    if not outputs:
        return ""
    return str(getattr(outputs[0], "text", ""))


def normalize_quantization_backend(quantization_backend: str | None) -> str | None:
    if quantization_backend is None or quantization_backend == "none":
        return None
    return quantization_backend


def build_latency_record(
    *,
    prompt: PromptExample,
    request_output: object,
    model_name: str,
    quantization_backend: str,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    requested_batch_size: int,
    actual_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_output_tokens: int,
    latency_ms: float,
    tokens_per_second: float,
) -> dict[str, Any]:
    token_ids = output_token_ids(request_output)
    return {
        "prompt_id": prompt.prompt_id,
        "task": prompt.task,
        "backend": "vllm",
        "model_name": model_name,
        "quantization_backend": quantization_backend,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "batch_size": requested_batch_size,
        "actual_batch_size": actual_batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_chars": len(prompt.prompt),
        "output_tokens": len(token_ids),
        "batch_output_tokens": batch_output_tokens,
        "latency_ms": latency_ms,
        "tokens_per_second": tokens_per_second,
        "generated_token_ids": token_ids,
        "output_text": output_text(request_output),
        "notes": "draft-only vLLM latency; no target verification or acceptance logging",
    }


def benchmark_vllm_draft(
    *,
    prompts: list[PromptExample],
    model_name: str,
    quantization_backend: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
    }
    normalized_quantization = normalize_quantization_backend(quantization_backend)
    if normalized_quantization is not None:
        llm_kwargs["quantization"] = normalized_quantization
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    records: list[dict[str, Any]] = []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for start_index in range(0, len(prompts), batch_size):
        batch = prompts[start_index : start_index + batch_size]
        synchronize_cuda_if_available()
        start_time = perf_counter()
        outputs = llm.generate([prompt.prompt for prompt in batch], sampling_params)
        synchronize_cuda_if_available()
        latency_ms = (perf_counter() - start_time) * 1000.0
        batch_output_tokens = sum(len(output_token_ids(output)) for output in outputs)
        tokens_per_second = (
            batch_output_tokens / (latency_ms / 1000.0) if latency_ms > 0.0 else 0.0
        )
        for prompt, request_output in zip(batch, outputs):
            records.append(
                build_latency_record(
                    prompt=prompt,
                    request_output=request_output,
                    model_name=model_name,
                    quantization_backend=quantization_backend,
                    dtype=dtype,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    requested_batch_size=batch_size,
                    actual_batch_size=len(batch),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    batch_output_tokens=batch_output_tokens,
                    latency_ms=latency_ms,
                    tokens_per_second=tokens_per_second,
                )
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run draft-only vLLM GPTQ Marlin latency on project prompt fixtures."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    parser.add_argument("--chat-prompts", type=Path, default=Path("configs/prompts_chat_extended.txt"))
    parser.add_argument(
        "--reasoning-prompts",
        type=Path,
        default=Path("configs/prompts_reasoning_extended.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=None,
        help="optional smoke limit for each task before running the full fixtures",
    )
    return parser.parse_args()


def apply_limit_per_task(
    prompts: list[PromptExample],
    limit_per_task: int | None,
) -> list[PromptExample]:
    if limit_per_task is None:
        return prompts
    seen: dict[str, int] = {}
    selected: list[PromptExample] = []
    for prompt in prompts:
        count = seen.get(prompt.task, 0)
        if count >= limit_per_task:
            continue
        selected.append(prompt)
        seen[prompt.task] = count + 1
    return selected


def main() -> None:
    args = parse_args()
    prompts = load_prompt_sets(
        [
            ("chat", args.chat_prompts),
            ("reasoning", args.reasoning_prompts),
        ]
    )
    prompts = apply_limit_per_task(prompts, args.limit_per_task)
    records = benchmark_vllm_draft(
        prompts=prompts,
        model_name=args.model,
        quantization_backend=args.quantization,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )
    output_path = args.output_dir / f"vllm_{args.quantization}_bs{args.batch_size}.jsonl"
    write_jsonl(output_path, records)
    write_latency_summary_csv([output_path], args.summary_output)


if __name__ == "__main__":
    main()
