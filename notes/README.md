# Public Notes

This directory keeps only public-facing research context that is useful beside the code.
Detailed private progress logs and local run ledgers are intentionally excluded.

## Current Framing

- The headline comparison should use the matched `max_new_tokens=256` runs, not the
  earlier 16-token pilot.
- At 256 tokens, GPTQ INT8/INT4 draft quantization gives a modest throughput gain in
  this single-request setup, roughly `5-9%` on the current 30-prompt chat/reasoning
  fixtures.
- INT8 preserves BF16-like acceptance more cleanly than INT4 in the current pilot.
- The result is engineering evidence, not a final Pareto-frontier claim: prompt count,
  batch size, quality coverage, and QSPEC reproduction remain open.

## Reproduction Pointers

Primary configs:

- `configs/run_bf16_qwen_chat_extended_256_2026-05-21.yaml`
- `configs/run_bf16_qwen_reasoning_extended_256_2026-05-21.yaml`

Primary artifacts:

- Raw chat logs: `results/raw_vllm_sd_chat_precision_256_2026-05-21/`
- Raw reasoning logs: `results/raw_vllm_sd_reasoning_precision_256_2026-05-21/`
- Summary CSV: `results/processed/summary_vllm_sd_precision_256_2026-05-21.csv`
- Figures: `results/figures_vllm_sd_precision_256_2026-05-21/`

Representative commands:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. .venv-vllm-cu121/bin/python scripts/run_sd_vllm_draft.py --config configs/run_bf16_qwen_chat_extended_256_2026-05-21.yaml --draft-precision bf16 --profile --ignore-eos --gpu-memory-utilization 0.50 --dtype bfloat16 --target-torch-dtype bf16
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. .venv-vllm-cu121/bin/python scripts/run_sd_vllm_draft.py --config configs/run_bf16_qwen_chat_extended_256_2026-05-21.yaml --draft-precision int8 --profile --ignore-eos --gpu-memory-utilization 0.50 --target-torch-dtype bf16
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. .venv-vllm-cu121/bin/python scripts/run_sd_vllm_draft.py --config configs/run_bf16_qwen_chat_extended_256_2026-05-21.yaml --draft-precision int4 --profile --ignore-eos --gpu-memory-utilization 0.50 --target-torch-dtype bf16

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. .venv-vllm-cu121/bin/python scripts/run_sd_vllm_draft.py --config configs/run_bf16_qwen_reasoning_extended_256_2026-05-21.yaml --draft-precision bf16 --profile --ignore-eos --gpu-memory-utilization 0.50 --dtype bfloat16 --target-torch-dtype bf16
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. .venv-vllm-cu121/bin/python scripts/run_sd_vllm_draft.py --config configs/run_bf16_qwen_reasoning_extended_256_2026-05-21.yaml --draft-precision int8 --profile --ignore-eos --gpu-memory-utilization 0.50 --target-torch-dtype bf16
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. .venv-vllm-cu121/bin/python scripts/run_sd_vllm_draft.py --config configs/run_bf16_qwen_reasoning_extended_256_2026-05-21.yaml --draft-precision int4 --profile --ignore-eos --gpu-memory-utilization 0.50 --target-torch-dtype bf16

PYTHONPATH=src:. .venv/bin/python scripts/aggregate_results.py --input results/raw_vllm_sd_chat_precision_256_2026-05-21 results/raw_vllm_sd_reasoning_precision_256_2026-05-21 --output results/processed/summary_vllm_sd_precision_256_2026-05-21.csv
PYTHONPATH=src:. .venv/bin/python scripts/plot_results.py --summary results/processed/summary_vllm_sd_precision_256_2026-05-21.csv --figures-dir results/figures_vllm_sd_precision_256_2026-05-21
```

## Open Work

- Scale chat/reasoning fixtures beyond 30 prompts before making stable task-level claims.
- Complete a valid QSPEC baseline or frame QSPEC only as external related work.
- Expand task-quality evaluation beyond exact-token overlap and the current GSM8K pilot.
- Add a batched speculative verification path before comparing against serving-oriented
  QSPEC throughput regimes.
