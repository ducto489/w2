from __future__ import annotations

import argparse
from pathlib import Path

from dqf.demo_comparison import build_demo_comparison_csv, generate_demo_comparison_figure


DEFAULT_HF_SUMMARIES = [
    Path("results/processed/summary_qwen_chat_extended_cache_2026-05-15.csv"),
    Path("results/processed/summary_qwen_reasoning_extended_cache_2026-05-15.csv"),
]
DEFAULT_VLLM_SUMMARIES = [Path("results/processed/summary_vllm_sd_extended_2026-05-16.csv")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build professor-demo comparison CSV and figure.")
    parser.add_argument(
        "--hf-summary",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--vllm-summary",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/processed/demo_backend_comparison_2026-05-16.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/figures_demo_backend_comparison_2026-05-16/backend_comparison.png"),
    )
    args = parser.parse_args()
    if args.hf_summary is None:
        args.hf_summary = DEFAULT_HF_SUMMARIES
    if args.vllm_summary is None:
        args.vllm_summary = DEFAULT_VLLM_SUMMARIES
    return args


def main() -> None:
    args = parse_args()
    build_demo_comparison_csv(
        hf_summary_paths=args.hf_summary,
        vllm_summary_paths=args.vllm_summary,
        output_path=args.output_csv,
    )
    generate_demo_comparison_figure(args.output_csv, args.output_figure)


if __name__ == "__main__":
    main()
