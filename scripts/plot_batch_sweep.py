from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_batch_sweep(summary_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_path)
    output_path = output_dir / "draft_throughput_vs_batch.png"

    fig, axes = plt.subplots(1, len(sorted(df["task"].unique())), figsize=(12, 5), squeeze=False)
    for ax, task in zip(axes[0], sorted(df["task"].unique())):
        task_df = df[df["task"] == task]
        for quantization in sorted(task_df["quantization_backend"].unique()):
            rows = task_df[task_df["quantization_backend"] == quantization].sort_values(
                "batch_size"
            )
            ax.plot(
                rows["batch_size"].astype(int),
                rows["tokens_per_second_mean"].astype(float),
                marker="o",
                label=_label_for_quantization(quantization),
            )
        ax.set_title(task)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Draft-only tokens / second")
        ax.set_xticks(sorted(task_df["batch_size"].astype(int).unique()))
        ax.legend()
    fig.suptitle("Draft-only vLLM throughput vs batch size")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _label_for_quantization(quantization: str) -> str:
    if quantization == "none":
        return "bf16"
    if quantization == "gptq":
        return "gptq int8"
    if quantization == "gptq_marlin":
        return "gptq int4"
    return quantization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot draft-only batch sweep throughput.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_batch_sweep(args.summary, args.figures_dir)


if __name__ == "__main__":
    main()
