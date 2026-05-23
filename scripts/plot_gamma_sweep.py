from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_gamma_sweep(summary_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_path)
    throughput_path = output_dir / "gamma_throughput_by_precision.png"
    acceptance_path = output_dir / "gamma_acceptance_by_precision.png"
    _plot_metric(
        df,
        metric="tokens_per_second",
        ylabel="Tokens / second",
        title="Throughput vs gamma",
        output_path=throughput_path,
    )
    _plot_metric(
        df,
        metric="acceptance_rate",
        ylabel="Acceptance rate",
        title="Acceptance vs gamma",
        output_path=acceptance_path,
    )
    return [throughput_path, acceptance_path]


def _plot_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(sorted(df["task"].unique())), figsize=(12, 5), squeeze=False)
    for ax, task in zip(axes[0], sorted(df["task"].unique())):
        task_df = df[df["task"] == task]
        for precision in sorted(task_df["draft_precision"].unique()):
            rows = task_df[task_df["draft_precision"] == precision].sort_values("gamma")
            x_values = rows["gamma"].astype(int).tolist()
            y_values = rows[f"{metric}_mean"].astype(float).tolist()
            yerr = _metric_error_bars(rows, metric)
            ax.errorbar(x_values, y_values, yerr=yerr, marker="o", capsize=3, label=precision)
        ax.set_title(task)
        ax.set_xlabel("Gamma")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(task_df["gamma"].astype(int).unique()))
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _metric_error_bars(rows: pd.DataFrame, metric: str) -> tuple[list[float], list[float]] | None:
    low_column = f"{metric}_ci_low"
    high_column = f"{metric}_ci_high"
    mean_column = f"{metric}_mean"
    if low_column not in rows or high_column not in rows:
        return None
    means = rows[mean_column].astype(float)
    lows = rows[low_column].astype(float)
    highs = rows[high_column].astype(float)
    return (means - lows).clip(lower=0.0).tolist(), (highs - means).clip(lower=0.0).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gamma sweep summary figures.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_gamma_sweep(args.summary, args.figures_dir)


if __name__ == "__main__":
    main()
