from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def generate_all_figures(summary_path: Path, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_path)

    position_path = figures_dir / "position_acceptance_by_precision.png"
    pareto_path = figures_dir / "pareto_acceptance_vs_latency.png"
    speed_quality_path = figures_dir / "speed_quality_frontier.png"

    _plot_position_acceptance(df, position_path)
    _plot_pareto(df, pareto_path)
    _plot_speed_quality(df, speed_quality_path)

    return [position_path, pareto_path, speed_quality_path]


def _plot_position_acceptance(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x_values = [1, 2, 3, 4]
    columns = [_position_acceptance_column(position) for position in x_values]
    for _, row in df.iterrows():
        y_values = [row[column] for column in columns]
        label = f'{row["task"]}:{row["draft_precision"]}'
        error_bars = _position_acceptance_error_bars(row, x_values)
        if error_bars is None:
            ax.plot(x_values, y_values, marker="o", label=label)
        else:
            ax.errorbar(x_values, y_values, yerr=error_bars, marker="o", capsize=3, label=label)
    ax.set_xlabel("Speculative position")
    ax.set_ylabel("Acceptance probability")
    ax.set_title("Position-wise acceptance by precision")
    ax.set_xticks(x_values)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _position_acceptance_column(position: int) -> str:
    return f"position_{position}_acceptance"


def _position_acceptance_error_bars(
    row: pd.Series,
    positions: list[int],
) -> tuple[list[float], list[float]] | None:
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for position in positions:
        mean_column = _position_acceptance_column(position)
        low_column = f"{mean_column}_ci_low"
        high_column = f"{mean_column}_ci_high"
        if low_column not in row or high_column not in row:
            return None
        value = float(row[mean_column])
        lower_errors.append(max(0.0, value - float(row[low_column])))
        upper_errors.append(max(0.0, float(row[high_column]) - value))
    return lower_errors, upper_errors


def _plot_pareto(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        ax.scatter(row["draft_time_ms_mean"], row["acceptance_rate_mean"], label=row["draft_precision"])
        ax.annotate(row["draft_precision"], (row["draft_time_ms_mean"], row["acceptance_rate_mean"]))
    ax.set_xlabel("Draft latency (ms)")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Pareto acceptance vs latency")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_speed_quality(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        ax.scatter(row["tokens_per_second_mean"], row["quality_mean"], label=row["draft_precision"])
        ax.annotate(row["draft_precision"], (row["tokens_per_second_mean"], row["quality_mean"]))
    ax.set_xlabel("Tokens / second")
    ax.set_ylabel("Quality")
    ax.set_title("Speed-quality frontier")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
