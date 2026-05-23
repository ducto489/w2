from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_FIELDS = [
    "display_order",
    "task",
    "variant",
    "backend",
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
    "draft_backend_ms_per_call",
]


DISPLAY_ORDER = {
    "HF BF16": 1,
    "HF bnb INT8": 2,
    "HF bnb INT4": 3,
    "vLLM GPTQ INT4": 4,
}


def build_demo_comparison_csv(
    *,
    hf_summary_paths: list[Path],
    vllm_summary_paths: list[Path],
    output_path: Path,
) -> Path:
    rows: list[dict[str, Any]] = []
    for path in hf_summary_paths:
        rows.extend(_load_rows(path, backend="hf"))
    for path in vllm_summary_paths:
        rows.extend(_load_rows(path, backend="vllm"))

    rows.sort(key=lambda row: (row["task"], int(row["display_order"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def generate_demo_comparison_figure(comparison_path: Path, output_path: Path) -> Path:
    df = pd.read_csv(comparison_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    _grouped_bar(
        axes[0, 0],
        df,
        "acceptance_rate_mean",
        "Acceptance rate",
        "Higher is better",
    )
    _grouped_bar(
        axes[0, 1],
        df,
        "tokens_per_second_mean",
        "Throughput (tokens/s)",
        "Higher is better",
    )
    _grouped_bar(
        axes[1, 0],
        df,
        "draft_time_ms_mean",
        "Draft time per prompt (ms)",
        "Lower is better",
    )
    _plot_position_acceptance(axes[1, 1], df)

    fig.suptitle("Qwen speculative decoding: HF bitsandbytes vs vLLM GPTQ draft", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _load_rows(path: Path, backend: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_comparison_row(row, backend) for row in csv.DictReader(handle)]


def _comparison_row(row: dict[str, str], backend: str) -> dict[str, Any]:
    variant = _variant_label(row, backend)
    return {
        "display_order": DISPLAY_ORDER[variant],
        "task": row["task"],
        "variant": variant,
        "backend": backend,
        "draft_precision": row["draft_precision"],
        "quant_method": row["quant_method"],
        "gamma": row["gamma"],
        "num_prompts": row["num_prompts"],
        "acceptance_rate_mean": row["acceptance_rate_mean"],
        "position_1_acceptance": row["position_1_acceptance"],
        "position_2_acceptance": row["position_2_acceptance"],
        "position_3_acceptance": row["position_3_acceptance"],
        "position_4_acceptance": row["position_4_acceptance"],
        "draft_time_ms_mean": row["draft_time_ms_mean"],
        "total_wall_time_ms_mean": row["total_wall_time_ms_mean"],
        "tokens_per_second_mean": row["tokens_per_second_mean"],
        "quality_mean": row["quality_mean"],
        "semantic_token_f1_mean": row.get("semantic_token_f1_mean", ""),
        "semantic_length_ratio_mean": row.get("semantic_length_ratio_mean", ""),
        "draft_backend_ms_per_call": _draft_backend_ms_per_call(row),
    }


def _variant_label(row: dict[str, str], backend: str) -> str:
    precision = row["draft_precision"].upper()
    quant_method = row["quant_method"]
    if backend == "vllm":
        return "vLLM GPTQ INT4"
    if quant_method == "bnb":
        return f"HF bnb {precision}"
    return f"HF {precision}"


def _draft_backend_ms_per_call(row: dict[str, str]) -> str:
    for key in ["draft_model_generate_ms_per_call", "draft_model_forward_ms_per_call"]:
        value = row.get(key, "")
        if value not in {"", None}:
            return value
    return ""


def _grouped_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
) -> None:
    pivot = df.pivot(index="variant", columns="task", values=column)
    pivot = pivot.reindex(DISPLAY_ORDER.keys())
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="y", alpha=0.25)


def _plot_position_acceptance(ax: plt.Axes, df: pd.DataFrame) -> None:
    columns = [
        "position_1_acceptance",
        "position_2_acceptance",
        "position_3_acceptance",
        "position_4_acceptance",
    ]
    x_values = [1, 2, 3, 4]
    for _, row in df.sort_values(["task", "display_order"]).iterrows():
        y_values = [row[column] for column in columns]
        ax.plot(x_values, y_values, marker="o", label=f'{row["task"]}: {row["variant"]}')
    ax.set_title("Position-wise acceptance")
    ax.set_xlabel("Speculative position")
    ax.set_ylabel("Acceptance probability")
    ax.set_xticks(x_values)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
