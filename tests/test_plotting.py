import csv
from pathlib import Path

import pandas as pd
import pytest

from dqf.plotting import generate_all_figures
from dqf.plotting import _position_acceptance_error_bars


def test_generate_all_figures_creates_expected_pngs(tmp_path: Path):
    summary_path = tmp_path / "summary.csv"
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "draft_precision",
                "quant_method",
                "gamma",
                "num_prompts",
                "acceptance_rate_mean",
                "acceptance_rate_std",
                "acceptance_rate_ci_low",
                "acceptance_rate_ci_high",
                "mean_accepted_length_mean",
                "position_1_acceptance",
                "position_1_acceptance_ci_low",
                "position_1_acceptance_ci_high",
                "position_2_acceptance",
                "position_2_acceptance_ci_low",
                "position_2_acceptance_ci_high",
                "position_3_acceptance",
                "position_3_acceptance_ci_low",
                "position_3_acceptance_ci_high",
                "position_4_acceptance",
                "position_4_acceptance_ci_low",
                "position_4_acceptance_ci_high",
                "draft_time_ms_mean",
                "total_wall_time_ms_mean",
                "tokens_per_second_mean",
                "quality_mean",
                "draft_cost_share_mean",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": "chat",
                "draft_precision": "bf16",
                "quant_method": "bnb",
                "gamma": "4",
                "num_prompts": "5",
                "acceptance_rate_mean": "0.75",
                "acceptance_rate_std": "0.05",
                "acceptance_rate_ci_low": "0.70",
                "acceptance_rate_ci_high": "0.80",
                "mean_accepted_length_mean": "2.5",
                "position_1_acceptance": "1.0",
                "position_1_acceptance_ci_low": "1.0",
                "position_1_acceptance_ci_high": "1.0",
                "position_2_acceptance": "1.0",
                "position_2_acceptance_ci_low": "0.9",
                "position_2_acceptance_ci_high": "1.0",
                "position_3_acceptance": "0.75",
                "position_3_acceptance_ci_low": "0.55",
                "position_3_acceptance_ci_high": "0.90",
                "position_4_acceptance": "0.25",
                "position_4_acceptance_ci_low": "0.10",
                "position_4_acceptance_ci_high": "0.45",
                "draft_time_ms_mean": "10.0",
                "total_wall_time_ms_mean": "30.0",
                "tokens_per_second_mean": "40.0",
                "quality_mean": "1.0",
                "draft_cost_share_mean": "0.33",
            }
        )

    figure_paths = generate_all_figures(summary_path, figures_dir)

    assert set(figure_paths) == {
        figures_dir / "position_acceptance_by_precision.png",
        figures_dir / "pareto_acceptance_vs_latency.png",
        figures_dir / "speed_quality_frontier.png",
    }
    for figure_path in figure_paths:
        assert figure_path.exists()
        assert figure_path.stat().st_size > 0


def test_position_acceptance_error_bars_use_ci_columns_when_available():
    row = pd.Series(
        {
            "position_1_acceptance": 0.8,
            "position_1_acceptance_ci_low": 0.7,
            "position_1_acceptance_ci_high": 0.9,
            "position_2_acceptance": 0.6,
            "position_2_acceptance_ci_low": 0.5,
            "position_2_acceptance_ci_high": 0.65,
        }
    )

    lower, upper = _position_acceptance_error_bars(row, [1, 2])
    assert lower == pytest.approx([0.1, 0.1])
    assert upper == pytest.approx([0.1, 0.05])
