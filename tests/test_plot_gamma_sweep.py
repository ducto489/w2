import csv
from pathlib import Path

from scripts.plot_gamma_sweep import plot_gamma_sweep


def test_plot_gamma_sweep_creates_throughput_and_acceptance_figures(tmp_path: Path):
    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "figures"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "draft_precision",
                "quant_method",
                "gamma",
                "tokens_per_second_mean",
                "tokens_per_second_ci_low",
                "tokens_per_second_ci_high",
                "acceptance_rate_mean",
                "acceptance_rate_ci_low",
                "acceptance_rate_ci_high",
            ],
        )
        writer.writeheader()
        for gamma in [2, 3]:
            writer.writerow(
                {
                    "task": "chat",
                    "draft_precision": "int4",
                    "quant_method": "gptq_marlin",
                    "gamma": gamma,
                    "tokens_per_second_mean": 20.0 + gamma,
                    "tokens_per_second_ci_low": 19.0 + gamma,
                    "tokens_per_second_ci_high": 21.0 + gamma,
                    "acceptance_rate_mean": 0.5,
                    "acceptance_rate_ci_low": 0.4,
                    "acceptance_rate_ci_high": 0.6,
                }
            )

    figure_paths = plot_gamma_sweep(summary_path, output_dir)

    assert figure_paths == [
        output_dir / "gamma_throughput_by_precision.png",
        output_dir / "gamma_acceptance_by_precision.png",
    ]
    for path in figure_paths:
        assert path.exists()
        assert path.stat().st_size > 0
