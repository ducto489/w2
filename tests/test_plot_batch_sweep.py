import csv
from pathlib import Path

from scripts.plot_batch_sweep import plot_batch_sweep


def test_plot_batch_sweep_creates_throughput_figure(tmp_path: Path):
    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "figures"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "backend",
                "quantization_backend",
                "batch_size",
                "tokens_per_second_mean",
            ],
        )
        writer.writeheader()
        for batch_size in [1, 4]:
            writer.writerow(
                {
                    "task": "chat",
                    "backend": "vllm",
                    "quantization_backend": "gptq_marlin",
                    "batch_size": batch_size,
                    "tokens_per_second_mean": 100.0 * batch_size,
                }
            )

    figure_path = plot_batch_sweep(summary_path, output_dir)

    assert figure_path == output_dir / "draft_throughput_vs_batch.png"
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
