import csv
import json
from pathlib import Path

from scripts.recommend_gamma_policy import recommend_gamma_policy
from scripts.recommend_precision_policy import recommend_precision_policy


def test_recommend_gamma_policy_selects_fastest_gamma_above_acceptance_floor(tmp_path: Path):
    summary_path = tmp_path / "gamma.csv"
    output_path = tmp_path / "policy.json"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "draft_precision",
                "quant_method",
                "gamma",
                "acceptance_rate_mean",
                "tokens_per_second_mean",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": "chat",
                "draft_precision": "int4",
                "quant_method": "gptq_marlin",
                "gamma": "2",
                "acceptance_rate_mean": "0.7",
                "tokens_per_second_mean": "20",
            }
        )
        writer.writerow(
            {
                "task": "chat",
                "draft_precision": "int4",
                "quant_method": "gptq_marlin",
                "gamma": "4",
                "acceptance_rate_mean": "0.4",
                "tokens_per_second_mean": "30",
            }
        )

    recommend_gamma_policy(summary_path, output_path, acceptance_floor=0.5)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["policy"][0]["gamma"] == 2


def test_recommend_precision_policy_selects_fastest_precision_above_acceptance_floor(
    tmp_path: Path,
):
    summary_path = tmp_path / "precision.csv"
    output_path = tmp_path / "policy.json"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "draft_precision",
                "quant_method",
                "acceptance_rate_mean",
                "tokens_per_second_mean",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": "chat",
                "draft_precision": "bf16",
                "quant_method": "none",
                "acceptance_rate_mean": "0.6",
                "tokens_per_second_mean": "20",
            }
        )
        writer.writerow(
            {
                "task": "chat",
                "draft_precision": "int4",
                "quant_method": "gptq_marlin",
                "acceptance_rate_mean": "0.5",
                "tokens_per_second_mean": "30",
            }
        )

    recommend_precision_policy(summary_path, output_path, acceptance_floor=0.5)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["policy"][0]["draft_precision"] == "int4"
