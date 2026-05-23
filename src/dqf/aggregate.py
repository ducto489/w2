from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from dqf.logging_utils import load_jsonl
from dqf.metrics import aggregate_position_acceptance, compute_draft_cost_share
from dqf.stats import bootstrap_mean_ci


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _ci(values: list[float]) -> tuple[float, float]:
    return bootstrap_mean_ci(values)


def _position_values(records: list[dict], position: int) -> list[float]:
    values: list[float] = []
    for record in records:
        position_acceptance = record.get("position_acceptance", [])
        if len(position_acceptance) > position:
            values.append(float(position_acceptance[position]))
    return values


def _profile_value(record: dict, profile_name: str, key: str) -> float | None:
    profile = record.get(profile_name)
    if not isinstance(profile, dict):
        return None
    value = profile.get(key)
    if value is None:
        return None
    return float(value)


def _profile_values(records: list[dict], profile_name: str, key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _profile_value(record, profile_name, key)
        if value is not None:
            values.append(value)
    return values


def _profile_ratio_values(
    records: list[dict],
    profile_name: str,
    numerator_key: str,
    denominator_key: str,
) -> list[float]:
    values: list[float] = []
    for record in records:
        numerator = _profile_value(record, profile_name, numerator_key)
        denominator = _profile_value(record, profile_name, denominator_key)
        if numerator is None or denominator is None or denominator <= 0.0:
            continue
        values.append(numerator / denominator)
    return values


def aggregate_jsonl_to_csv(input_paths: list[Path], output_path: Path) -> Path:
    grouped: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for input_path in input_paths:
        for record in load_jsonl(input_path):
            key = (
                record["task"],
                record["draft_precision"],
                record["quant_method"],
                int(record["gamma"]),
            )
            grouped[key].append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
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
                "mean_accepted_length_ci_low",
                "mean_accepted_length_ci_high",
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
                "draft_time_ms_ci_low",
                "draft_time_ms_ci_high",
                "total_wall_time_ms_mean",
                "total_wall_time_ms_ci_low",
                "total_wall_time_ms_ci_high",
                "tokens_per_second_mean",
                "tokens_per_second_ci_low",
                "tokens_per_second_ci_high",
                "quality_mean",
                "quality_ci_low",
                "quality_ci_high",
                "semantic_exact_match_mean",
                "semantic_normalized_match_mean",
                "semantic_token_f1_mean",
                "semantic_length_ratio_mean",
                "task_exact_match_mean",
                "task_exact_match_ci_low",
                "task_exact_match_ci_high",
                "draft_cost_share_mean",
                "draft_cost_share_ci_low",
                "draft_cost_share_ci_high",
                "draft_forward_calls_mean",
                "draft_model_forward_time_ms_mean",
                "draft_model_forward_ms_per_call",
                "draft_input_tokens_per_forward",
                "draft_extend_cache_ratio",
                "draft_cache_reset_calls_mean",
                "draft_propose_calls_mean",
                "draft_model_generate_time_ms_mean",
                "draft_model_generate_ms_per_call",
                "draft_generated_tokens_per_call",
                "target_forward_calls_mean",
                "target_model_forward_time_ms_mean",
                "target_model_forward_ms_per_call",
            ],
        )
        writer.writeheader()

        for key, records in sorted(grouped.items()):
            task, draft_precision, quant_method, gamma = key
            position_values = aggregate_position_acceptance(
                [record["position_acceptance"] for record in records]
            )
            acceptance_rates = [float(record["acceptance_rate"]) for record in records]
            accepted_lengths = [float(record["mean_accepted_length"]) for record in records]
            draft_times = [float(record["draft_time_ms"]) for record in records]
            total_wall_times = [float(record["total_wall_time_ms"]) for record in records]
            tokens_per_second = [float(record["tokens_per_second"]) for record in records]
            quality_scores = [
                float(record["quality_score"])
                for record in records
                if record["quality_score"] is not None
            ]
            task_exact_matches = [
                float(record["task_exact_match"])
                for record in records
                if record.get("task_exact_match") is not None
            ]
            draft_cost_shares = [
                record.get(
                    "draft_cost_share",
                    compute_draft_cost_share(record["draft_time_ms"], record["total_wall_time_ms"]),
                )
                for record in records
            ]
            draft_forward_calls = _profile_values(records, "draft_profile", "forward_calls")
            draft_forward_times = _profile_values(
                records,
                "draft_profile",
                "model_forward_time_ms",
            )
            draft_cache_resets = _profile_values(records, "draft_profile", "cache_reset_calls")
            draft_propose_calls = _profile_values(records, "draft_profile", "propose_calls")
            draft_generate_times = _profile_values(
                records,
                "draft_profile",
                "model_generate_time_ms",
            )
            target_forward_calls = _profile_values(records, "target_profile", "forward_calls")
            target_forward_times = _profile_values(
                records,
                "target_profile",
                "model_forward_time_ms",
            )
            acceptance_rate_ci = _ci(acceptance_rates)
            accepted_length_ci = _ci(accepted_lengths)
            position_cis = [_ci(_position_values(records, index)) for index in range(4)]
            draft_time_ci = _ci(draft_times)
            total_wall_time_ci = _ci(total_wall_times)
            tokens_per_second_ci = _ci(tokens_per_second)
            quality_ci = _ci(quality_scores)
            task_exact_match_ci = _ci(task_exact_matches)
            draft_cost_share_ci = _ci([float(value) for value in draft_cost_shares])
            row = {
                "task": task,
                "draft_precision": draft_precision,
                "quant_method": quant_method,
                "gamma": gamma,
                "num_prompts": len(records),
                "acceptance_rate_mean": _mean(acceptance_rates),
                "acceptance_rate_std": _std(acceptance_rates),
                "acceptance_rate_ci_low": acceptance_rate_ci[0],
                "acceptance_rate_ci_high": acceptance_rate_ci[1],
                "mean_accepted_length_mean": _mean(accepted_lengths),
                "mean_accepted_length_ci_low": accepted_length_ci[0],
                "mean_accepted_length_ci_high": accepted_length_ci[1],
                "position_1_acceptance": position_values[0] if len(position_values) > 0 else 0.0,
                "position_1_acceptance_ci_low": position_cis[0][0],
                "position_1_acceptance_ci_high": position_cis[0][1],
                "position_2_acceptance": position_values[1] if len(position_values) > 1 else 0.0,
                "position_2_acceptance_ci_low": position_cis[1][0],
                "position_2_acceptance_ci_high": position_cis[1][1],
                "position_3_acceptance": position_values[2] if len(position_values) > 2 else 0.0,
                "position_3_acceptance_ci_low": position_cis[2][0],
                "position_3_acceptance_ci_high": position_cis[2][1],
                "position_4_acceptance": position_values[3] if len(position_values) > 3 else 0.0,
                "position_4_acceptance_ci_low": position_cis[3][0],
                "position_4_acceptance_ci_high": position_cis[3][1],
                "draft_time_ms_mean": _mean(draft_times),
                "draft_time_ms_ci_low": draft_time_ci[0],
                "draft_time_ms_ci_high": draft_time_ci[1],
                "total_wall_time_ms_mean": _mean(total_wall_times),
                "total_wall_time_ms_ci_low": total_wall_time_ci[0],
                "total_wall_time_ms_ci_high": total_wall_time_ci[1],
                "tokens_per_second_mean": _mean(tokens_per_second),
                "tokens_per_second_ci_low": tokens_per_second_ci[0],
                "tokens_per_second_ci_high": tokens_per_second_ci[1],
                "quality_mean": _mean(quality_scores),
                "quality_ci_low": quality_ci[0],
                "quality_ci_high": quality_ci[1],
                "semantic_exact_match_mean": _mean(
                    [
                        float(record["semantic_exact_match"])
                        for record in records
                        if record.get("semantic_exact_match") is not None
                    ]
                ),
                "semantic_normalized_match_mean": _mean(
                    [
                        float(record["semantic_normalized_match"])
                        for record in records
                        if record.get("semantic_normalized_match") is not None
                    ]
                ),
                "semantic_token_f1_mean": _mean(
                    [
                        float(record["semantic_token_f1"])
                        for record in records
                        if record.get("semantic_token_f1") is not None
                    ]
                ),
                "semantic_length_ratio_mean": _mean(
                    [
                        float(record["semantic_length_ratio"])
                        for record in records
                        if record.get("semantic_length_ratio") is not None
                    ]
                ),
                "task_exact_match_mean": _mean(task_exact_matches),
                "task_exact_match_ci_low": task_exact_match_ci[0],
                "task_exact_match_ci_high": task_exact_match_ci[1],
                "draft_cost_share_mean": _mean(draft_cost_shares),
                "draft_cost_share_ci_low": draft_cost_share_ci[0],
                "draft_cost_share_ci_high": draft_cost_share_ci[1],
                "draft_forward_calls_mean": _mean(draft_forward_calls),
                "draft_model_forward_time_ms_mean": _mean(draft_forward_times),
                "draft_model_forward_ms_per_call": _mean(
                    _profile_ratio_values(
                        records,
                        "draft_profile",
                        "model_forward_time_ms",
                        "forward_calls",
                    )
                ),
                "draft_input_tokens_per_forward": _mean(
                    _profile_ratio_values(
                        records,
                        "draft_profile",
                        "input_tokens_total",
                        "forward_calls",
                    )
                ),
                "draft_extend_cache_ratio": _mean(
                    _profile_ratio_values(
                        records,
                        "draft_profile",
                        "extend_cache_calls",
                        "forward_calls",
                    )
                ),
                "draft_cache_reset_calls_mean": _mean(draft_cache_resets),
                "draft_propose_calls_mean": _mean(draft_propose_calls),
                "draft_model_generate_time_ms_mean": _mean(draft_generate_times),
                "draft_model_generate_ms_per_call": _mean(
                    _profile_ratio_values(
                        records,
                        "draft_profile",
                        "model_generate_time_ms",
                        "propose_calls",
                    )
                ),
                "draft_generated_tokens_per_call": _mean(
                    _profile_ratio_values(
                        records,
                        "draft_profile",
                        "generated_tokens",
                        "propose_calls",
                    )
                ),
                "target_forward_calls_mean": _mean(target_forward_calls),
                "target_model_forward_time_ms_mean": _mean(target_forward_times),
                "target_model_forward_ms_per_call": _mean(
                    _profile_ratio_values(
                        records,
                        "target_profile",
                        "model_forward_time_ms",
                        "forward_calls",
                    )
                ),
            }
            writer.writerow(row)

    return output_path
