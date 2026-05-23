from __future__ import annotations

from statistics import mean


def compute_acceptance_rate(accepted_tokens: int, proposed_tokens: int) -> float:
    if proposed_tokens <= 0:
        return 0.0
    return accepted_tokens / proposed_tokens


def compute_mean_accepted_length(accepted_tokens: int, speculation_steps: int) -> float:
    if speculation_steps <= 0:
        return 0.0
    return accepted_tokens / speculation_steps


def aggregate_position_acceptance(position_series: list[list[float]]) -> list[float]:
    if not position_series:
        return []
    width = max(len(row) for row in position_series)
    columns: list[list[float]] = [[] for _ in range(width)]
    for row in position_series:
        for index in range(width):
            columns[index].append(row[index] if index < len(row) else 0.0)
    return [mean(column) for column in columns]


def compute_draft_cost_share(draft_time_ms: float, total_wall_time_ms: float) -> float:
    if total_wall_time_ms <= 0:
        return 0.0
    return draft_time_ms / total_wall_time_ms
