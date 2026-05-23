from dqf.metrics import (
    aggregate_position_acceptance,
    compute_acceptance_rate,
    compute_draft_cost_share,
    compute_mean_accepted_length,
)


def test_compute_acceptance_rate_uses_proposed_tokens():
    assert compute_acceptance_rate(accepted_tokens=6, proposed_tokens=10) == 0.6


def test_compute_acceptance_rate_handles_zero_proposals():
    assert compute_acceptance_rate(accepted_tokens=0, proposed_tokens=0) == 0.0


def test_compute_mean_accepted_length_uses_steps():
    assert compute_mean_accepted_length(accepted_tokens=8, speculation_steps=4) == 2.0


def test_compute_mean_accepted_length_handles_zero_steps():
    assert compute_mean_accepted_length(accepted_tokens=0, speculation_steps=0) == 0.0


def test_aggregate_position_acceptance_computes_column_means():
    series = [[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]]
    assert aggregate_position_acceptance(series) == [0.5, 0.75, 0.25]


def test_compute_draft_cost_share_uses_total_wall_time():
    assert compute_draft_cost_share(draft_time_ms=40.0, total_wall_time_ms=100.0) == 0.4


def test_compute_draft_cost_share_handles_zero_total_time():
    assert compute_draft_cost_share(draft_time_ms=40.0, total_wall_time_ms=0.0) == 0.0
