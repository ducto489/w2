from dqf.stats import bootstrap_mean_ci


def test_bootstrap_mean_ci_returns_zero_interval_for_empty_values():
    assert bootstrap_mean_ci([]) == (0.0, 0.0)


def test_bootstrap_mean_ci_returns_point_interval_for_single_value():
    assert bootstrap_mean_ci([0.75]) == (0.75, 0.75)


def test_bootstrap_mean_ci_is_deterministic_and_bounds_mean():
    low, high = bootstrap_mean_ci([0.0, 0.5, 1.0], n_resamples=200, seed=123)

    assert low <= 0.5 <= high
    assert (low, high) == bootstrap_mean_ci([0.0, 0.5, 1.0], n_resamples=200, seed=123)
