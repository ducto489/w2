from __future__ import annotations

import random
from statistics import mean


def bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        value = float(values[0])
        return value, value
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    rng = random.Random(seed)
    sample_size = len(values)
    bootstrap_means = [
        mean(values[rng.randrange(sample_size)] for _ in range(sample_size))
        for _ in range(n_resamples)
    ]
    bootstrap_means.sort()
    tail_probability = (1.0 - confidence) / 2.0
    low_index = int(tail_probability * n_resamples)
    high_index = int((1.0 - tail_probability) * n_resamples) - 1
    high_index = max(low_index, min(high_index, n_resamples - 1))
    return float(bootstrap_means[low_index]), float(bootstrap_means[high_index])
