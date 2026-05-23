from __future__ import annotations

import re
from collections import Counter


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall(text.lower()))


def compute_semantic_quality(generated_text: str, baseline_text: str) -> dict[str, float]:
    generated_normalized = normalize_text(generated_text)
    baseline_normalized = normalize_text(baseline_text)
    return {
        "semantic_exact_match": 1.0 if generated_text == baseline_text else 0.0,
        "semantic_normalized_match": 1.0
        if generated_normalized == baseline_normalized and baseline_normalized != ""
        else 0.0,
        "semantic_token_f1": _token_f1(generated_normalized, baseline_normalized),
        "semantic_length_ratio": _length_ratio(generated_normalized, baseline_normalized),
    }


def _token_f1(generated_normalized: str, baseline_normalized: str) -> float:
    generated_tokens = generated_normalized.split()
    baseline_tokens = baseline_normalized.split()
    if not generated_tokens and not baseline_tokens:
        return 1.0
    if not generated_tokens or not baseline_tokens:
        return 0.0
    generated_counts = Counter(generated_tokens)
    baseline_counts = Counter(baseline_tokens)
    overlap = sum((generated_counts & baseline_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(generated_tokens)
    recall = overlap / len(baseline_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _length_ratio(generated_normalized: str, baseline_normalized: str) -> float:
    generated_length = len(generated_normalized.split())
    baseline_length = len(baseline_normalized.split())
    if generated_length == 0 and baseline_length == 0:
        return 1.0
    if baseline_length == 0:
        return 0.0
    return generated_length / baseline_length
