from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


_NUMBER_PATTERN = r"[-+]?\$?\d[\d,]*(?:\.\d+)?"


def normalize_numeric_answer(answer: str | int | float) -> str:
    text = str(answer).strip()
    text = text.replace("$", "").replace(",", "")
    text = text.rstrip(".")
    try:
        value = Decimal(text)
    except InvalidOperation:
        return text.lower()
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        return "0"
    return normalized


def extract_gsm8k_answer(text: str) -> str | None:
    marker_match = re.search(r"####\s*(%s)" % _NUMBER_PATTERN, text)
    if marker_match:
        return normalize_numeric_answer(marker_match.group(1))

    boxed_match = re.search(r"\\boxed\{\s*(%s)\s*\}" % _NUMBER_PATTERN, text)
    if boxed_match:
        return normalize_numeric_answer(boxed_match.group(1))

    answer_match = re.search(
        r"(?:the\s+)?answer\s+is\s*(%s)" % _NUMBER_PATTERN,
        text,
        flags=re.IGNORECASE,
    )
    if answer_match:
        return normalize_numeric_answer(answer_match.group(1))

    numbers = re.findall(_NUMBER_PATTERN, text)
    if not numbers:
        return None
    return normalize_numeric_answer(numbers[-1])
