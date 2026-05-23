from dqf.task_quality import extract_gsm8k_answer, normalize_numeric_answer


def test_extract_gsm8k_answer_prefers_hash_answer_marker():
    text = "We compute several values: 10, 20. #### 42"

    assert extract_gsm8k_answer(text) == "42"


def test_extract_gsm8k_answer_handles_boxed_and_answer_is_patterns():
    assert extract_gsm8k_answer("Final result is \\boxed{1,234}.") == "1234"
    assert extract_gsm8k_answer("Therefore, the answer is -3.5.") == "-3.5"


def test_extract_gsm8k_answer_falls_back_to_last_number():
    assert extract_gsm8k_answer("First 12, then 18, so final 30.") == "30"


def test_normalize_numeric_answer_removes_commas_and_trailing_decimal_zeroes():
    assert normalize_numeric_answer("$1,200.00") == "1200"
    assert normalize_numeric_answer("03.50") == "3.5"
