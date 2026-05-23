import json
from pathlib import Path

import torch

from dqf.logging_utils import load_jsonl
from dqf.sd_runner import DraftTargetPair, PromptExample, run_speculative_decode
from scripts.run_sd import HFNextTokenAdapter


class FakeModel:
    def __init__(self, name: str, planned_tokens: list[int]):
        self.name = name
        self.planned_tokens = planned_tokens
        self.calls: list[list[int]] = []

    def next_token(self, context_tokens: list[int]) -> int:
        step = len(context_tokens) - 1
        self.calls.append(list(context_tokens))
        if step >= len(self.planned_tokens):
            return self.planned_tokens[-1]
        return self.planned_tokens[step]


class TinyTorchModel:
    def __call__(self, input_ids):
        batch, seq = input_ids.shape
        logits = torch.zeros((batch, seq, 8), dtype=torch.float32)
        logits[:, -1, 5] = 3.0
        return type("Output", (), {"logits": logits})()


class DeviceAwareTinyTorchModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.last_input_device = None

    def __call__(self, input_ids):
        self.last_input_device = input_ids.device
        batch, seq = input_ids.shape
        logits = torch.zeros((batch, seq, 8), dtype=torch.float32, device=input_ids.device)
        logits[:, -1, 6] = 3.0
        return type("Output", (), {"logits": logits})()


def test_run_speculative_decode_continues_after_rejection_until_max_new_tokens(tmp_path: Path):
    prompt = PromptExample(prompt_id="p1", task="chat", prompt="hi", prompt_tokens=[101])
    draft_model = FakeModel(name="draft", planned_tokens=[11, 12, 13, 14])
    target_model = FakeModel(name="target", planned_tokens=[11, 12, 99, 100])
    output_path = tmp_path / "run.jsonl"

    result = run_speculative_decode(
        pair=DraftTargetPair(target_model=target_model, draft_model=draft_model),
        prompts=[prompt],
        gamma=4,
        max_new_tokens=4,
        output_path=output_path,
        draft_precision="bf16",
        quant_method="none",
    )

    records = load_jsonl(output_path)

    assert result.summary["num_records"] == 1
    assert len(records) == 1
    record = records[0]
    assert record["prompt_id"] == "p1"
    assert record["num_draft_tokens_proposed"] == 8
    assert record["num_tokens_accepted"] == 2
    assert record["num_tokens_rejected"] == 6
    assert record["position_acceptance"] == [0.5, 0.5, 0.0, 0.0]
    assert record["acceptance_rate"] == 0.25
    assert record["mean_accepted_length"] == 1.0
    assert record["output_tokens"] == 4
    assert record["generated_tokens"] == [11, 12, 99, 100]
    assert record["quality_score"] is None

    output_path.read_text(encoding="utf-8")


def test_run_speculative_decode_rejects_non_positive_gamma(tmp_path: Path):
    prompt = PromptExample(prompt_id="p1", task="chat", prompt="hi", prompt_tokens=[101])
    draft_model = FakeModel(name="draft", planned_tokens=[11])
    target_model = FakeModel(name="target", planned_tokens=[11])

    try:
        run_speculative_decode(
            pair=DraftTargetPair(target_model=target_model, draft_model=draft_model),
            prompts=[prompt],
            gamma=0,
            max_new_tokens=1,
            output_path=tmp_path / "run.jsonl",
            draft_precision="bf16",
            quant_method="none",
        )
    except ValueError as exc:
        assert "gamma" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-positive gamma")


def test_hf_next_token_adapter_returns_token_id():
    adapter = HFNextTokenAdapter(name="tiny", model=TinyTorchModel(), tokenizer=None)

    token = adapter.next_token([1, 2, 3])

    assert token == 5


def test_hf_next_token_adapter_places_inputs_on_model_device():
    model = DeviceAwareTinyTorchModel()
    adapter = HFNextTokenAdapter(name="tiny", model=model, tokenizer=None)
    original_tensor = torch.tensor
    tensor_device = {"value": None}

    def tracking_tensor(*args, **kwargs):
        tensor_device["value"] = kwargs.get("device")
        return original_tensor(*args, **kwargs)

    torch.tensor = tracking_tensor

    try:
        token = adapter.next_token([1, 2, 3])
    finally:
        torch.tensor = original_tensor

    assert token == 6
    assert tensor_device["value"] == model.device
    assert model.last_input_device == model.device
