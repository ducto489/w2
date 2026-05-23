import json
from pathlib import Path
from types import SimpleNamespace

import torch

from dqf.logging_utils import load_jsonl
from dqf.sd_runner import DraftTargetPair
from dqf.tasks import PromptExample
from scripts.run_sd import HFNextTokenAdapter, run_benchmark_with_quality, write_target_only_baseline


class FakeModel:
    def __init__(self, name: str, planned_tokens: list[int]):
        self.name = name
        self.planned_tokens = planned_tokens

    def next_token(self, context_tokens: list[int]) -> int:
        step = len(context_tokens) - 1
        if step >= len(self.planned_tokens):
            return self.planned_tokens[-1]
        return self.planned_tokens[step]


class FakeCausalLM:
    def __init__(self):
        self.name = "fake-causal-lm"
        self.device = torch.device("cpu")
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        call_index = len(self.calls)
        self.calls.append(
            {
                "input_ids": input_ids.detach().cpu().tolist()[0],
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )
        logits = torch.zeros((1, input_ids.shape[1], 8), dtype=torch.float32)
        logits[0, -1, (call_index + 1) % 8] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=f"past-{call_index}")


def test_write_target_only_baseline_serializes_prompt_outputs(tmp_path: Path):
    prompts = [
        PromptExample(prompt_id="chat-0", task="chat", prompt="hi", prompt_tokens=[101]),
        PromptExample(prompt_id="chat-1", task="chat", prompt="yo", prompt_tokens=[202]),
    ]
    baseline_path = tmp_path / "baseline.json"

    write_target_only_baseline(
        target_model=FakeModel(name="target", planned_tokens=[11, 12, 13]),
        prompts=prompts,
        max_new_tokens=3,
        output_path=baseline_path,
    )

    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert payload == {
        "chat-0": [11, 12, 13],
        "chat-1": [11, 12, 13],
    }


def test_run_benchmark_with_quality_populates_quality_scores(tmp_path: Path):
    prompt = PromptExample(prompt_id="chat-0", task="chat", prompt="hi", prompt_tokens=[101])
    output_path = tmp_path / "bf16.jsonl"

    run_benchmark_with_quality(
        pair=DraftTargetPair(
            target_model=FakeModel(name="target", planned_tokens=[11, 12, 13]),
            draft_model=FakeModel(name="draft", planned_tokens=[11, 12, 13]),
        ),
        prompts=[prompt],
        gamma=2,
        max_new_tokens=3,
        output_path=output_path,
        draft_precision="bf16",
        quant_method="none",
    )

    records = load_jsonl(output_path)
    assert records[0]["generated_tokens"] == [11, 12, 13]
    assert records[0]["quality_score"] == 1.0

    baseline_path = tmp_path / "bf16_target_only.json"
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert payload["chat-0"] == [11, 12, 13]


def test_run_benchmark_omits_profiles_when_adapter_profile_disabled(tmp_path: Path):
    prompt = PromptExample(prompt_id="chat-0", task="chat", prompt="hi", prompt_tokens=[101])
    output_path = tmp_path / "bf16.jsonl"

    run_benchmark_with_quality(
        pair=DraftTargetPair(
            target_model=HFNextTokenAdapter(
                name="target",
                model=FakeCausalLM(),
                tokenizer=object(),
            ),
            draft_model=HFNextTokenAdapter(
                name="draft",
                model=FakeCausalLM(),
                tokenizer=object(),
            ),
        ),
        prompts=[prompt],
        gamma=2,
        max_new_tokens=2,
        output_path=output_path,
        draft_precision="bf16",
        quant_method="none",
    )

    record = load_jsonl(output_path)[0]

    assert "draft_profile" not in record
    assert "target_profile" not in record


def test_run_benchmark_writes_profiles_when_adapter_profile_enabled(tmp_path: Path):
    prompt = PromptExample(prompt_id="chat-0", task="chat", prompt="hi", prompt_tokens=[101])
    output_path = tmp_path / "bf16.jsonl"

    run_benchmark_with_quality(
        pair=DraftTargetPair(
            target_model=HFNextTokenAdapter(
                name="target",
                model=FakeCausalLM(),
                tokenizer=object(),
                profile=True,
            ),
            draft_model=HFNextTokenAdapter(
                name="draft",
                model=FakeCausalLM(),
                tokenizer=object(),
                profile=True,
            ),
        ),
        prompts=[prompt],
        gamma=2,
        max_new_tokens=2,
        output_path=output_path,
        draft_precision="bf16",
        quant_method="none",
    )

    record = load_jsonl(output_path)[0]

    assert record["draft_profile"]["next_token_calls"] > 0.0
    assert record["draft_profile"]["forward_calls"] > 0.0
    assert record["draft_profile"]["input_tokens_total"] > 0.0
    assert record["target_profile"]["next_token_calls"] > 0.0


def test_hf_next_token_adapter_reuses_cache_for_extended_contexts():
    model = FakeCausalLM()
    adapter = HFNextTokenAdapter(name="fake", model=model, tokenizer=object())

    first_token = adapter.next_token([1, 2, 3])
    second_token = adapter.next_token([1, 2, 3, 4])

    assert first_token == 1
    assert second_token == 2
    assert model.calls == [
        {"input_ids": [1, 2, 3], "past_key_values": None, "use_cache": True},
        {"input_ids": [4], "past_key_values": "past-0", "use_cache": True},
    ]


def test_hf_next_token_adapter_resets_cache_for_unrelated_contexts():
    model = FakeCausalLM()
    adapter = HFNextTokenAdapter(name="fake", model=model, tokenizer=object())

    adapter.next_token([1, 2, 3])
    adapter.next_token([9, 10])

    assert model.calls == [
        {"input_ids": [1, 2, 3], "past_key_values": None, "use_cache": True},
        {"input_ids": [9, 10], "past_key_values": None, "use_cache": True},
    ]
