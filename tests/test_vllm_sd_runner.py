from pathlib import Path

from dqf.logging_utils import load_jsonl
from dqf.tasks import PromptExample
from dqf.vllm_sd_runner import DraftTargetProposerPair, run_speculative_decode_with_draft_proposer


class FakeTarget:
    def __init__(self, planned_tokens: list[int], profile: bool = False):
        self.name = "fake-target"
        self.planned_tokens = planned_tokens
        self.calls: list[list[int]] = []
        self.profile = profile

    def next_token(self, context_tokens: list[int]) -> int:
        self.calls.append(list(context_tokens))
        step = len(context_tokens) - 1
        if step >= len(self.planned_tokens):
            return self.planned_tokens[-1]
        return self.planned_tokens[step]

    def profile_snapshot(self):
        if not self.profile:
            return None
        return {"next_token_calls": float(len(self.calls))}


class FakeDraftProposer:
    def __init__(self, cycles: list[list[int]], profile: bool = False):
        self.name = "fake-vllm-draft"
        self.cycles = cycles
        self.calls: list[tuple[list[int], int]] = []
        self.profile = profile

    def propose_tokens(self, context_tokens: list[int], max_tokens: int) -> list[int]:
        self.calls.append((list(context_tokens), max_tokens))
        if len(self.calls) > len(self.cycles):
            return self.cycles[-1][:max_tokens]
        return self.cycles[len(self.calls) - 1][:max_tokens]

    def profile_snapshot(self):
        if not self.profile:
            return None
        return {"propose_calls": float(len(self.calls))}


def test_run_speculative_decode_with_draft_proposer_logs_acceptance_and_continues(
    tmp_path: Path,
):
    prompt = PromptExample(prompt_id="chat-0", task="chat", prompt="hi", prompt_tokens=[101])
    output_path = tmp_path / "vllm.jsonl"

    result = run_speculative_decode_with_draft_proposer(
        pair=DraftTargetProposerPair(
            target_model=FakeTarget([11, 12, 99, 100]),
            draft_proposer=FakeDraftProposer([[11, 12, 13, 14], [100, 101, 102, 103]]),
        ),
        prompts=[prompt],
        gamma=4,
        max_new_tokens=4,
        output_path=output_path,
        draft_precision="int4",
        quant_method="gptq_marlin",
    )

    records = load_jsonl(output_path)

    assert result.summary["num_records"] == 1
    assert len(records) == 1
    record = records[0]
    assert record["prompt_id"] == "chat-0"
    assert record["target_model"] == "fake-target"
    assert record["draft_model"] == "fake-vllm-draft"
    assert record["draft_precision"] == "int4"
    assert record["quant_method"] == "gptq_marlin"
    assert record["num_draft_tokens_proposed"] == 8
    assert record["num_tokens_accepted"] == 3
    assert record["num_tokens_rejected"] == 2
    assert record["position_acceptance"] == [1.0, 0.5, 0.0, 0.0]
    assert record["acceptance_rate"] == 0.375
    assert record["mean_accepted_length"] == 1.5
    assert record["generated_tokens"] == [11, 12, 99, 100]
    assert record["quality_score"] is None
    assert record["notes"] == "vLLM draft proposer with HF target verification"


def test_run_speculative_decode_with_draft_proposer_writes_profile_deltas(tmp_path: Path):
    prompt = PromptExample(prompt_id="chat-0", task="chat", prompt="hi", prompt_tokens=[101])
    output_path = tmp_path / "vllm.jsonl"

    run_speculative_decode_with_draft_proposer(
        pair=DraftTargetProposerPair(
            target_model=FakeTarget([11, 12], profile=True),
            draft_proposer=FakeDraftProposer([[11, 12]], profile=True),
        ),
        prompts=[prompt],
        gamma=2,
        max_new_tokens=2,
        output_path=output_path,
        draft_precision="int4",
        quant_method="gptq_marlin",
    )

    record = load_jsonl(output_path)[0]

    assert record["draft_profile"] == {"propose_calls": 1.0}
    assert record["target_profile"] == {"next_token_calls": 2.0}
