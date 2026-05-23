from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from dqf.logging_utils import append_jsonl
from dqf.metrics import (
    compute_acceptance_rate,
    compute_draft_cost_share,
    compute_mean_accepted_length,
)
from dqf.tasks import PromptExample


class NextTokenModel(Protocol):
    name: str

    def next_token(self, context_tokens: list[int]) -> int:
        ...


@dataclass(frozen=True)
class DraftTargetPair:
    target_model: NextTokenModel
    draft_model: NextTokenModel


@dataclass(frozen=True)
class RunResult:
    summary: dict[str, Any]


def _profile_snapshot(model: NextTokenModel) -> dict[str, float] | None:
    snapshot = getattr(model, "profile_snapshot", None)
    if not callable(snapshot):
        return None
    payload = snapshot()
    if payload is None:
        return None
    return dict(payload)


def _profile_delta(
    start: dict[str, float] | None,
    end: dict[str, float] | None,
) -> dict[str, float] | None:
    if start is None or end is None:
        return None
    delta: dict[str, float] = {}
    for key, end_value in end.items():
        start_value = start.get(key, 0.0)
        delta[key] = end_value - start_value
    return delta


def run_target_only_decode(
    target_model: NextTokenModel,
    prompts: list[PromptExample],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    outputs = []
    for prompt in prompts:
        context = list(prompt.prompt_tokens or [])
        generated: list[int] = []
        start_time = perf_counter()
        for _ in range(max_new_tokens):
            token = target_model.next_token(context)
            context.append(token)
            generated.append(token)
        elapsed_ms = (perf_counter() - start_time) * 1000.0
        outputs.append(
            {
                "prompt_id": prompt.prompt_id,
                "task": prompt.task,
                "output_tokens": generated,
                "total_wall_time_ms": elapsed_ms,
            }
        )
    return outputs


def run_speculative_decode(
    pair: DraftTargetPair,
    prompts: list[PromptExample],
    gamma: int,
    max_new_tokens: int,
    output_path: Path,
    draft_precision: str,
    quant_method: str,
) -> RunResult:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    run_id = str(uuid4())
    records_written = 0
    for prompt in prompts:
        draft_profile_start = _profile_snapshot(pair.draft_model)
        target_profile_start = _profile_snapshot(pair.target_model)
        base_context = list(prompt.prompt_tokens or [])
        working_context = list(base_context)
        generated: list[int] = []
        proposed_tokens: list[int] = []
        accepted_tokens = 0
        rejected_tokens = 0
        position_acceptance = [0.0 for _ in range(gamma)]
        speculation_steps = 0
        draft_time_ms = 0.0
        verify_time_ms = 0.0
        wall_start = perf_counter()

        while len(generated) < max_new_tokens:
            speculation_steps += 1
            cycle_context = list(working_context)
            cycle_proposals: list[int] = []

            draft_start = perf_counter()
            for _ in range(gamma):
                draft_token = pair.draft_model.next_token(cycle_context)
                cycle_context.append(draft_token)
                cycle_proposals.append(draft_token)
                proposed_tokens.append(draft_token)
            draft_time_ms += (perf_counter() - draft_start) * 1000.0

            verify_start = perf_counter()
            accepted_in_cycle = 0
            for position, draft_token in enumerate(cycle_proposals):
                verifier_token = pair.target_model.next_token(working_context)
                if verifier_token == draft_token:
                    working_context.append(draft_token)
                    generated.append(draft_token)
                    accepted_tokens += 1
                    accepted_in_cycle += 1
                    position_acceptance[position] += 1.0
                    if len(generated) >= max_new_tokens:
                        break
                else:
                    working_context.append(verifier_token)
                    generated.append(verifier_token)
                    rejected_tokens += len(cycle_proposals) - position
                    break
            else:
                verifier_token = pair.target_model.next_token(working_context)
                working_context.append(verifier_token)
                generated.append(verifier_token)

            verify_time_ms += (perf_counter() - verify_start) * 1000.0

        total_wall_time_ms = (perf_counter() - wall_start) * 1000.0
        output_count = len(generated)
        draft_profile = _profile_delta(
            draft_profile_start,
            _profile_snapshot(pair.draft_model),
        )
        target_profile = _profile_delta(
            target_profile_start,
            _profile_snapshot(pair.target_model),
        )
        record = {
            "run_id": run_id,
            "task": prompt.task,
            "prompt_id": prompt.prompt_id,
            "target_model": pair.target_model.name,
            "draft_model": pair.draft_model.name,
            "draft_precision": draft_precision,
            "quant_method": quant_method,
            "gamma": gamma,
            "batch_size": 1,
            "input_tokens": len(base_context),
            "output_tokens": output_count,
            "generated_tokens": generated,
            "num_draft_tokens_proposed": len(proposed_tokens),
            "num_tokens_accepted": accepted_tokens,
            "num_tokens_rejected": rejected_tokens,
            "mean_accepted_length": compute_mean_accepted_length(accepted_tokens, speculation_steps),
            "acceptance_rate": compute_acceptance_rate(accepted_tokens, len(proposed_tokens)),
            "position_acceptance": [
                value / speculation_steps for value in position_acceptance
            ],
            "draft_time_ms": draft_time_ms,
            "verify_time_ms": verify_time_ms,
            "total_wall_time_ms": total_wall_time_ms,
            "tokens_per_second": (output_count / total_wall_time_ms * 1000.0) if total_wall_time_ms > 0 else 0.0,
            "quality_score": None,
            "notes": "",
            "draft_cost_share": compute_draft_cost_share(draft_time_ms, total_wall_time_ms),
        }
        if draft_profile is not None:
            record["draft_profile"] = draft_profile
        if target_profile is not None:
            record["target_profile"] = target_profile
        append_jsonl(output_path, record)
        records_written += 1

    return RunResult(summary={"run_id": run_id, "num_records": records_written})
