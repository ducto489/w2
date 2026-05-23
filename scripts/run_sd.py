from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from dqf.config import load_run_config
from dqf.hf_runner import load_hf_pair
from dqf.sd_runner import DraftTargetPair, run_speculative_decode, run_target_only_decode
from dqf.tasks import PromptExample, load_prompt_examples
from scripts.evaluate_quality import evaluate_quality_file


class HFNextTokenAdapter:
    def __init__(self, name: str, model: object, tokenizer: object, profile: bool = False):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.profile = profile
        self._cached_context_tokens: list[int] = []
        self._past_key_values: object | None = None
        self._cached_next_token: int | None = None
        self._profile = {
            "next_token_calls": 0.0,
            "exact_cache_hits": 0.0,
            "forward_calls": 0.0,
            "extend_cache_calls": 0.0,
            "full_context_calls": 0.0,
            "cache_reset_calls": 0.0,
            "unsupported_cache_fallbacks": 0.0,
            "input_tokens_total": 0.0,
            "input_tokens_max": 0.0,
            "model_forward_time_ms": 0.0,
        }

    def next_token(self, context_tokens: list[int]) -> int:
        self._profile["next_token_calls"] += 1.0
        if context_tokens == self._cached_context_tokens and self._cached_next_token is not None:
            self._profile["exact_cache_hits"] += 1.0
            return self._cached_next_token

        input_tokens = self._input_tokens_for_context(context_tokens)
        self._profile["input_tokens_total"] += float(len(input_tokens))
        self._profile["input_tokens_max"] = max(
            self._profile["input_tokens_max"],
            float(len(input_tokens)),
        )
        input_ids = torch.tensor(
            [input_tokens],
            dtype=torch.long,
            device=self._model_device(),
        )
        with torch.no_grad():
            try:
                outputs = self._timed_forward(
                    input_ids=input_ids,
                    past_key_values=self._past_key_values,
                    use_cache=True,
                )
            except TypeError as exc:
                if not self._is_unsupported_cache_argument_error(exc):
                    raise
                self._profile["unsupported_cache_fallbacks"] += 1.0
                self._cached_context_tokens = []
                self._past_key_values = None
                self._cached_next_token = None
                input_ids = torch.tensor(
                    [context_tokens],
                    dtype=torch.long,
                    device=self._model_device(),
                )
                self._profile["input_tokens_total"] += float(len(context_tokens))
                self._profile["input_tokens_max"] = max(
                    self._profile["input_tokens_max"],
                    float(len(context_tokens)),
                )
                outputs = self._timed_forward(input_ids=input_ids)
        logits = outputs.logits[0, -1]
        next_token = int(torch.argmax(logits).item())
        self._cached_context_tokens = list(context_tokens)
        self._past_key_values = getattr(outputs, "past_key_values", None)
        self._cached_next_token = next_token
        return next_token

    def _input_tokens_for_context(self, context_tokens: list[int]) -> list[int]:
        if self._can_extend_cache(context_tokens):
            self._profile["extend_cache_calls"] += 1.0
            return context_tokens[len(self._cached_context_tokens):]

        if self._past_key_values is not None:
            self._profile["cache_reset_calls"] += 1.0
        self._profile["full_context_calls"] += 1.0
        self._cached_context_tokens = []
        self._past_key_values = None
        self._cached_next_token = None
        return list(context_tokens)

    def _timed_forward(self, **kwargs: object) -> object:
        self._profile["forward_calls"] += 1.0
        if not self.profile:
            return self.model(**kwargs)

        self._synchronize_if_cuda()
        start_time = perf_counter()
        outputs = self.model(**kwargs)
        self._synchronize_if_cuda()
        self._profile["model_forward_time_ms"] += (perf_counter() - start_time) * 1000.0
        return outputs

    def _synchronize_if_cuda(self) -> None:
        device = self._model_device()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def profile_snapshot(self) -> dict[str, float] | None:
        if not self.profile:
            return None
        return dict(self._profile)

    def _can_extend_cache(self, context_tokens: list[int]) -> bool:
        if self._past_key_values is None:
            return False
        if len(context_tokens) <= len(self._cached_context_tokens):
            return False
        return context_tokens[: len(self._cached_context_tokens)] == self._cached_context_tokens

    def _is_unsupported_cache_argument_error(self, exc: TypeError) -> bool:
        message = str(exc)
        return "unexpected keyword argument" in message and (
            "past_key_values" in message or "use_cache" in message
        )

    def _model_device(self) -> torch.device:
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            return torch.device(model_device)

        parameters = getattr(self.model, "parameters", None)
        if parameters is None:
            return torch.device("cpu")

        first_parameter = next(parameters(), None)
        if first_parameter is None:
            return torch.device("cpu")
        return first_parameter.device


def write_target_only_baseline(
    target_model: object,
    prompts: list[PromptExample],
    max_new_tokens: int,
    output_path: Path,
) -> Path:
    outputs = run_target_only_decode(
        target_model=target_model,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
    )
    payload = {
        str(output["prompt_id"]): [int(token) for token in output["output_tokens"]]
        for output in outputs
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)
    return output_path


def run_benchmark_with_quality(
    pair: DraftTargetPair,
    prompts: list[PromptExample],
    gamma: int,
    max_new_tokens: int,
    output_path: Path,
    draft_precision: str,
    quant_method: str,
) -> Path:
    run_speculative_decode(
        pair=pair,
        prompts=prompts,
        gamma=gamma,
        max_new_tokens=max_new_tokens,
        output_path=output_path,
        draft_precision=draft_precision,
        quant_method=quant_method,
    )
    baseline_path = output_path.with_name(f"{output_path.stem}_target_only.json")
    write_target_only_baseline(
        target_model=pair.target_model,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        output_path=baseline_path,
    )
    evaluate_quality_file(output_path, baseline_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--precision", required=True, choices=["bf16", "fp16", "int8", "int4"])
    parser.add_argument(
        "--profile",
        action="store_true",
        help="write per-prompt HF adapter profiling counters into the raw JSONL records",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="override the config results_dir for clean reruns or diagnostics",
    )
    parser.add_argument(
        "--quantized-draft-torch-dtype",
        choices=["auto", "fp16", "bf16"],
        default="auto",
        help="optional torch_dtype override for int8/int4 draft model loading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(Path(args.config))
    target_handle, draft_handle = load_hf_pair(
        target_model_name=config.target_model,
        draft_model_name=config.draft_model,
        draft_precision=args.precision,
        quant_method=config.quant_method,
        quantized_draft_torch_dtype=(
            None if args.quantized_draft_torch_dtype == "auto" else args.quantized_draft_torch_dtype
        ),
    )
    raw_prompts = load_prompt_examples(config.prompts_path, config.task)
    prompts = [
        PromptExample(
            prompt_id=prompt.prompt_id,
            task=prompt.task,
            prompt=prompt.prompt,
            prompt_tokens=target_handle.tokenizer.encode(prompt.prompt, add_special_tokens=False),
        )
        for prompt in raw_prompts
    ]
    pair = DraftTargetPair(
        target_model=HFNextTokenAdapter(
            target_handle.name,
            target_handle.model,
            target_handle.tokenizer,
            profile=args.profile,
        ),
        draft_model=HFNextTokenAdapter(
            draft_handle.name,
            draft_handle.model,
            draft_handle.tokenizer,
            profile=args.profile,
        ),
    )
    results_dir = args.results_dir if args.results_dir is not None else config.results_dir
    output_path = results_dir / f"{args.precision}.jsonl"
    run_benchmark_with_quality(
        pair=pair,
        prompts=prompts,
        gamma=config.gamma,
        max_new_tokens=config.max_new_tokens,
        output_path=output_path,
        draft_precision=args.precision,
        quant_method=config.quant_method,
    )


if __name__ == "__main__":
    main()
