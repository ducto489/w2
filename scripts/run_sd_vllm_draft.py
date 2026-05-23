from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from dqf.config import load_run_config
from dqf.hf_runner import HFModelHandle
from dqf.vllm_sd_runner import DraftTargetProposerPair, run_speculative_decode_with_draft_proposer
from dqf.tasks import PromptExample, load_prompt_examples
from scripts.evaluate_quality import evaluate_quality_file
from scripts.run_sd import HFNextTokenAdapter, write_target_only_baseline


DEFAULT_BF16_DRAFT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_GPTQ_INT8_DRAFT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8"
DEFAULT_GPTQ_INT4_DRAFT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
DEFAULT_GPTQ_INT4_QUANTIZATION = "gptq_marlin"


@dataclass(frozen=True)
class VLLMDraftSettings:
    model_name: str
    quantization: str | None
    output_name: str
    quant_method: str


def resolve_vllm_draft_settings(
    *,
    draft_precision: str,
    draft_model: str | None,
    quantization: str | None,
    output_name: str | None,
) -> VLLMDraftSettings:
    if draft_precision == "bf16":
        return VLLMDraftSettings(
            model_name=draft_model or DEFAULT_BF16_DRAFT_MODEL,
            quantization=quantization,
            output_name=output_name or "vllm_bf16",
            quant_method=quantization or "none",
        )
    if draft_precision == "int8":
        return VLLMDraftSettings(
            model_name=draft_model or DEFAULT_GPTQ_INT8_DRAFT_MODEL,
            quantization=quantization,
            output_name=output_name or "vllm_gptq_int8",
            quant_method=quantization or "gptq",
        )
    if draft_precision == "int4":
        resolved_quantization = quantization or DEFAULT_GPTQ_INT4_QUANTIZATION
        return VLLMDraftSettings(
            model_name=draft_model or DEFAULT_GPTQ_INT4_DRAFT_MODEL,
            quantization=resolved_quantization,
            output_name=output_name or "vllm_gptq_marlin",
            quant_method=resolved_quantization,
        )
    raise ValueError(f"unsupported vLLM draft precision: {draft_precision}")


def resolve_generation_settings(
    config: object,
    *,
    gamma: int | None,
    max_new_tokens: int | None,
) -> tuple[int, int]:
    resolved_gamma = int(gamma if gamma is not None else getattr(config, "gamma"))
    resolved_max_new_tokens = int(
        max_new_tokens if max_new_tokens is not None else getattr(config, "max_new_tokens")
    )
    if resolved_gamma <= 0:
        raise ValueError("gamma must be positive")
    if resolved_max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    return resolved_gamma, resolved_max_new_tokens


class VLLMDraftProposer:
    def __init__(
        self,
        name: str,
        llm: object,
        sampling_params: object,
        profile: bool = False,
    ):
        self.name = name
        self.llm = llm
        self.sampling_params = sampling_params
        self.profile = profile
        self._profile = {
            "propose_calls": 0.0,
            "generated_tokens": 0.0,
            "model_generate_time_ms": 0.0,
        }

    def propose_tokens(self, context_tokens: list[int], max_tokens: int) -> list[int]:
        self._profile["propose_calls"] += 1.0
        start_time = perf_counter()
        outputs = self.llm.generate(
            prompt_token_ids=[list(context_tokens)],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        elapsed_ms = (perf_counter() - start_time) * 1000.0
        token_ids = self._output_token_ids(outputs[0])
        self._profile["generated_tokens"] += float(len(token_ids))
        self._profile["model_generate_time_ms"] += elapsed_ms
        if len(token_ids) != max_tokens:
            raise RuntimeError(
                f"vLLM draft returned {len(token_ids)} tokens for requested gamma={max_tokens}"
            )
        return token_ids

    def profile_snapshot(self) -> dict[str, float] | None:
        if not self.profile:
            return None
        return dict(self._profile)

    def _output_token_ids(self, request_output: object) -> list[int]:
        outputs = getattr(request_output, "outputs", None)
        if not outputs:
            return []
        token_ids = getattr(outputs[0], "token_ids", None)
        if token_ids is None:
            return []
        return [int(token_id) for token_id in token_ids]


def load_hf_target(model_name: str, torch_dtype_name: str) -> HFModelHandle:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers and torch are required to load the target model") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("vLLM-draft speculative decoding requires a CUDA-visible host")

    if torch_dtype_name == "bf16":
        torch_dtype = torch.bfloat16
    elif torch_dtype_name == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"unsupported target torch dtype: {torch_dtype_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = model.to("cuda")
    model.eval()
    return HFModelHandle(name=model_name, model=model, tokenizer=tokenizer)


def load_vllm_draft(
    *,
    model_name: str,
    quantization: str | None,
    gamma: int,
    temperature: float,
    top_p: float,
    gpu_memory_utilization: float,
    dtype: str,
    ignore_eos: bool = False,
) -> tuple[object, object]:
    LLM, SamplingParams = _import_vllm()

    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
    }
    if quantization is not None:
        llm_kwargs["quantization"] = quantization
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        max_tokens=gamma,
        temperature=temperature,
        top_p=top_p,
        ignore_eos=ignore_eos,
    )
    return llm, sampling_params


def _import_vllm() -> tuple[type, type]:
    from vllm import LLM, SamplingParams

    return LLM, SamplingParams


def verify_tokenizer_compatibility(
    *,
    target_tokenizer: object,
    vllm_llm: object,
    prompts: list[PromptExample],
) -> None:
    get_tokenizer = getattr(vllm_llm, "get_tokenizer", None)
    if not callable(get_tokenizer):
        return
    vllm_tokenizer = get_tokenizer()
    for prompt in prompts[: min(len(prompts), 3)]:
        target_ids = target_tokenizer.encode(prompt.prompt, add_special_tokens=False)
        vllm_ids = vllm_tokenizer.encode(prompt.prompt, add_special_tokens=False)
        if list(target_ids) != list(vllm_ids):
            raise RuntimeError(
                "target HF tokenizer and vLLM draft tokenizer disagree on prompt token IDs"
            )


def prepare_prompts(
    *,
    prompts_path: Path,
    task: str,
    tokenizer: object,
    limit: int | None,
) -> list[PromptExample]:
    raw_prompts = load_prompt_examples(prompts_path, task)
    if limit is not None:
        raw_prompts = raw_prompts[:limit]
    return [
        PromptExample(
            prompt_id=prompt.prompt_id,
            task=prompt.task,
            prompt=prompt.prompt,
            prompt_tokens=tokenizer.encode(prompt.prompt, add_special_tokens=False),
        )
        for prompt in raw_prompts
    ]


def run_benchmark_with_quality(
    *,
    pair: DraftTargetProposerPair,
    prompts: list[PromptExample],
    gamma: int,
    max_new_tokens: int,
    output_path: Path,
    draft_precision: str,
    quant_method: str,
) -> Path:
    run_speculative_decode_with_draft_proposer(
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
    parser = argparse.ArgumentParser(
        description="Run speculative decoding with vLLM draft and HF target verification."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--draft-precision", choices=["bf16", "int8", "int4"], default="int4")
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gamma", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--target-torch-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="write vLLM draft and HF target profile counters into raw JSONL records",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(Path(args.config))
    draft_settings = resolve_vllm_draft_settings(
        draft_precision=args.draft_precision,
        draft_model=args.draft_model,
        quantization=args.quantization,
        output_name=args.output_name,
    )
    gamma, max_new_tokens = resolve_generation_settings(
        config,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
    )
    target_handle = load_hf_target(config.target_model, args.target_torch_dtype)
    prompts = prepare_prompts(
        prompts_path=config.prompts_path,
        task=config.task,
        tokenizer=target_handle.tokenizer,
        limit=args.limit,
    )
    vllm_llm, sampling_params = load_vllm_draft(
        model_name=draft_settings.model_name,
        quantization=draft_settings.quantization,
        gamma=gamma,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        ignore_eos=args.ignore_eos,
    )
    verify_tokenizer_compatibility(
        target_tokenizer=target_handle.tokenizer,
        vllm_llm=vllm_llm,
        prompts=prompts,
    )
    pair = DraftTargetProposerPair(
        target_model=HFNextTokenAdapter(
            target_handle.name,
            target_handle.model,
            target_handle.tokenizer,
            profile=args.profile,
        ),
        draft_proposer=VLLMDraftProposer(
            name=draft_settings.model_name,
            llm=vllm_llm,
            sampling_params=sampling_params,
            profile=args.profile,
        ),
    )
    results_dir = args.results_dir if args.results_dir is not None else config.results_dir
    output_path = results_dir / f"{draft_settings.output_name}.jsonl"
    run_benchmark_with_quality(
        pair=pair,
        prompts=prompts,
        gamma=gamma,
        max_new_tokens=max_new_tokens,
        output_path=output_path,
        draft_precision=args.draft_precision,
        quant_method=draft_settings.quant_method,
    )


if __name__ == "__main__":
    main()
