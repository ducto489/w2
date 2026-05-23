from __future__ import annotations

from dataclasses import dataclass

from dqf.quantization import QuantizationRequest, validate_quantization_request


@dataclass(frozen=True)
class HFModelHandle:
    name: str
    model: object
    tokenizer: object


def load_hf_pair(
    target_model_name: str,
    draft_model_name: str,
    draft_precision: str,
    quant_method: str,
    quantized_draft_torch_dtype: str | None = None,
) -> tuple[HFModelHandle, HFModelHandle]:
    validate_quantization_request(
        QuantizationRequest(draft_precision=draft_precision, quant_method=quant_method)
    )
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("transformers is required to load Hugging Face models") from exc

    if draft_precision in {"bf16", "fp16"} and not torch.cuda.is_available():
        raise RuntimeError("BF16/FP16 draft loading requires a CUDA-visible host")

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        device_map="cuda",
    )

    if draft_precision in {"int8", "int4"} and quant_method == "bnb":
        if not torch.cuda.is_available():
            raise RuntimeError("quantized draft loading requires a CUDA-visible host")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=draft_precision == "int8",
            load_in_4bit=draft_precision == "int4",
        )
        draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
        draft_kwargs = {
            "device_map": "auto",
            "quantization_config": quantization_config,
        }
        if quantized_draft_torch_dtype == "fp16":
            draft_kwargs["torch_dtype"] = torch.float16
        elif quantized_draft_torch_dtype == "bf16":
            draft_kwargs["torch_dtype"] = torch.bfloat16
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            **draft_kwargs,
        )
        return (
            HFModelHandle(name=target_model_name, model=target_model, tokenizer=tokenizer),
            HFModelHandle(name=draft_model_name, model=draft_model, tokenizer=draft_tokenizer),
        )

    if draft_precision in {"int8", "int4"} and quant_method in {"awq", "gptq"}:
        if not torch.cuda.is_available():
            raise RuntimeError("pre-quantized draft loading requires a CUDA-visible host")
        draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
        draft_kwargs = {"device_map": "auto"}
        if quantized_draft_torch_dtype == "fp16":
            draft_kwargs["torch_dtype"] = torch.float16
        elif quantized_draft_torch_dtype == "bf16":
            draft_kwargs["torch_dtype"] = torch.bfloat16
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            **draft_kwargs,
        )
        return (
            HFModelHandle(name=target_model_name, model=target_model, tokenizer=tokenizer),
            HFModelHandle(name=draft_model_name, model=draft_model, tokenizer=draft_tokenizer),
        )

    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
    torch_dtype = torch.bfloat16 if draft_precision == "bf16" else torch.float16
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch_dtype,
        device_map="cuda",
    )
    return (
        HFModelHandle(name=target_model_name, model=target_model, tokenizer=tokenizer),
        HFModelHandle(name=draft_model_name, model=draft_model, tokenizer=draft_tokenizer),
    )
