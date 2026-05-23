from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_DRAFT_PRECISIONS = {"bf16", "fp16", "int8", "int4"}
SUPPORTED_QUANT_METHODS = {"none", "bnb", "awq", "gptq"}


@dataclass(frozen=True)
class QuantizationRequest:
    draft_precision: str
    quant_method: str


def validate_quantization_request(request: QuantizationRequest) -> None:
    if request.draft_precision not in SUPPORTED_DRAFT_PRECISIONS:
        raise ValueError(f"unsupported draft precision: {request.draft_precision}")
    if request.quant_method not in SUPPORTED_QUANT_METHODS:
        raise ValueError(f"unsupported quant_method: {request.quant_method}")
    if request.draft_precision in {"int8", "int4"} and request.quant_method == "none":
        raise ValueError("quantized draft precision requires a quant_method")
