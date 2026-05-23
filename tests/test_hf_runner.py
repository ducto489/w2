import sys
from types import SimpleNamespace

import pytest

from dqf.hf_runner import load_hf_pair


class FakeAutoTokenizer:
    calls: list[str] = []

    @classmethod
    def from_pretrained(cls, name: str):
        cls.calls.append(name)
        return f"tokenizer:{name}"


class FakeAutoModelForCausalLM:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, name: str, **kwargs):
        cls.calls.append((name, kwargs))
        return SimpleNamespace(name=name, kwargs=kwargs)


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_load_hf_pair_uses_cuda_for_bf16(monkeypatch: pytest.MonkeyPatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        bfloat16="bf16",
        float16="fp16",
    )
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        AutoTokenizer=FakeAutoTokenizer,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )
    FakeAutoModelForCausalLM.calls = []
    FakeAutoTokenizer.calls = []

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    target_handle, draft_handle = load_hf_pair(
        target_model_name="target-model",
        draft_model_name="draft-model",
        draft_precision="bf16",
        quant_method="none",
    )

    assert target_handle.name == "target-model"
    assert draft_handle.name == "draft-model"
    assert FakeAutoTokenizer.calls == ["target-model", "draft-model"]
    assert FakeAutoModelForCausalLM.calls == [
        ("target-model", {"device_map": "cuda"}),
        ("draft-model", {"torch_dtype": "bf16", "device_map": "cuda"}),
    ]


def test_load_hf_pair_rejects_bf16_without_cuda(monkeypatch: pytest.MonkeyPatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        bfloat16="bf16",
        float16="fp16",
    )
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        AutoTokenizer=FakeAutoTokenizer,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="BF16/FP16 draft loading requires a CUDA-visible host"):
        load_hf_pair(
            target_model_name="target-model",
            draft_model_name="draft-model",
            draft_precision="bf16",
            quant_method="none",
        )


def test_load_hf_pair_can_force_fp16_for_quantized_draft(monkeypatch: pytest.MonkeyPatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        bfloat16="bf16",
        float16="fp16",
    )
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        AutoTokenizer=FakeAutoTokenizer,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )
    FakeAutoModelForCausalLM.calls = []
    FakeAutoTokenizer.calls = []

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    load_hf_pair(
        target_model_name="target-model",
        draft_model_name="draft-model",
        draft_precision="int8",
        quant_method="bnb",
        quantized_draft_torch_dtype="fp16",
    )

    assert FakeAutoModelForCausalLM.calls == [
        ("target-model", {"device_map": "cuda"}),
        (
            "draft-model",
            {
                "device_map": "auto",
                "quantization_config": FakeAutoModelForCausalLM.calls[1][1]["quantization_config"],
                "torch_dtype": "fp16",
            },
        ),
    ]


def test_load_hf_pair_uses_prequantized_model_for_awq(monkeypatch: pytest.MonkeyPatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        bfloat16="bf16",
        float16="fp16",
    )
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        AutoTokenizer=FakeAutoTokenizer,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )
    FakeAutoModelForCausalLM.calls = []
    FakeAutoTokenizer.calls = []

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    load_hf_pair(
        target_model_name="target-model",
        draft_model_name="draft-model-awq",
        draft_precision="int4",
        quant_method="awq",
        quantized_draft_torch_dtype="fp16",
    )

    assert FakeAutoModelForCausalLM.calls == [
        ("target-model", {"device_map": "cuda"}),
        ("draft-model-awq", {"device_map": "auto", "torch_dtype": "fp16"}),
    ]
