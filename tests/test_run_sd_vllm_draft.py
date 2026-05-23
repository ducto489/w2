from types import SimpleNamespace

from scripts.run_sd_vllm_draft import (
    DEFAULT_BF16_DRAFT_MODEL,
    DEFAULT_GPTQ_INT8_DRAFT_MODEL,
    DEFAULT_GPTQ_INT4_DRAFT_MODEL,
    VLLMDraftProposer,
    resolve_generation_settings,
    resolve_vllm_draft_settings,
)


class FakeLLM:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.calls = []

    def generate(self, *, prompt_token_ids, sampling_params, use_tqdm):
        self.calls.append(
            {
                "prompt_token_ids": prompt_token_ids,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
            }
        )
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=self.token_ids)])]


def test_vllm_draft_proposer_uses_prompt_token_ids_and_returns_generated_tokens():
    llm = FakeLLM([11, 12, 13, 14])
    sampling_params = object()
    proposer = VLLMDraftProposer(
        name="qwen-gptq",
        llm=llm,
        sampling_params=sampling_params,
        profile=False,
    )

    tokens = proposer.propose_tokens([101, 102], max_tokens=4)

    assert tokens == [11, 12, 13, 14]
    assert llm.calls == [
        {
            "prompt_token_ids": [[101, 102]],
            "sampling_params": sampling_params,
            "use_tqdm": False,
        }
    ]


def test_vllm_draft_proposer_profiles_calls_and_generated_tokens():
    proposer = VLLMDraftProposer(
        name="qwen-gptq",
        llm=FakeLLM([11, 12]),
        sampling_params=object(),
        profile=True,
    )

    before = proposer.profile_snapshot()
    proposer.propose_tokens([101], max_tokens=2)
    after = proposer.profile_snapshot()

    assert before == {
        "propose_calls": 0.0,
        "generated_tokens": 0.0,
        "model_generate_time_ms": 0.0,
    }
    assert after["propose_calls"] == 1.0
    assert after["generated_tokens"] == 2.0
    assert after["model_generate_time_ms"] >= 0.0


def test_resolve_vllm_draft_settings_defaults_bf16_to_unquantized_qwen():
    settings = resolve_vllm_draft_settings(
        draft_precision="bf16",
        draft_model=None,
        quantization=None,
        output_name=None,
    )

    assert settings.model_name == DEFAULT_BF16_DRAFT_MODEL
    assert settings.quantization is None
    assert settings.output_name == "vllm_bf16"
    assert settings.quant_method == "none"


def test_resolve_vllm_draft_settings_defaults_int4_to_gptq_marlin():
    settings = resolve_vllm_draft_settings(
        draft_precision="int4",
        draft_model=None,
        quantization=None,
        output_name=None,
    )

    assert settings.model_name == DEFAULT_GPTQ_INT4_DRAFT_MODEL
    assert settings.quantization == "gptq_marlin"
    assert settings.output_name == "vllm_gptq_marlin"
    assert settings.quant_method == "gptq_marlin"


def test_resolve_vllm_draft_settings_defaults_int8_to_gptq_auto_detection():
    settings = resolve_vllm_draft_settings(
        draft_precision="int8",
        draft_model=None,
        quantization=None,
        output_name=None,
    )

    assert settings.model_name == DEFAULT_GPTQ_INT8_DRAFT_MODEL
    assert settings.quantization is None
    assert settings.output_name == "vllm_gptq_int8"
    assert settings.quant_method == "gptq"


def test_resolve_vllm_draft_settings_allows_int8_quantization_override():
    settings = resolve_vllm_draft_settings(
        draft_precision="int8",
        draft_model=None,
        quantization="gptq",
        output_name="custom_int8",
    )

    assert settings.quantization == "gptq"
    assert settings.output_name == "custom_int8"
    assert settings.quant_method == "gptq"


def test_resolve_generation_settings_uses_config_defaults_without_overrides():
    config = SimpleNamespace(gamma=4, max_new_tokens=16)

    assert resolve_generation_settings(config, gamma=None, max_new_tokens=None) == (4, 16)


def test_resolve_generation_settings_applies_cli_overrides():
    config = SimpleNamespace(gamma=4, max_new_tokens=16)

    assert resolve_generation_settings(config, gamma=2, max_new_tokens=8) == (2, 8)


def test_resolve_generation_settings_rejects_non_positive_overrides():
    config = SimpleNamespace(gamma=4, max_new_tokens=16)

    try:
        resolve_generation_settings(config, gamma=0, max_new_tokens=None)
    except ValueError as exc:
        assert "gamma" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_vllm_draft_omits_quantization_when_none(monkeypatch):
    captured = {}

    class FakeLLMClass:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeSamplingParamsClass:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    import scripts.run_sd_vllm_draft as module

    monkeypatch.setattr(
        module,
        "_import_vllm",
        lambda: (FakeLLMClass, FakeSamplingParamsClass),
    )

    llm, sampling_params = module.load_vllm_draft(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        quantization=None,
        gamma=4,
        temperature=0.0,
        top_p=1.0,
        gpu_memory_utilization=0.50,
        dtype="bfloat16",
        ignore_eos=True,
    )

    assert isinstance(llm, FakeLLMClass)
    assert captured == {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "gpu_memory_utilization": 0.50,
        "dtype": "bfloat16",
    }
    assert sampling_params.kwargs == {
        "max_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
    }
