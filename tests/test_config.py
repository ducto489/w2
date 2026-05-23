from pathlib import Path

from dqf.config import load_run_config
from dqf.quantization import QuantizationRequest, validate_quantization_request


def test_load_run_config_reads_expected_fields(tmp_path: Path):
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        "\n".join(
            [
                "task: chat",
                "prompts_path: configs/prompts_chat.txt",
                "target_model: gpt2",
                "draft_model: gpt2",
                "gamma: 4",
                "max_new_tokens: 32",
                "batch_size: 1",
                "quant_method: none",
                "results_dir: results/raw",
            ]
        ),
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.task == "chat"
    assert config.gamma == 4
    assert config.prompts_path == Path("configs/prompts_chat.txt")


def test_benchmark_configs_use_valid_quantization_settings():
    bf16_config = load_run_config(Path("configs/run_bf16.yaml"))
    int8_config = load_run_config(Path("configs/run_int8.yaml"))
    int4_config = load_run_config(Path("configs/run_int4.yaml"))
    strong_smoke_bf16_config = load_run_config(Path("configs/run_bf16_strong_smoke_2026-05-15.yaml"))
    strong_smoke_int8_config = load_run_config(Path("configs/run_int8_strong_smoke_2026-05-15.yaml"))
    strong_smoke_int4_config = load_run_config(Path("configs/run_int4_strong_smoke_2026-05-15.yaml"))
    qwen_chat_bf16_config = load_run_config(Path("configs/run_bf16_qwen_chat_extended_2026-05-15.yaml"))
    qwen_chat_int8_config = load_run_config(Path("configs/run_int8_qwen_chat_extended_2026-05-15.yaml"))
    qwen_chat_int4_config = load_run_config(Path("configs/run_int4_qwen_chat_extended_2026-05-15.yaml"))
    qwen_reasoning_bf16_config = load_run_config(
        Path("configs/run_bf16_qwen_reasoning_extended_2026-05-15.yaml")
    )
    qwen_reasoning_int8_config = load_run_config(
        Path("configs/run_int8_qwen_reasoning_extended_2026-05-15.yaml")
    )
    qwen_reasoning_int4_config = load_run_config(
        Path("configs/run_int4_qwen_reasoning_extended_2026-05-15.yaml")
    )
    qwen_awq_profile_config = load_run_config(
        Path("configs/run_int4_qwen_chat_extended_awq_profile_2026-05-15.yaml")
    )
    qwen_gptq_profile_config = load_run_config(
        Path("configs/run_int4_qwen_chat_extended_gptq_profile_2026-05-15.yaml")
    )

    validate_quantization_request(
        QuantizationRequest(draft_precision="bf16", quant_method=bf16_config.quant_method)
    )
    validate_quantization_request(
        QuantizationRequest(draft_precision="int8", quant_method=int8_config.quant_method)
    )
    validate_quantization_request(
        QuantizationRequest(draft_precision="int4", quant_method=int4_config.quant_method)
    )
    validate_quantization_request(
        QuantizationRequest(
            draft_precision="bf16",
            quant_method=strong_smoke_bf16_config.quant_method,
        )
    )
    validate_quantization_request(
        QuantizationRequest(
            draft_precision="int8",
            quant_method=strong_smoke_int8_config.quant_method,
        )
    )
    validate_quantization_request(
        QuantizationRequest(
            draft_precision="int4",
            quant_method=strong_smoke_int4_config.quant_method,
        )
    )
    for precision, config in [
        ("bf16", qwen_chat_bf16_config),
        ("int8", qwen_chat_int8_config),
        ("int4", qwen_chat_int4_config),
        ("bf16", qwen_reasoning_bf16_config),
        ("int8", qwen_reasoning_int8_config),
        ("int4", qwen_reasoning_int4_config),
        ("int4", qwen_awq_profile_config),
        ("int4", qwen_gptq_profile_config),
    ]:
        validate_quantization_request(
            QuantizationRequest(draft_precision=precision, quant_method=config.quant_method)
        )

    assert bf16_config.quant_method == "none"
    assert int8_config.quant_method == "bnb"
    assert int4_config.quant_method == "bnb"
    assert strong_smoke_bf16_config.target_model == "Qwen/Qwen2.5-7B-Instruct"
    assert strong_smoke_bf16_config.draft_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert strong_smoke_bf16_config.prompts_path == Path("configs/prompts_chat.txt")
    assert strong_smoke_bf16_config.max_new_tokens == 8
    assert strong_smoke_bf16_config.results_dir == Path("results/raw_strong_smoke_2026-05-15")
    assert strong_smoke_bf16_config.quant_method == "none"
    assert strong_smoke_int8_config.quant_method == "bnb"
    assert strong_smoke_int4_config.quant_method == "bnb"
    assert qwen_chat_bf16_config.task == "chat"
    assert qwen_chat_bf16_config.prompts_path == Path("configs/prompts_chat_extended.txt")
    assert qwen_chat_bf16_config.max_new_tokens == 16
    assert qwen_chat_bf16_config.results_dir == Path("results/raw_qwen_chat_extended_2026-05-15")
    assert qwen_chat_bf16_config.quant_method == "none"
    assert qwen_chat_int8_config.quant_method == "bnb"
    assert qwen_chat_int4_config.quant_method == "bnb"
    assert qwen_reasoning_bf16_config.task == "reasoning"
    assert qwen_reasoning_bf16_config.prompts_path == Path("configs/prompts_reasoning_extended.txt")
    assert qwen_reasoning_bf16_config.max_new_tokens == 16
    assert qwen_reasoning_bf16_config.results_dir == Path(
        "results/raw_qwen_reasoning_extended_2026-05-15"
    )
    assert qwen_reasoning_bf16_config.quant_method == "none"
    assert qwen_reasoning_int8_config.quant_method == "bnb"
    assert qwen_reasoning_int4_config.quant_method == "bnb"
    assert qwen_awq_profile_config.draft_model == "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
    assert qwen_awq_profile_config.quant_method == "awq"
    assert qwen_gptq_profile_config.draft_model == "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
    assert qwen_gptq_profile_config.quant_method == "gptq"


def test_quantization_validation_accepts_alternate_backends():
    validate_quantization_request(
        QuantizationRequest(draft_precision="int4", quant_method="awq")
    )
    validate_quantization_request(
        QuantizationRequest(draft_precision="int4", quant_method="gptq")
    )
