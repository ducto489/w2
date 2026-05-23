from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    task: str
    prompts_path: Path
    target_model: str
    draft_model: str
    gamma: int
    max_new_tokens: int
    batch_size: int
    quant_method: str
    results_dir: Path


def load_run_config(path: Path) -> RunConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    return RunConfig(
        task=payload["task"],
        prompts_path=Path(payload["prompts_path"]),
        target_model=payload["target_model"],
        draft_model=payload["draft_model"],
        gamma=int(payload["gamma"]),
        max_new_tokens=int(payload["max_new_tokens"]),
        batch_size=int(payload.get("batch_size", 1)),
        quant_method=str(payload.get("quant_method", "none")),
        results_dir=Path(payload["results_dir"]),
    )
