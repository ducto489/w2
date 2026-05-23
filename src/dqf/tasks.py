from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptExample:
    prompt_id: str
    task: str
    prompt: str
    prompt_tokens: list[int] | None = None


def load_prompt_examples(path: Path, task: str) -> list[PromptExample]:
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            prompts.append(PromptExample(prompt_id=f"{task}-{index}", task=task, prompt=text))
    return prompts
