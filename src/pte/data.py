from __future__ import annotations

import json
from pathlib import Path


def load_prompt_answer_jsonl(path: str | Path) -> tuple[list[str], list[str]]:
    path = Path(path)
    prompts: list[str] = []
    answers: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            answer = row.get("answer")
            if not isinstance(prompt, str) or not isinstance(answer, str):
                raise ValueError(f"line {line_no}: expected string fields 'prompt' and 'answer'")
            prompts.append(prompt)
            answers.append(answer)
    if not prompts:
        raise ValueError(f"no valid rows found in {path}")
    return prompts, answers
