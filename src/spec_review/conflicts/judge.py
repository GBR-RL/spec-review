"""LLM judgement of candidate conflicts: can both requirements be met at the same time?"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from spec_review.llm.client import LLM, Completion

TOP_K = 50

SYSTEM = """You check pairs of requirements from the same specification for conflicts.

Two requirements conflict when a system cannot satisfy both at once: they demand different
values, behaviours or limits for the same thing, or one forbids what the other requires.
Requirements that are about different things, or that only repeat each other, do not conflict.

Answer "yes" or "no", give your confidence from 1 (guess) to 5 (certain) and name the point of
conflict in a few words (empty if none)."""


class Verdict(BaseModel):
    conflict: Literal["yes", "no"]
    confidence: int = Field(ge=1, le=5)
    point: str = Field(max_length=160)


def judge(llm: LLM, a: str, b: str) -> tuple[dict[str, Any], Completion]:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Requirement 1: {a}\nRequirement 2: {b}"},
    ]
    parsed, c = llm.parse(msgs, Verdict, max_tokens=80)
    out: dict[str, Any] = json.loads(parsed.model_dump_json())
    out["score"] = out["confidence"] if out["conflict"] == "yes" else -out["confidence"]
    return out, c
