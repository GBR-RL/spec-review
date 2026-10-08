"""LLM verification of candidate trace links.

For every source, the model reads the source and each of its top-k recovered candidates and
says whether a trace link exists (the target realises, tests or details the source) with a
confidence from 1 to 5. Candidates are re-ranked by that verdict; the retrieval order breaks
ties and everything below the top k keeps its place.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from spec_review.llm.client import LLM, Completion

TOP_K = 10
MAX_WORDS = 350

SYSTEM = """You check trace links in a software project. A trace link exists when the second
artifact implements, tests, refines or otherwise directly realises what the first one asks for.
Sharing a few words is not enough; it must be about the same functionality.

Answer "yes" or "no" and give your confidence from 1 (guess) to 5 (certain)."""


class Verdict(BaseModel):
    link: Literal["yes", "no"]
    confidence: int = Field(ge=1, le=5)


def _clip(text: str) -> str:
    words = text.split()
    return " ".join(words[:MAX_WORDS]) + (" ..." if len(words) > MAX_WORDS else "")


def verify(
    llm: LLM, source: str, source_kind: str, target: str, target_kind: str
) -> tuple[dict[str, Any], Completion]:
    user = (
        f"First artifact ({source_kind}):\n{_clip(source)}\n\n"
        f"Second artifact ({target_kind}):\n{_clip(target)}"
    )
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    parsed, c = llm.parse(msgs, Verdict, max_tokens=30)
    out: dict[str, Any] = json.loads(parsed.model_dump_json())
    out["score"] = out["confidence"] if out["link"] == "yes" else -out["confidence"]
    return out, c


def rerank(ranked: list[str], scores: dict[str, float], k: int = TOP_K) -> list[str]:
    """Sort the top k by verdict score, keeping retrieval order for ties and the tail as is."""
    head = ranked[:k]
    order = sorted(range(len(head)), key=lambda i: (-scores.get(head[i], 0.0), i))
    return [head[i] for i in order] + ranked[k:]
