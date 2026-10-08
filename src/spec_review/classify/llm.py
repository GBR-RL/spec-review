"""Few-shot LLM classification: class definitions plus similar labelled requirements.

The examples for a requirement come only from the training projects of its fold, so the LLM is
held to the same rule as the trained classifiers: it never sees the requirement's project.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

from spec_review.llm.client import LLM, Completion

DEFINITIONS = {
    "F": "functional: what the system does, a behaviour or function",
    "A": "availability: when and how reliably the system is available (uptime, hours)",
    "FT": "fault tolerance: behaviour under failures, recovery, backups",
    "L": "legal: laws, regulations, standards, licences, compliance",
    "LF": "look and feel: appearance, style, colours, layout",
    "MN": "maintainability: ease of changing, updating, supporting the system",
    "O": "operability: the operating environment, interfaces to other systems, conditions of use",
    "PE": "performance: speed, response time, throughput, capacity, resource use",
    "PO": "portability: running on or moving to other platforms and environments",
    "SC": "scalability: growth in users, data or load",
    "SE": "security: access control, authentication, privacy, protection of data",
    "US": "usability: ease of use and learning, user efficiency and satisfaction",
}
K_EXAMPLES = 8

SYSTEM = """Classify a software requirement into exactly one class.

Classes:
{definitions}

Answer with the class code only, in the JSON field "label"."""


class Label(BaseModel):
    label: Literal["F", "A", "FT", "L", "LF", "MN", "O", "PE", "PO", "SC", "SE", "US"]


def examples_for(data: pd.DataFrame, fold_of: np.ndarray, k: int = K_EXAMPLES) -> list[list[int]]:
    """Row indices of the k most similar requirements from the same fold's training projects."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    x = vec.fit_transform(data.text)
    sims = (x @ x.T).toarray()
    out = []
    for i in range(len(data)):
        allowed = fold_of != fold_of[i]
        candidates = np.where(allowed)[0]
        best = candidates[np.argsort(-sims[i, candidates])[:k]]
        out.append(best.tolist())
    return out


def messages(text: str, shots: list[tuple[str, str]]) -> list[dict[str, str]]:
    definitions = "\n".join(f"- {code}: {d}" for code, d in DEFINITIONS.items())
    msgs = [{"role": "system", "content": SYSTEM.format(definitions=definitions)}]
    for example, label in shots:
        msgs.append({"role": "user", "content": example})
        msgs.append({"role": "assistant", "content": json.dumps({"label": label})})
    msgs.append({"role": "user", "content": text})
    return msgs


def classify(llm: LLM, text: str, shots: list[tuple[str, str]]) -> tuple[str, Completion]:
    parsed, c = llm.parse(messages(text, shots), Label, max_tokens=20)
    out: dict[str, Any] = json.loads(parsed.model_dump_json())
    return str(out["label"]), c
