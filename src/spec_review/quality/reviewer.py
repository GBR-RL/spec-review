"""LLM review of a requirement: what is wrong with it, and a rewrite that fixes it.

Two prompts. `pronoun` answers the narrow question ReqEval measures: does a marked pronoun have
one antecedent every reader would pick? `review` is the general review used by the linter: the
rule findings are passed in as hints, the model adds problems the rules cannot see (ambiguous
references, untestable intent, missing conditions) and proposes a rewrite. Rewrites are checked
again by the rules, so a rewrite that introduces new defects shows up.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from spec_review.llm.client import LLM, Completion
from spec_review.quality.rules import Finding

PRONOUN_SYSTEM = """You review software and system requirements for ambiguity.

A pronoun in the requirement is marked with <referential>...</referential>. Decide whether a
careful reader could reasonably link it to more than one noun phrase in the requirement.

- ambiguous: true if two or more noun phrases are plausible antecedents (for example both agree
  in number and could both make sense), false if one antecedent is clearly intended.
- antecedent: the noun phrase you think it refers to, copied from the requirement.
- reason: one short sentence."""

CANDIDATES_SYSTEM = """You review software and system requirements for ambiguity.

A pronoun in the requirement is marked with <referential>...</referential>. List every noun
phrase in the requirement that a careful reader could take it to refer to: it must agree with
the pronoun in number and make sense in the sentence. Include a candidate even if you think it
is less likely. Then say which one you would pick."""

REVIEW_SYSTEM = """You review engineering requirements against the INCOSE Guide to Writing
Requirements and ISO/IEC/IEEE 29148. A good requirement is one verifiable statement with an
explicit subject, "shall", and no vague, open-ended or undecided wording.

You get the requirement and the findings of a rule checker (which can be wrong). Report at most
five real problems, using exactly these types:
- vague: a word or phrase that cannot be measured ("fast", "user-friendly", "normal")
- ambiguous-reference: a pronoun or reference that could point to more than one thing
- untestable: no way to verify it passed, even with numbers added
- incomplete: a condition, actor or value is missing
- multiple: more than one requirement in one statement
- passive: an action without the actor that performs it
- weak-modal: "should", "will", "may" instead of "shall"
- open-ended: "etc.", "such as", lists that are not closed

Then rewrite it as one or more requirements, one statement each. Keep every number, unit,
condition and constraint of the original; change only what fixes a problem. Where a value is
needed but unknown, write [value]. If the requirement is fine, report no issues and return it
unchanged."""


class PronounVerdict(BaseModel):
    ambiguous: bool
    antecedent: str = Field(max_length=200)
    reason: str = Field(max_length=300)


class Candidates(BaseModel):
    candidates: list[str] = Field(max_length=6)
    chosen: str = Field(max_length=200)


class Issue(BaseModel):
    type: Literal[
        "vague", "ambiguous-reference", "untestable", "incomplete", "multiple", "passive",
        "weak-modal", "open-ended", "other",
    ]  # fmt: skip
    quote: str = Field(max_length=200)
    explanation: str = Field(max_length=300)


class Review(BaseModel):
    issues: list[Issue] = Field(max_length=5)
    rewrites: list[str] = Field(min_length=1, max_length=4)


def pronoun(llm: LLM, marked: str) -> tuple[dict[str, Any], Completion]:
    msgs = [
        {"role": "system", "content": PRONOUN_SYSTEM},
        {"role": "user", "content": f"Requirement:\n{marked}"},
    ]
    parsed, c = llm.parse(msgs, PronounVerdict, max_tokens=200)
    out: dict[str, Any] = json.loads(parsed.model_dump_json())
    return out, c


def pronoun_candidates(llm: LLM, marked: str) -> tuple[dict[str, Any], Completion]:
    """Ambiguous when the model lists two or more plausible antecedents."""
    msgs = [
        {"role": "system", "content": CANDIDATES_SYSTEM},
        {"role": "user", "content": f"Requirement:\n{marked}"},
    ]
    parsed, c = llm.parse(msgs, Candidates, max_tokens=250)
    out: dict[str, Any] = json.loads(parsed.model_dump_json())
    distinct = {" ".join(x.lower().split()) for x in out["candidates"] if x.strip()}
    out["ambiguous"] = len(distinct) >= 2
    out["antecedent"] = out["chosen"]
    return out, c


def review(
    llm: LLM, text: str, findings: list[Finding] | None = None
) -> tuple[dict[str, Any], Completion]:
    hints = "\n".join(f"- {f.rule}: '{f.text}' ({f.message})" for f in findings or [])
    user = f"Requirement:\n{text}\n\nRule checker findings:\n{hints or '- none'}"
    msgs = [{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": user}]
    parsed, c = llm.parse(msgs, Review, max_tokens=900)
    out: dict[str, Any] = json.loads(parsed.model_dump_json())
    return out, c
