import json
from typing import Any

import httpx
import numpy as np

from spec_review.eval import quality
from spec_review.llm.client import LLM
from spec_review.quality import reviewer, rules


def _llm(content: dict[str, Any], seen: list[dict[str, Any]]) -> LLM:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {"prompt_tokens": 90, "completion_tokens": 20},
            },
        )

    return LLM("m", base_url="http://llm/v1", transport=httpx.MockTransport(handler))


def test_pronoun_verdict_is_parsed() -> None:
    seen: list[dict[str, Any]] = []
    reply = {"ambiguous": True, "antecedent": "the pump", "reason": "Two devices fit."}
    out, c = reviewer.pronoun(
        _llm(reply, seen), "The pump feeds the valve until <referential>it</referential> closes."
    )
    assert out == reply
    assert c.prompt_tokens == 90
    assert "<referential>it</referential>" in seen[0]["messages"][1]["content"]


def test_review_passes_rule_findings_as_hints() -> None:
    seen: list[dict[str, Any]] = []
    reply = {
        "issues": [{"type": "vague", "quote": "quickly", "explanation": "Not measurable."}],
        "rewrite": "The pump shall reach nominal speed within [value] s of power-up.",
    }
    text = "The pump should start quickly."
    out, _ = reviewer.review(_llm(reply, seen), text, rules.check(text))
    assert out["issues"][0]["type"] == "vague"
    hints = seen[0]["messages"][1]["content"]
    assert "vague-term: 'quickly'" in hints
    assert "weak-modal" in hints


def test_candidate_heuristic_counts_noun_phrases() -> None:
    assert quality.candidate_heuristic(
        "The operator sends the report to the manager when <referential>he</referential> is ready."
    )
    assert not quality.candidate_heuristic(
        "The pump stops when <referential>it</referential> overheats."
    )


def test_scores_for_a_constant_prediction() -> None:
    y = np.array([True, True, False, False])
    s = quality._scores(y, np.ones_like(y))
    assert s["accuracy"] == 0.5
    assert s["cohen_kappa"] == 0.0


def test_candidates_make_a_pronoun_ambiguous() -> None:
    reply = {"candidates": ["the pump", "The Pump ", "the valve"], "chosen": "the pump"}
    out, _ = reviewer.pronoun_candidates(_llm(reply, []), "x <referential>it</referential>")
    assert out["ambiguous"] is True
    one = {"candidates": ["the pump", "the  pump"], "chosen": "the pump"}
    out, _ = reviewer.pronoun_candidates(_llm(one, []), "x <referential>it</referential>")
    assert out["ambiguous"] is False
    assert out["antecedent"] == "the pump"
