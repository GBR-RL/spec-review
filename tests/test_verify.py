import json
from typing import Any

import httpx

from spec_review.llm.client import LLM
from spec_review.trace import verify


def test_rerank_moves_confirmed_links_up_and_keeps_the_tail() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    scores = {"a": -4.0, "b": 2.0, "c": 5.0}
    assert verify.rerank(ranked, scores, k=3) == ["c", "b", "a", "d", "e"]
    # Unjudged candidates (score 0) sit between confirmed and rejected ones, in retrieval order.
    assert verify.rerank(["x", "y", "z"], {"x": -1.0}, k=3) == ["y", "z", "x"]


def test_verdict_score_sign_follows_the_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = json.dumps({"link": "no", "confidence": 4})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}], "usage": {}},
        )

    llm = LLM("m", base_url="http://llm/v1", transport=httpx.MockTransport(handler))
    out: dict[str, Any]
    out, _ = verify.verify(llm, "Book a visit", "use case", "class Visit", "code")
    assert out["score"] == -4
