import os

import pytest

from spec_review.trace import graph, recover

EDGES = [("UC1", "ID1"), ("ID1", "CC1"), ("UC1", "TC1"), ("TC1", "CC2"), ("CC2", "CC3")]


def test_impact_follows_edges_downstream_within_hops() -> None:
    g = graph.Graph(EDGES)
    assert g.impact("UC1", max_hops=1) == {"ID1", "TC1"}
    assert g.impact("UC1", max_hops=2) == {"ID1", "TC1", "CC1", "CC2"}
    assert g.impact("UC1", max_hops=3) == {"ID1", "TC1", "CC1", "CC2", "CC3"}
    assert g.impact("CC3") == set()


def test_impact_scores() -> None:
    gold = graph.Graph(EDGES)
    guess = graph.Graph([("UC1", "ID1"), ("ID1", "CC1"), ("UC1", "CC9")])
    s = graph.impact_scores(gold, guess, ["UC1", "CC3"], max_hops=2)
    assert s["starts"] == 1  # CC3 has no true impact
    assert s["recall"] == pytest.approx(2 / 4)
    assert s["precision"] == pytest.approx(2 / 3)


def test_average_precision() -> None:
    assert recover.average_precision(["a", "b", "c"], {"a", "c"}) == pytest.approx((1 + 2 / 3) / 2)
    assert recover.average_precision(["x", "y"], {"a"}) == 0.0


def test_scores_skip_sources_without_links() -> None:
    s = recover.score_rankings({"s1": ["t1", "t2"], "s2": ["t2"]}, {("s1", "t2")})
    assert s["sources_with_links"] == 1
    assert s["map"] == pytest.approx(0.5)
    assert s["recall@1"] == 0.0
    assert s["recall@5"] == 1.0


@pytest.mark.skipif("NEO4J_URI" not in os.environ, reason="needs a Neo4j server")
def test_neo4j_agrees_with_the_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    kinds = {"UC1": "use case", "ID1": "interaction diagram", "TC1": "test case",
             "CC1": "code", "CC2": "code", "CC3": "code"}  # fmt: skip
    arts = pd.DataFrame(
        {"dataset": "T", "id": list(kinds), "kind": list(kinds.values()), "text": ""}
    )
    monkeypatch.setattr(graph.coest, "load", lambda: (arts, pd.DataFrame()))
    store = graph.Neo4jGraph(
        os.environ["NEO4J_URI"], password=os.environ.get("NEO4J_PASSWORD", "neo4j")
    )
    try:
        store.load("T", {"gold": EDGES, "guess": [("UC1", "CC3")]})
        reference = graph.Graph(EDGES)
        for start in kinds:
            for hops in (1, 2, 3):
                assert store.impact("T", start, "gold", hops) == reference.impact(start, hops)
        assert store.impact("T", "UC1", "guess") == {"CC3"}
    finally:
        store.close()
