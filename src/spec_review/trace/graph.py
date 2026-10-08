"""The trace graph and impact analysis: what else changes when a requirement changes?

Artifacts are nodes; trace links point downstream (use case to interaction diagram, test case
and code; diagram and test case to code). The impact of a change is everything reachable
downstream within a few hops. The graph is built either from the ground-truth links or from
the recovered ones (the top k candidates of every source), so the impact sets the recovered
graph would report can be compared with the true ones.

`Graph` is a plain-Python reference; `Neo4jGraph` stores the same graph in Neo4j and answers
the same question in Cypher. A test checks that the two agree.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from spec_review.config import RUNS
from spec_review.data import coest
from spec_review.trace.recover import PAIRS

DOWNSTREAM = {
    ("use case", "interaction diagram"),
    ("use case", "test case"),
    ("use case", "code"),
    ("interaction diagram", "code"),
    ("test case", "code"),
    ("high-level requirement", "low-level requirement"),
}
Edge = tuple[str, str]


class Graph:
    def __init__(self, edges: Iterable[Edge]) -> None:
        self.out: dict[str, set[str]] = defaultdict(set)
        for a, b in edges:
            self.out[a].add(b)

    def impact(self, start: str, max_hops: int = 3) -> set[str]:
        seen: set[str] = set()
        frontier = {start}
        for _ in range(max_hops):
            frontier = {n for f in frontier for n in self.out.get(f, ())} - seen - {start}
            if not frontier:
                break
            seen |= frontier
        return seen


def gold_edges(dataset: str) -> list[Edge]:
    arts, links = coest.load()
    kind = dict(zip(arts[arts.dataset == dataset].id, arts[arts.dataset == dataset].kind,
                    strict=True))  # fmt: skip
    edges = []
    for s, t in zip(links[links.dataset == dataset].source, links[links.dataset == dataset].target,
                    strict=True):  # fmt: skip
        if (kind.get(s), kind.get(t)) in DOWNSTREAM:
            edges.append((s, t))
        elif (kind.get(t), kind.get(s)) in DOWNSTREAM:
            edges.append((t, s))
    return sorted(set(edges))


def predicted_edges(method: str, dataset: str, k: int) -> list[Edge]:
    rankings = json.loads((RUNS / "trace" / f"{method}_rankings.json").read_text())
    edges: list[Edge] = []
    for pair in PAIRS:
        if pair.dataset != dataset or pair.name not in rankings:
            continue
        for source, ranked in rankings[pair.name].items():
            edges.extend((source, target) for target in ranked[:k])
    return sorted(set(edges))


def impact_scores(
    gold: Graph, predicted: Graph, starts: list[str], max_hops: int = 3
) -> dict[str, Any]:
    """Precision and recall of the predicted impact set against the true one, per start."""
    precision, recall, sizes = [], [], []
    for s in starts:
        truth, guess = gold.impact(s, max_hops), predicted.impact(s, max_hops)
        if not truth:
            continue
        hit = len(truth & guess)
        recall.append(hit / len(truth))
        precision.append(hit / len(guess) if guess else 0.0)
        sizes.append((len(truth), len(guess)))
    return {
        "starts": len(recall),
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "mean_true_impact": float(np.mean([t for t, _ in sizes])),
        "mean_predicted_impact": float(np.mean([g for _, g in sizes])),
    }


class Neo4jGraph:
    """The same graph in Neo4j: nodes :Artifact, relationships :TRACES with an `origin`."""

    def __init__(self, uri: str, user: str = "neo4j", password: str = "neo4j") -> None:
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def load(self, dataset: str, edges: dict[str, list[Edge]]) -> None:
        """Replace the dataset's subgraph. `edges` maps an origin ("gold", a method) to edges."""
        arts, _ = coest.load()
        a = arts[arts.dataset == dataset]
        nodes = [
            {"key": f"{dataset}/{i}", "id": i, "kind": k} for i, k in zip(a.id, a.kind, strict=True)
        ]
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT artifact_key IF NOT EXISTS "
                  "FOR (n:Artifact) REQUIRE n.key IS UNIQUE")  # fmt: skip
            s.run("MATCH (n:Artifact {dataset: $d}) DETACH DELETE n", d=dataset)
            s.run("UNWIND $nodes AS n CREATE (:Artifact {key: n.key, id: n.id, kind: n.kind, "
                  "dataset: $d})", nodes=nodes, d=dataset)  # fmt: skip
            for origin, es in edges.items():
                rows = [{"a": f"{dataset}/{x}", "b": f"{dataset}/{y}"} for x, y in es]
                s.run("UNWIND $rows AS r MATCH (a:Artifact {key: r.a}), (b:Artifact {key: r.b}) "
                      "CREATE (a)-[:TRACES {origin: $o}]->(b)", rows=rows, o=origin)  # fmt: skip

    def impact(self, dataset: str, start: str, origin: str, max_hops: int = 3) -> set[str]:
        query = (
            f"MATCH p = (a:Artifact {{key: $key}})-[:TRACES*1..{int(max_hops)}]->(x:Artifact) "
            "WHERE all(r IN relationships(p) WHERE r.origin = $origin) AND x.key <> $key "
            "RETURN DISTINCT x.id AS id"
        )
        with self.driver.session() as s:
            result = s.run(query, key=f"{dataset}/{start}", origin=origin)
            return {str(record["id"]) for record in result}
