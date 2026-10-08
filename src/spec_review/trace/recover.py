"""Trace link recovery: rank every candidate target for every source artifact.

A "pair" is one direction of tracing inside one dataset, for example EasyClinic use cases to
test cases. Every source gets a ranked list of all targets of that kind; the ground-truth links
score the ranking. Sources without any true link are left out of the averages (there is nothing
to find), as is usual in traceability research.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

from spec_review.config import ROOT, RUNS
from spec_review.data import coest

RESULTS = ROOT / "docs" / "results" / "traceability"
KS = (1, 5, 10)


@dataclass(frozen=True)
class Pair:
    dataset: str
    source_kind: str
    target_kind: str

    @property
    def name(self) -> str:
        short = {"high-level requirement": "HLR", "low-level requirement": "LLR",
                 "use case": "UC", "code": "CODE", "test case": "TC",
                 "interaction diagram": "ID"}  # fmt: skip
        return f"{self.dataset}:{short[self.source_kind]}-{short[self.target_kind]}"


PAIRS = (
    Pair("CM1", "high-level requirement", "low-level requirement"),
    Pair("eTOUR", "use case", "code"),
    Pair("iTrust", "use case", "code"),
    Pair("EasyClinic", "use case", "interaction diagram"),
    Pair("EasyClinic", "use case", "test case"),
    Pair("EasyClinic", "use case", "code"),
    # Second hop of EasyClinic's chain, for multi-hop impact analysis.
    Pair("EasyClinic", "interaction diagram", "code"),
    Pair("EasyClinic", "test case", "code"),
)

# similarity(source_texts, target_texts) -> (n_sources, n_targets) scores
Similarity = Callable[[list[str], list[str]], NDArray[np.float64]]


def tfidf(sources: list[str], targets: list[str]) -> NDArray[np.float64]:
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", token_pattern=r"[A-Za-z]{2,}")
    vec.fit(sources + targets)
    s, t = vec.transform(sources), vec.transform(targets)
    return np.asarray((s @ t.T).toarray(), dtype=np.float64)


def pair_data(pair: Pair) -> tuple[pd.DataFrame, pd.DataFrame, set[tuple[str, str]]]:
    arts, links = coest.load()
    a = arts[arts.dataset == pair.dataset]
    src = a[a.kind == pair.source_kind].reset_index(drop=True)
    tgt = a[a.kind == pair.target_kind].reset_index(drop=True)
    ids_s, ids_t = set(src.id), set(tgt.id)
    gold = {
        (s, t)
        for s, t in zip(links.source, links.target, strict=True)
        if (s in ids_s and t in ids_t)
    } | {
        (t, s)
        for s, t in zip(links.source, links.target, strict=True)
        if (t in ids_s and s in ids_t)
    }
    gold = {g for g in gold if g[0] in ids_s and g[1] in ids_t}
    return src, tgt, gold


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    hits, total = 0, 0.0
    for i, t in enumerate(ranked, 1):
        if t in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def score_rankings(rankings: dict[str, list[str]], gold: set[tuple[str, str]]) -> dict[str, Any]:
    by_source: dict[str, set[str]] = {}
    for s, t in gold:
        by_source.setdefault(s, set()).add(t)
    aps: list[float] = []
    recalls: dict[int, list[float]] = {k: [] for k in KS}
    for s, relevant in by_source.items():
        ranked = rankings.get(s, [])
        aps.append(average_precision(ranked, relevant))
        for k in KS:
            recalls[k].append(len(relevant & set(ranked[:k])) / len(relevant))
    rng = np.random.default_rng(0)
    ap = np.array(aps)
    boot = [ap[rng.integers(0, len(ap), len(ap))].mean() for _ in range(1000)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "sources_with_links": len(aps),
        "links": len(gold),
        "map": float(ap.mean()),
        "map_ci": (float(lo), float(hi)),
        **{f"recall@{k}": float(np.mean(v)) for k, v in recalls.items()},
    }


def rank(pair: Pair, similarity: Similarity) -> dict[str, list[str]]:
    src, tgt, _ = pair_data(pair)
    scores = similarity(src.text.tolist(), tgt.text.tolist())
    order = np.argsort(-scores, axis=1)
    targets = tgt.id.to_numpy()
    return {sid: targets[order[i]].tolist() for i, sid in enumerate(src.id)}


def evaluate(method: str, similarity: Similarity) -> dict[str, Any]:
    out: dict[str, Any] = {"method": method, "pairs": {}}
    rankings_all = {}
    for pair in PAIRS:
        _, _, gold = pair_data(pair)
        rankings = rank(pair, similarity)
        rankings_all[pair.name] = rankings
        out["pairs"][pair.name] = score_rankings(rankings, gold)
    out["mean_map"] = float(np.mean([p["map"] for p in out["pairs"].values()]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{method}.json").write_text(json.dumps(_round(out), indent=2) + "\n")
    runs = RUNS / "trace"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{method}_rankings.json").write_text(json.dumps(rankings_all))
    return out


def _round(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_round(x) for x in v]
    return v


DENSE_MODELS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "bge-m3": ("BAAI/bge-m3", "", ""),
}


def dense(model_key: str) -> Similarity:
    """Cosine similarity of sentence embeddings (multilingual: eTOUR's code is in Italian)."""
    hf_name, q_prefix, p_prefix = DENSE_MODELS[model_key]

    def similarity(sources: list[str], targets: list[str]) -> NDArray[np.float64]:
        model = _cached_model(hf_name)
        s = model.encode([q_prefix + t for t in sources], normalize_embeddings=True)
        t = model.encode([p_prefix + x for x in targets], normalize_embeddings=True)
        return np.asarray(s @ t.T, dtype=np.float64)

    return similarity


_MODELS: dict[str, Any] = {}


def _cached_model(hf_name: str) -> Any:
    if hf_name not in _MODELS:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(hf_name, device="cpu")
        model.max_seq_length = 512
        _MODELS[hf_name] = model
    return _MODELS[hf_name]


def hybrid(*parts: Similarity, weights: tuple[float, ...] | None = None) -> Similarity:
    """Weighted sum of per-source min-max normalised similarities."""

    def similarity(sources: list[str], targets: list[str]) -> NDArray[np.float64]:
        total = np.zeros((len(sources), len(targets)))
        for w, part in zip(weights or (1.0,) * len(parts), parts, strict=True):
            m = part(sources, targets)
            lo, hi = m.min(axis=1, keepdims=True), m.max(axis=1, keepdims=True)
            total += w * (m - lo) / np.maximum(hi - lo, 1e-9)
        return total

    return similarity
