"""Find conflicting requirement pairs: rank every pair, score the ranking.

Conflicts are rare (83 in 26,431 pairs), so the question a reviewer asks is "how far down the
list do I have to read?". Rankings are scored with average precision per document set, the
precision of the top 10, and recall within the top 2n pairs (n = number of true conflicts).

Scorers: lexical and embedding similarity (conflicts are usually near-identical sentences that
differ in one value or qualifier), the contradiction probability of a natural-language-inference
cross-encoder, and their product.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score

from spec_review.config import ROOT, RUNS
from spec_review.data import conflicts

RESULTS = ROOT / "docs" / "results" / "conflicts"
NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"

# scorer(pairs of one document set) -> one score per pair, higher = more likely a conflict
Scorer = Callable[[pd.DataFrame], NDArray[np.float64]]


def tfidf_similarity(pairs: pd.DataFrame) -> NDArray[np.float64]:
    texts = sorted(set(pairs.a) | set(pairs.b))
    vec = TfidfVectorizer(sublinear_tf=True).fit(texts)
    a, b = vec.transform(pairs.a), vec.transform(pairs.b)
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def embedding_similarity(model_name: str = "BAAI/bge-m3") -> Scorer:
    def score(pairs: pd.DataFrame) -> NDArray[np.float64]:
        from sentence_transformers import SentenceTransformer

        texts = sorted(set(pairs.a) | set(pairs.b))
        model = SentenceTransformer(model_name, device="cpu")
        vectors = model.encode(texts, normalize_embeddings=True)
        index = {t: i for i, t in enumerate(texts)}
        va = vectors[[index[t] for t in pairs.a]]
        vb = vectors[[index[t] for t in pairs.b]]
        return np.asarray((va * vb).sum(axis=1), dtype=np.float64)

    return score


def nli_contradiction(model_name: str = NLI_MODEL) -> Scorer:
    """Mean contradiction probability over both directions of the pair."""

    def score(pairs: pd.DataFrame) -> NDArray[np.float64]:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name, device="cpu")
        labels = [model.config.id2label[i].lower() for i in range(len(model.config.id2label))]
        c = labels.index("contradiction")
        forward = model.predict(list(zip(pairs.a, pairs.b, strict=True)), batch_size=64,
                                apply_softmax=True)  # fmt: skip
        backward = model.predict(list(zip(pairs.b, pairs.a, strict=True)), batch_size=64,
                                 apply_softmax=True)  # fmt: skip
        return np.asarray((forward[:, c] + backward[:, c]) / 2, dtype=np.float64)

    return score


def ranking_scores(y: NDArray[np.bool_], score: NDArray[np.float64]) -> dict[str, Any]:
    order = np.argsort(-score, kind="stable")
    n = int(y.sum())
    return {
        "pairs": len(y),
        "conflicts": n,
        "average_precision": float(average_precision_score(y, score)),
        "precision@10": float(y[order[:10]].mean()),
        "recall@2n": float(y[order[: 2 * n]].sum() / n),
        "rank_of_first_conflict": int(np.argmax(y[order]) + 1),
    }


def evaluate(name: str, scorer: Scorer) -> dict[str, Any]:
    df = conflicts.load()
    report: dict[str, Any] = {"method": name, "datasets": {}}
    scored = []
    for dataset, group in df.groupby("dataset"):
        pairs = group.reset_index(drop=True)
        s = scorer(pairs)
        report["datasets"][str(dataset)] = ranking_scores(pairs.conflict.to_numpy(), s)
        scored.append(pairs.assign(score=s))
    report["mean_average_precision"] = float(
        np.mean([d["average_precision"] for d in report["datasets"].values()])
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(_round(report), indent=2) + "\n")
    runs = RUNS / "conflicts"
    runs.mkdir(parents=True, exist_ok=True)
    pd.concat(scored).to_parquet(runs / f"{name}.parquet", index=False)
    return report


def product(*scorers: Scorer) -> Scorer:
    def score(pairs: pd.DataFrame) -> NDArray[np.float64]:
        out = np.ones(len(pairs))
        for s in scorers:
            v = s(pairs)
            out *= (v - v.min()) / max(float(v.max() - v.min()), 1e-9)
        return out

    return score


def _round(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    return v
