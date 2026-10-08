"""Cross-validation over projects for the PROMISE_exp classification tasks.

Folds are drawn over projects (stratified group k-fold), so a model never sees requirements of
the project it is tested on. Every requirement is predicted exactly once, by the model of the
fold that held its project out; scores are computed on the pooled predictions.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from spec_review.config import ROOT, SEED
from spec_review.data import promise

RESULTS = ROOT / "docs" / "results" / "classification"
N_FOLDS = 5


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    select: Callable[[pd.DataFrame], pd.DataFrame]
    target: Callable[[pd.DataFrame], pd.Series]


TASKS = {
    t.name: t
    for t in (
        Task("fr-nfr", "functional vs non-functional", lambda d: d,
             lambda d: d.functional.map({True: "F", False: "NFR"})),
        Task("nfr-11", "the 11 non-functional classes", lambda d: d[~d.functional],
             lambda d: d.label),
        Task("all-12", "functional and the 11 non-functional classes", lambda d: d,
             lambda d: d.label),
    )
}  # fmt: skip

# fit(train_texts, train_labels) -> predict(test_texts) -> labels
Method = Callable[[Sequence[str], Sequence[str]], Callable[[Sequence[str]], Sequence[str]]]


def folds(
    df: pd.DataFrame, y: pd.Series, seed: int = SEED, grouped: bool = True
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped by project; `grouped=False` gives the plain stratified folds common in the
    literature, kept only to measure how much they flatter a model."""
    if not grouped:
        plain = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        return list(plain.split(np.zeros(len(df)), y))
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(df)), y, df.project))


def bootstrap_macro_f1(y: np.ndarray, p: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        stats.append(f1_score(y[i], p[i], average="macro", zero_division=0))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate(
    task_name: str, method_name: str, method: Method, grouped: bool = True
) -> dict[str, Any]:
    task = TASKS[task_name]
    data = task.select(promise.load()).reset_index(drop=True)
    y = task.target(data).reset_index(drop=True)
    pred = np.empty(len(data), dtype=object)
    fold_of = np.empty(len(data), dtype=int)
    for k, (train, test) in enumerate(folds(data, y, grouped=grouped)):
        predict = method(data.text.iloc[train].tolist(), y.iloc[train].tolist())
        pred[test] = list(predict(data.text.iloc[test].tolist()))
        fold_of[test] = k
    y_arr = y.to_numpy()
    labels = sorted(set(y_arr))
    per_class = f1_score(y_arr, pred, labels=labels, average=None, zero_division=0)
    per_fold = [
        float(f1_score(y_arr[fold_of == k], pred[fold_of == k], average="macro", zero_division=0))
        for k in range(N_FOLDS)
    ]
    report = {
        "task": task_name,
        "method": method_name if grouped else f"{method_name}_random-folds",
        "folds": "grouped by project" if grouped else "random (projects shared)",
        "requirements": len(data),
        "projects": int(data.project.nunique()),
        "macro_f1": float(f1_score(y_arr, pred, average="macro", zero_division=0)),
        "macro_f1_ci": bootstrap_macro_f1(y_arr, pred),
        "weighted_f1": float(f1_score(y_arr, pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_arr, pred)),
        "macro_f1_per_fold": per_fold,
        "f1_per_class": {lab: float(v) for lab, v in zip(labels, per_class, strict=True)},
    }
    save(report, data, y_arr, pred)
    return report


def _round(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_round(x) for x in v]
    return v


def save(report: dict[str, Any], data: pd.DataFrame, y: np.ndarray, pred: np.ndarray) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"{report['task']}_{report['method']}"
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(_round(report), indent=2) + "\n")
    runs = ROOT / "runs" / "classification"
    runs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": data.id, "gold": y, "pred": pred}).to_parquet(runs / f"{name}.parquet")
    return path
