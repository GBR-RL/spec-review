"""PROMISE_exp: 969 requirements from 48 projects, labelled functional or one of 11 NFR classes.

The split is drawn over projects, not requirements: requirements of one project share
vocabulary, templates and authors, so a random split would let a classifier recognise the
project rather than the requirement type.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from spec_review.config import PROCESSED, RAW, SEED

CLASSES = {
    "F": "functional",
    "A": "availability",
    "FT": "fault tolerance",
    "L": "legal",
    "LF": "look and feel",
    "MN": "maintainability",
    "O": "operability",
    "PE": "performance",
    "PO": "portability",
    "SC": "scalability",
    "SE": "security",
    "US": "usability",
}
OUT = PROCESSED / "promise.parquet"


def parse_arff(text: str) -> pd.DataFrame:
    """Rows of the @DATA section: project id, quoted requirement text, class code."""
    body = text.split("@DATA", 1)[1]
    rows = []
    reader = csv.reader(io.StringIO(body), quotechar="'", skipinitialspace=True)
    for row in reader:
        if not row or row[0].startswith("%"):
            continue
        project, requirement, label = row[0].strip(), ",".join(row[1:-1]), row[-1].strip()
        rows.append(
            {"project": int(project), "text": " ".join(requirement.split()), "label": label}
        )
    df = pd.DataFrame(rows)
    unknown = set(df.label) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown classes {unknown}")
    df["functional"] = df.label == "F"
    df.insert(0, "id", [f"P{p:02d}-{i:03d}" for i, p in enumerate(df.project)])
    return df


def assign_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """About 70/10/20 over projects, stratified on functional vs non-functional."""
    df = df.copy()
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest, test = next(folds.split(np.zeros(len(df)), df.functional, df.project))
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
    sub = df.iloc[rest]
    train, dev = next(inner.split(np.zeros(len(sub)), sub.functional, sub.project))
    split = np.full(len(df), "", dtype=object)
    split[test] = "test"
    split[rest[train]] = "train"
    split[rest[dev]] = "dev"
    df["split"] = split
    return df


def build(raw: Path = RAW, out: Path = OUT) -> pd.DataFrame:
    text = (raw / "PROMISE_exp.arff").read_text(encoding="utf-8", errors="replace")
    df = assign_splits(parse_arff(text))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def load(path: Path = OUT) -> pd.DataFrame:
    return pd.read_parquet(path)
