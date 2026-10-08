"""Labelled requirement pairs for conflict detection (Malik et al., S3CDA).

UAV, WorldVista, PURE and OPENCOSS: every pair of requirements in each document set, labelled
Conflict or Neutral. The documents are public; most conflicts are synthetic, written by the
authors from existing requirements following INCOSE guidelines (typically a minimal edit such
as "current location" to "past location"), because real conflicts were too rare.

There is no official release. The files come from a research repository that redistributes
them; they are downloaded at run time from a pinned commit, checked by SHA-256, and never
committed here. The IBM-DOORS-derived CN/CDN sets in the same archive are not public and are
not used.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from spec_review.config import PROCESSED, RAW
from spec_review.data.download import Source, fetch

SOURCE = Source(
    "conflicts.zip",
    "https://github.com/Gechuyan/Enhancing-Requirement-Conflict-Detection-with-Data-Augmentation"
    "-and-Selective-Knowledge-Distillation/raw/24586ec53fd8ceea0a56ef6f250340c206f3f9c4/dataset.zip",
    "e8bfdf04b77d76e0eac076e8e8437279806f5452669228aeb31b6440993a6487",
)
DATASETS = ("uav", "worldvista", "pure", "opencoss")
OUT = PROCESSED / "conflict_pairs.parquet"


def build(raw: Path = RAW, out: Path = OUT) -> pd.DataFrame:
    path = fetch(SOURCE, raw)
    frames = []
    with zipfile.ZipFile(path) as z:
        for name in DATASETS:
            for part in ("train", "test"):  # the archive's split is not used; pairs are pooled
                df = pd.read_csv(io.BytesIO(z.read(f"{part}/{name}.csv")))
                frames.append(df.assign(dataset=name))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Text1": "a", "Text2": "b", "Class": "label"})[["dataset", "a", "b",
                                                                           "label"]]  # fmt: skip
    df["a"], df["b"] = df.a.str.strip().str.strip('"'), df.b.str.strip().str.strip('"')
    # Unordered pairs: (a, b) and (b, a) are the same pair.
    key = [tuple(sorted(p)) for p in zip(df.a, df.b, strict=True)]
    df["pair"] = key
    df["conflict"] = df.label.str.lower().eq("conflict")
    df = df.sort_values("conflict", ascending=False).drop_duplicates(["dataset", "pair"])
    df = df.drop(columns="pair").sort_values(["dataset", "a", "b"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def load(path: Path = OUT) -> pd.DataFrame:
    return pd.read_parquet(path)
