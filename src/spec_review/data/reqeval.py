"""NLP4RE ReqEval: is the marked pronoun in a requirement ambiguous?

Each row is a requirement with one pronoun marked, labelled NOCUOUS (readers may resolve it
differently) or INNOCUOUS (with the antecedent readers agree on). It covers one kind of
ambiguity only, referential ambiguity, and the reviewer is scored on exactly that.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from spec_review.config import PROCESSED, RAW

OUT = PROCESSED / "reqeval.parquet"
_MARK = re.compile(r"<referential>(.*?)</referential>", re.IGNORECASE)


def parse(path: Path, split: str) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3 or not parts[0].strip():
            continue
        rid, marked, label = parts[0].strip(), parts[1].strip(), parts[2].strip().upper()
        antecedent = parts[3].strip() if len(parts) > 3 else ""
        m = _MARK.search(marked)
        if m is None or label not in ("NOCUOUS", "INNOCUOUS"):
            raise ValueError(f"unexpected ReqEval row {rid!r}")
        rows.append(
            {
                "id": rid,
                "split": split,
                "marked": marked,
                "text": _MARK.sub(r"\1", marked),
                "pronoun": m.group(1),
                "ambiguous": label == "NOCUOUS",
                "antecedent": antecedent,
                "domain": rid.split("#")[0],
            }
        )
    return pd.DataFrame(rows)


def build(raw: Path = RAW, out: Path = OUT) -> pd.DataFrame:
    df = pd.concat(
        [parse(raw / "reqeval_training.tsv", "train"), parse(raw / "reqeval_test.tsv", "test")],
        ignore_index=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def load(path: Path = OUT) -> pd.DataFrame:
    return pd.read_parquet(path)
