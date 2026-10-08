import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spec_review.conflicts import detect
from spec_review.data import conflicts


def _archive(path: Path) -> Path:
    rows = pd.DataFrame(
        {
            "Text1": ['"The GCS shall show the current location."', "The UAV shall log flights."],
            "Text2": [
                "The GCS shall show the past location.",
                "The GCS shall show the current location.",
            ],
            "Class": ["Conflict", "Neutral"],
        }
    )
    swapped = rows.iloc[[0]].rename(columns={"Text1": "Text2", "Text2": "Text1"})
    with zipfile.ZipFile(path, "w") as z:
        for name in conflicts.DATASETS:
            for part in ("train", "test"):
                buf = io.StringIO()
                df = rows if part == "train" else swapped.assign(Class="Neutral")
                df.to_csv(buf, index=False)
                z.writestr(f"{part}/{name}.csv", buf.getvalue())
    return path


def test_pairs_are_unordered_and_conflict_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path / "conflicts.zip")
    monkeypatch.setattr(conflicts, "fetch", lambda source, raw: archive)
    df = conflicts.build(tmp_path, tmp_path / "pairs.parquet")
    uav = df[df.dataset == "uav"]
    assert len(uav) == 2  # the swapped duplicate collapses into its pair
    assert uav.conflict.sum() == 1  # and the Conflict label wins over Neutral
    assert not uav.a.str.startswith('"').any()


def test_ranking_scores() -> None:
    y = np.array([False, True, False, True])
    s = detect.ranking_scores(y, np.array([0.9, 0.8, 0.1, 0.7]))
    assert s["conflicts"] == 2
    assert s["rank_of_first_conflict"] == 2
    assert s["recall@2n"] == 1.0
    assert 0 < s["average_precision"] < 1


def test_near_identical_pairs_score_highest_with_tfidf() -> None:
    pairs = pd.DataFrame(
        {
            "a": ["The GCS shall show the current location.", "The UAV shall log flights."],
            "b": ["The GCS shall show the past location.", "Operators shall wear helmets."],
        }
    )
    s = detect.tfidf_similarity(pairs)
    assert s[0] > s[1]


def test_product_normalises_each_scorer() -> None:
    pairs = pd.DataFrame({"a": ["x", "y", "z"], "b": ["x", "y", "z"]})
    combined = detect.product(
        lambda p: np.array([1.0, 2.0, 3.0]), lambda p: np.array([10.0, 30.0, 20.0])
    )
    assert combined(pairs).tolist() == pytest.approx([0.0, 0.5, 0.5])
