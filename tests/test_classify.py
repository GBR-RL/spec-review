import numpy as np
import pandas as pd

from spec_review.classify import baselines, cv


def _data() -> pd.DataFrame:
    rows = []
    for p in range(30):
        for i in range(8):
            functional = i % 2 == 0
            word = "display records" if functional else "respond within seconds"
            rows.append(
                {
                    "id": f"P{p}-{i}",
                    "project": p,
                    "text": f"The system shall {word} {p}",
                    "label": "F" if functional else "PE",
                    "functional": functional,
                }
            )
    return pd.DataFrame(rows)


def test_grouped_folds_never_share_a_project() -> None:
    df = _data()
    y = df.functional.map({True: "F", False: "NFR"})
    for train, test in cv.folds(df, y):
        assert not set(df.project.iloc[train]) & set(df.project.iloc[test])
    random_folds = cv.folds(df, y, grouped=False)
    shared = [set(df.project.iloc[a]) & set(df.project.iloc[b]) for a, b in random_folds]
    assert any(shared)


def test_every_requirement_predicted_once(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cv.promise, "load", _data)
    monkeypatch.setattr(cv, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(cv, "ROOT", tmp_path)
    report = cv.evaluate("fr-nfr", "tfidf_lr", baselines.tfidf_lr)
    assert report["requirements"] == 240
    assert report["macro_f1"] > 0.95
    assert len(report["macro_f1_per_fold"]) == cv.N_FOLDS
    preds = pd.read_parquet(tmp_path / "runs" / "classification" / "fr-nfr_tfidf_lr.parquet")
    assert preds.id.is_unique
    assert len(preds) == 240


def test_bootstrap_interval_brackets_the_score() -> None:
    y = np.array(["a", "b"] * 50)
    p = np.where(np.arange(100) % 5 == 0, "b", y)
    lo, hi = cv.bootstrap_macro_f1(y, p, n_boot=200)
    assert lo < 0.9 < hi
