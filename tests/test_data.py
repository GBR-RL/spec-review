from pathlib import Path

import pandas as pd
import pytest

from spec_review.config import PROCESSED
from spec_review.data import coest, promise, reqeval

ARFF = """@RELATION nfr
@ATTRIBUTE ProjectID {1,2}
@ATTRIBUTE RequirementText string
@ATTRIBUTE _class_ {F,PE,SE}
@DATA
1,'The system shall refresh the display every 60 seconds.',PE
1,'The system shall let users log in, view and edit profiles.',F
% a comment line
2,'Only managers shall  see salary data.',SE
"""


def test_parse_arff_keeps_commas_inside_quotes() -> None:
    df = promise.parse_arff(ARFF)
    assert len(df) == 3
    assert df.text.iloc[1] == "The system shall let users log in, view and edit profiles."
    assert df.text.iloc[2] == "Only managers shall see salary data."
    assert df.functional.tolist() == [False, True, False]
    assert df.id.is_unique


def test_project_split_keeps_projects_together() -> None:
    rows = [
        {"id": f"P{p}-{i}", "project": p, "text": "x", "label": "F" if i % 2 else "PE",
         "functional": bool(i % 2)}
        for p in range(40) for i in range(6)
    ]  # fmt: skip
    df = promise.assign_splits(pd.DataFrame(rows), seed=1)
    assert df.groupby("project").split.nunique().max() == 1
    assert set(df.split) == {"train", "dev", "test"}


def test_reqeval_rows(tmp_path: Path) -> None:
    f = tmp_path / "t.tsv"
    f.write_text(
        "ID\tSentence\tDetected as\tResolved as\n"
        "lib#1\tThe system stores files so <referential>they</referential> are safe.\tNOCUOUS\t\n"
        "lib#2\tThe user saves <referential>it</referential> once.\tINNOCUOUS\tthe file\n",
        encoding="utf-8",
    )
    df = reqeval.parse(f, "train")
    assert df.ambiguous.tolist() == [True, False]
    assert df.pronoun.tolist() == ["they", "it"]
    assert "<referential>" not in df.text.iloc[0]
    assert df.antecedent.iloc[1] == "the file"


def test_code_to_text_splits_identifiers() -> None:
    text = coest.code_to_text("public class AddApptAction { int patient_id; // store it }")
    assert text == "public class Add Appt Action int patient id store it"


def test_easyclinic_oracle(tmp_path: Path) -> None:
    docs = tmp_path / "EasyClinic" / "2 - docs (English)"
    for folder in ("1 - use cases", "3 - test cases"):
        (docs / folder).mkdir(parents=True)
    (docs / "1 - use cases" / "1.txt").write_text("Book a visit", encoding="utf-8")
    (docs / "3 - test cases" / "51.txt").write_text("Test booking", encoding="utf-8")
    oracle = tmp_path / "EasyClinic" / "oracle"
    oracle.mkdir()
    (oracle / "UC_TC.txt").write_text("1.txt 51.txt\n2.txt\n", encoding="utf-8")
    (oracle / "UC_ID.txt").write_text("1.txt:51.txt\n", encoding="utf-8")
    for folder in ("2 - Interaction diagrams", "4 - class description"):
        (docs / folder).mkdir()
    arts, links = coest._easyclinic(tmp_path / "EasyClinic")
    assert {a["id"] for a in arts} == {"UC1", "TC51"}
    assert {"dataset": "EasyClinic", "source": "UC1", "target": "TC51"} in links
    assert {"dataset": "EasyClinic", "source": "UC1", "target": "ID51"} in links


@pytest.mark.data
def test_prepared_datasets() -> None:
    if not (PROCESSED / "promise.parquet").exists():
        pytest.skip("run `spec-review data` first")
    p = promise.load()
    assert len(p) == 969
    assert p.groupby("project").split.nunique().max() == 1
    r = reqeval.load()
    assert r.id.is_unique
    arts, links = coest.load()
    assert set(arts.dataset) == {"CM1", "EasyClinic", "eTOUR", "iTrust"}
    assert not links.duplicated().any()
