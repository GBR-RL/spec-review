from pathlib import Path

from spec_review import reqif
from spec_review.quality.lint import lint, render

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "pump.reqif"


def test_reads_ids_and_xhtml_text() -> None:
    objects = reqif.read(EXAMPLE)
    assert [o.id for o in objects] == ["PUMP-01", "PUMP-02"]
    assert objects[0].text.startswith("The controller shall stop the pump within 2 s")
    assert "<" not in objects[0].text


def test_round_trip(tmp_path: Path) -> None:
    rows = [
        {"ID": "R-1", "Text": "The pump should start quickly.", "spec-review findings": "vague"},
        {"ID": "R-2", "Text": "The valve shall close in 1 s.", "spec-review findings": ""},
    ]
    path = reqif.write(tmp_path / "out.reqif", rows)
    back = reqif.read(path)
    assert [(o.id, o.text) for o in back] == [(r["ID"], r["Text"]) for r in rows]
    assert back[0].attributes["spec-review findings"] == "vague"
    assert reqif.looks_like_reqif(path)


def test_lint_points_at_the_spec_object_line() -> None:
    reqs = lint(EXAMPLE)
    assert [r.id for r in reqs] == ["PUMP-01", "PUMP-02"]
    assert reqs[0].findings == []
    assert {f.rule for f in reqs[1].findings} == {"weak-modal", "vague-term", "escape-clause"}
    source = EXAMPLE.read_text(encoding="utf-8").splitlines()
    assert 'IDENTIFIER="so-2"' in source[reqs[1].line - 1]
    assert ",line=" in render(reqs, EXAMPLE, "github")
