from pathlib import Path

from spec_review.quality import report

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_report_highlights_findings_and_escapes_text(tmp_path: Path) -> None:
    f = tmp_path / "r.req"
    f.write_text("R-1: The <pump> should start quickly.\nR-2: The valve shall close in 1 s.\n",
                 encoding="utf-8")  # fmt: skip
    out = report.write([f, EXAMPLES / "pump.reqif"], tmp_path / "report.html")
    page = out.read_text(encoding="utf-8")
    assert "<mark>quickly</mark>" in page
    assert "&lt;pump&gt;" in page
    assert "no findings" in page
    assert "pump.reqif" in page
