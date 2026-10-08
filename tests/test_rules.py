import pytest

from spec_review.quality import rules


def _rules(text: str) -> list[str]:
    return [f.rule for f in rules.check(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The system shall respond quickly.", "vague-term"),
        ("The system shall log errors if possible.", "escape-clause"),
        ("The system shall export CSV, PDF, etc.", "open-ended"),
        ("The system should encrypt passwords.", "weak-modal"),
        ("The display refreshes every second.", "no-modal"),
        ("The operator shall start and/or stop the pump.", "and-or"),
        ("The timeout shall be TBD.", "placeholder"),
        ("The pump shall stop when it overheats.", "pronoun"),
        ("All passwords shall be encrypted.", "passive-no-agent"),
        ("The system shall log in users. The system shall log out idle users.", "multiple"),
        ("The new version shall load pages faster.", "comparative"),
    ],
)
def test_each_rule_fires(text: str, expected: str) -> None:
    assert expected in _rules(text)


@pytest.mark.parametrize(
    "text",
    [
        "The controller shall close the valve within 200 ms of a pressure above 8 bar.",
        "Passwords shall be encrypted by the authentication service with AES-256.",
        "The system shall load a page in less than 2 s, faster than version 1.",
        "The system shall support TCP/IP and report N/A for missing values.",
        "The HMI shall display the dosing rate in ml/min and the speed in km/h.",
    ],
)
def test_clean_requirements_pass(text: str) -> None:
    assert rules.check(text) == []


def test_findings_carry_positions() -> None:
    text = "The system shall respond quickly."
    f = rules.check(text)[0]
    assert text[f.start : f.end] == f.text == "quickly"


def test_rule_selection() -> None:
    text = "The system should respond quickly."
    assert {f.rule for f in rules.check(text, rules=("vague-term",))} == {"vague-term"}


def test_lint_file_and_github_annotations(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from spec_review.quality.lint import lint, render

    f = tmp_path / "pump.req"
    f.write_text(
        "# heading\n\nPUMP-01: The pump should start quickly.\n"
        "- [PUMP-02] The controller shall stop the pump within 2 s.\n",
        encoding="utf-8",
    )
    reqs = lint(f)
    assert [r.id for r in reqs] == ["PUMP-01", "PUMP-02"]
    assert [r.line for r in reqs] == [3, 4]
    assert reqs[1].findings == []
    out = render(reqs, f, "github")
    assert out.startswith("::warning file=")
    assert ",line=3," in out
    assert "PUMP-01" in out
