"""A static HTML report of the rule findings for one or more requirements files."""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

from spec_review.quality.lint import Requirement, lint

RULES = {
    "vague-term": "Not measurable",
    "escape-clause": "Lets the requirement be skipped",
    "open-ended": "Open-ended list",
    "weak-modal": "No binding 'shall'",
    "no-modal": "No modal verb at all",
    "and-or": "'and/or' or a slash",
    "placeholder": "Undecided content (TBD)",
    "pronoun": "Pronoun instead of a name",
    "passive-no-agent": "Passive voice without an actor",
    "multiple": "More than one requirement",
    "comparative": "Comparative without a reference",
}

STYLE = """
:root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#5d6675; --line:#dfe3ea;
        --mark:#fde8b0; --ok:#1baf7a; --warn:#b26b00; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171c; --card:#1d2128; --ink:#e7eaf0; --muted:#9aa3b2; --line:#2c323c;
          --mark:#5a4512; --ok:#199e70; --warn:#e0a640; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width:1000px; margin:0 auto; padding:24px 16px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
.muted { color:var(--muted); }
.summary { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
.chip { border:1px solid var(--line); border-radius:999px; padding:2px 10px; font-size:13px;
        background:var(--card); }
table { width:100%; border-collapse:collapse; background:var(--card);
        border:1px solid var(--line); border-radius:10px; overflow:hidden; }
td, th { text-align:left; vertical-align:top; padding:10px 12px; border-top:1px solid var(--line); }
th { font-size:13px; color:var(--muted); font-weight:600; border-top:none; }
td.id { white-space:nowrap; font-family:ui-monospace, Consolas, monospace; font-size:13px; }
mark { background:var(--mark); color:inherit; border-radius:3px; padding:0 2px; }
ul { margin:0; padding-left:18px; } li { color:var(--warn); }
.clean { color:var(--ok); }
"""


def _highlight(req: Requirement) -> str:
    spans = sorted({(f.start, f.end) for f in req.findings if f.end > f.start})
    out, pos = [], 0
    for start, end in spans:
        if start < pos:
            continue
        out.append(html.escape(req.text[pos:start]))
        out.append(f"<mark>{html.escape(req.text[start:end])}</mark>")
        pos = end
    out.append(html.escape(req.text[pos:]))
    return "".join(out)


def render(files: dict[Path, list[Requirement]]) -> str:
    total = sum(len(r) for r in files.values())
    flagged = sum(bool(r.findings) for rs in files.values() for r in rs)
    counts = Counter(f.rule for rs in files.values() for r in rs for f in r.findings)
    chips = "".join(
        f'<span class="chip">{html.escape(RULES.get(rule, rule))}: {n}</span>'
        for rule, n in counts.most_common()
    )
    sections = []
    for path, reqs in files.items():
        rows = []
        for r in reqs:
            if r.findings:
                items = "".join(f"<li>{html.escape(f.message)}</li>" for f in r.findings)
                verdict = f"<ul>{items}</ul>"
            else:
                verdict = '<span class="clean">no findings</span>'
            rows.append(
                f'<tr><td class="id">{html.escape(r.id)}<br><span class="muted">line {r.line}'
                f"</span></td><td>{_highlight(r)}</td><td>{verdict}</td></tr>"
            )
        sections.append(
            f"<h2>{html.escape(path.as_posix())}</h2><table><tr><th>ID</th><th>Requirement</th>"
            f"<th>Findings</th></tr>{''.join(rows)}</table>"
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Requirements review</title><style>{STYLE}</style></head><body><main>"
        f"<h1>Requirements review</h1><p class='muted'>{flagged} of {total} requirements "
        f"have findings, checked against INCOSE and ISO/IEC/IEEE 29148 writing rules.</p>"
        f'<div class="summary">{chips}</div>{"".join(sections)}</main></body></html>'
    )


def write(paths: list[Path], out: Path) -> Path:
    out.write_text(render({p: lint(p) for p in paths}), encoding="utf-8")
    return out
