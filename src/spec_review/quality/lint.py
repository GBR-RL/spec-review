"""Lint a requirements file: one requirement per line, optionally prefixed with an ID.

    REQ-12: The pump controller shall stop the pump within 2 s of a pressure above 8 bar.

Blank lines, Markdown headings and lines starting with '#' or '//' are skipped. Output formats:
plain text, JSON, and GitHub workflow annotations (inline warnings on a pull request).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from spec_review.quality.rules import Finding, check

_ID_TOKEN = r"[A-Za-z][\w.-]*\d[\w.-]*"
# "[REQ-12] text", "[REQ-12]: text", "REQ-12: text" or "REQ-12 | text", optionally as a list item.
_ID = re.compile(
    rf"^\s*(?:[-*]\s+)?(?:\[(?P<b>{_ID_TOKEN})\]\s*[:|]?\s*|(?P<p>{_ID_TOKEN})\s*[:|]\s+)"
)


@dataclass
class Requirement:
    id: str
    text: str
    line: int
    column: int
    findings: list[Finding] = field(default_factory=list)


def parse(text: str) -> list[Requirement]:
    reqs = []
    for n, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        m = _ID.match(raw)
        rid = (m.group("b") or m.group("p")) if m else f"L{n}"
        offset = m.end() if m else len(raw) - len(raw.lstrip())
        reqs.append(Requirement(rid, raw[offset:].rstrip(), n, offset + 1))
    return reqs


def parse_reqif(path: Path) -> list[Requirement]:
    """Requirements from a ReqIF file; the line is where the SPEC-OBJECT starts."""
    from spec_review import reqif

    source = path.read_text(encoding="utf-8", errors="replace").splitlines()
    reqs = []
    for obj in reqif.read(path):
        line = next(
            (n for n, text in enumerate(source, 1) if f'IDENTIFIER="{obj.identifier}"' in text), 1
        )
        reqs.append(Requirement(obj.id, obj.text, line, 1))
    return reqs


def lint(path: Path) -> list[Requirement]:
    from spec_review import reqif

    if reqif.looks_like_reqif(path):
        reqs = parse_reqif(path)
    else:
        reqs = parse(path.read_text(encoding="utf-8"))
    for r in reqs:
        r.findings = check(r.text)
    return reqs


def render(reqs: list[Requirement], path: Path, fmt: str = "text") -> str:
    if fmt == "json":
        return json.dumps(
            [
                {"id": r.id, "line": r.line, "text": r.text,
                 "findings": [f.as_dict() for f in r.findings]}
                for r in reqs
            ],
            indent=2,
        )  # fmt: skip
    lines = []
    for r in reqs:
        for f in r.findings:
            col = r.column + f.start
            if fmt == "github":
                msg = f"{r.id}: {f.message} ({f.text!r})" if f.text else f"{r.id}: {f.message}"
                lines.append(
                    f"::warning file={path.as_posix()},line={r.line},col={col},"
                    f"title=spec-review {f.rule}::{msg}"
                )
            else:
                quote = f" '{f.text}'" if f.text else ""
                lines.append(f"{path}:{r.line}:{col}: {f.rule}{quote}: {f.message} [{r.id}]")
    return "\n".join(lines)
