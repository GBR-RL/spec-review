"""CoEST traceability datasets as two tables: artifacts and ground-truth trace links.

- CM1-NASA: high-level to low-level requirements (NASA science instrument).
- eTOUR: use cases to code classes (tourism system, Italian identifiers).
- iTrust: use cases to Java and JSP code (medical records).
- EasyClinic: use cases, interaction diagrams, test cases and code classes, with links between
  every pair of artifact types (clinic management, translated from Italian).

Source code artifacts are reduced to their text: identifiers split on camelCase, comments and
string literals kept.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from spec_review.config import PROCESSED, RAW

COEST = RAW / "coest"
ARTIFACTS = PROCESSED / "coest_artifacts.parquet"
LINKS = PROCESSED / "coest_links.parquet"

# (dataset, source xml, target xml, answer xml, source kind, target kind), paths below the
# dataset's extracted folder.
XML_SETS = (
    ("CM1", "CM1-sourceArtifacts.xml", "CM1-targetArtifacts.xml", "CM1-answerSet.xml",
     "high-level requirement", "low-level requirement"),
    ("eTOUR", "eTOUR/source_req.xml", "eTOUR/target_code.xml", "eTOUR/answer_req_code.xml",
     "use case", "code"),
    ("iTrust", "iTrust/source_req.xml", "iTrust/target_code.xml", "iTrust/answer_req_code.xml",
     "use case", "code"),
)  # fmt: skip
EASYCLINIC_KINDS = {
    "UC": ("1 - use cases", "use case"),
    "ID": ("2 - Interaction diagrams", "interaction diagram"),
    "TC": ("3 - test cases", "test case"),
    "CC": ("4 - class description", "code"),
}
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def code_to_text(code: str) -> str:
    """Identifiers split on camelCase and underscores; comments and strings kept as words."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", code)
    split = (_CAMEL.sub(" ", w).replace("_", " ") for w in words)
    return " ".join(" ".join(split).split())


def _artifacts(xml: Path, base: Path, is_code: bool) -> Iterator[tuple[str, str]]:
    root = ET.fromstring(read_text(xml).lstrip("﻿"))
    external = (root.findtext("collection_info/content_location") or "").strip() == "external"
    for a in root.iter("artifact"):
        aid = (a.findtext("id") or "").strip()
        content = (a.findtext("content") or "").strip()
        if external:
            path = base / content
            if not path.exists():
                continue
            content = read_text(path)
            if is_code:
                content = code_to_text(content)
        yield aid, " ".join(content.split())


def _links(xml: Path) -> Iterator[tuple[str, str]]:
    root = ET.fromstring(read_text(xml).lstrip("﻿"))
    for link in root.iter("link"):
        yield (
            (link.findtext("source_artifact_id") or "").strip(),
            (link.findtext("target_artifact_id") or "").strip(),
        )


def _xml_dataset(
    spec: tuple[str, str, str, str, str, str], base: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    name, src, tgt, ans, src_kind, tgt_kind = spec
    files = {rel: next(base.rglob(Path(rel).name)) for rel in (src, tgt, ans)}
    arts = []
    for rel, kind in ((src, src_kind), (tgt, tgt_kind)):
        folder = files[rel].parent
        for aid, text in _artifacts(files[rel], folder, is_code=kind == "code"):
            arts.append({"dataset": name, "id": aid, "kind": kind, "text": text})
    links = [{"dataset": name, "source": s, "target": t} for s, t in _links(files[ans])]
    return arts, links


def _easyclinic(base: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    docs = next(base.rglob("2 - docs (English)"))
    oracle = next(base.rglob("oracle"))
    arts = []
    for prefix, (folder, kind) in EASYCLINIC_KINDS.items():
        for f in sorted((docs / folder).glob("*.txt")):
            text = read_text(f)
            arts.append(
                {"dataset": "EasyClinic", "id": f"{prefix}{f.stem}", "kind": kind,
                 "text": " ".join(text.split())}
            )  # fmt: skip
    links: list[dict[str, str]] = []
    for f in sorted(oracle.glob("*_*.txt")):
        src_prefix, tgt_prefix = f.stem.split("_")
        if src_prefix not in EASYCLINIC_KINDS or tgt_prefix not in EASYCLINIC_KINDS:
            continue  # e.g. the "SimpleUC" variant
        for line in read_text(f).splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            src = f"{src_prefix}{Path(parts[0]).stem}"
            links.extend(
                {"dataset": "EasyClinic", "source": src, "target": f"{tgt_prefix}{Path(p).stem}"}
                for p in parts[1:]
            )
    return arts, links


def build(
    coest: Path = COEST, artifacts_out: Path = ARTIFACTS, links_out: Path = LINKS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    arts: list[dict[str, str]] = []
    links: list[dict[str, str]] = []
    for spec in XML_SETS:
        folder = coest / ("CM1-NASA" if spec[0] == "CM1" else spec[0])
        a, lk = _xml_dataset(spec, folder)
        arts += a
        links += lk
    a, lk = _easyclinic(coest / "EasyClinic")
    arts += a
    links += lk
    artifacts = pd.DataFrame(arts).drop_duplicates(["dataset", "id"])
    trace = pd.DataFrame(links).drop_duplicates()
    # Keep only links whose both ends exist as artifacts.
    known = set(zip(artifacts.dataset, artifacts.id, strict=True))
    ok = [
        (d, s) in known and (d, t) in known
        for d, s, t in zip(trace.dataset, trace.source, trace.target, strict=True)
    ]
    trace = trace[ok].reset_index(drop=True)
    artifacts_out.parent.mkdir(parents=True, exist_ok=True)
    artifacts.to_parquet(artifacts_out, index=False)
    trace.to_parquet(links_out, index=False)
    return artifacts, trace


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(ARTIFACTS), pd.read_parquet(LINKS)
