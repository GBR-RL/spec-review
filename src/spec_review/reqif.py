"""Read and write ReqIF, the OMG exchange format of requirements tools (DOORS, Polarion, Jama).

Reading maps every SPEC-OBJECT to an identifier and a text: the identifier is the value of an
attribute named like "ReqIF.ForeignID" or "ID" (falling back to the object's IDENTIFIER), the
text the value of "ReqIF.Text" / "ReqIF.Description" / "Text" (falling back to the longest
string or XHTML value). Writing produces a minimal, valid ReqIF 1.2 document with one string
attribute per field, so review findings can travel back into a requirements tool.
"""

from __future__ import annotations

import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

NS = "http://www.omg.org/spec/ReqIF/20110401/reqif.xsd"
XHTML = "http://www.w3.org/1999/xhtml"
ID_NAMES = ("reqif.foreignid", "foreignid", "id", "identifier", "req id")
TEXT_NAMES = ("reqif.text", "reqif.description", "text", "description", "object text")


@dataclass
class SpecObject:
    identifier: str
    id: str
    text: str
    attributes: dict[str, str] = field(default_factory=dict)


def _q(tag: str) -> str:
    return f"{{{NS}}}{tag}"


def _text_of(value: ET.Element) -> str:
    if value.tag == _q("ATTRIBUTE-VALUE-XHTML"):
        the_value = value.find(_q("THE-VALUE"))
        raw = "" if the_value is None else "".join(the_value.itertext())
    else:
        raw = value.get("THE-VALUE", "")
    return " ".join(raw.split())


def read(path: Path) -> list[SpecObject]:
    root = ET.parse(path).getroot()
    names: dict[str, str] = {}
    for definition in root.iter():
        if definition.tag.startswith(_q("ATTRIBUTE-DEFINITION-")) and definition.get("IDENTIFIER"):
            names[definition.get("IDENTIFIER", "")] = definition.get("LONG-NAME", "")
    objects = []
    for obj in root.iter(_q("SPEC-OBJECT")):
        attrs: dict[str, str] = {}
        values = obj.find(_q("VALUES"))
        for value in [] if values is None else list(values):
            ref = value.find(f"{_q('DEFINITION')}/*")
            name = names.get(ref.text or "", ref.text or "") if ref is not None else ""
            attrs[name or f"attr{len(attrs)}"] = _text_of(value)
        lowered = {k.lower(): v for k, v in attrs.items()}
        rid = next((lowered[n] for n in ID_NAMES if lowered.get(n)), obj.get("IDENTIFIER", ""))
        text = next((lowered[n] for n in TEXT_NAMES if lowered.get(n)), "")
        if not text and attrs:
            text = max(attrs.values(), key=len)
        if text:
            objects.append(SpecObject(obj.get("IDENTIFIER", ""), rid, text, attrs))
    return objects


def _ident(prefix: str) -> str:
    return f"_{prefix}-{uuid.uuid4().hex[:12]}"


def write(path: Path, rows: list[dict[str, str]], title: str = "spec-review export") -> Path:
    """One SPEC-OBJECT per row; every key becomes a string attribute (first key = ID)."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    ET.register_namespace("", NS)

    def el(parent: ET.Element, tag: str, attrs: dict[str, str] | None = None) -> ET.Element:
        return ET.SubElement(parent, _q(tag), attrs or {})

    root = ET.Element(_q("REQ-IF"))
    header = el(el(root, "THE-HEADER"), "REQ-IF-HEADER", {"IDENTIFIER": _ident("header")})
    for tag, text in (
        ("CREATION-TIME", now),
        ("REQ-IF-TOOL-ID", "spec-review"),
        ("REQ-IF-VERSION", "1.0"),
        ("SOURCE-TOOL-ID", "spec-review"),
        ("TITLE", title),
    ):
        el(header, tag).text = text
    content = el(el(root, "CORE-CONTENT"), "REQ-IF-CONTENT")
    datatype_id, type_id = _ident("string"), _ident("type")
    el(
        el(content, "DATATYPES"),
        "DATATYPE-DEFINITION-STRING",
        {"IDENTIFIER": datatype_id, "LAST-CHANGE": now, "LONG-NAME": "String",
         "MAX-LENGTH": "10000"},
    )  # fmt: skip
    spec_type = el(
        el(content, "SPEC-TYPES"),
        "SPEC-OBJECT-TYPE",
        {"IDENTIFIER": type_id, "LAST-CHANGE": now, "LONG-NAME": "Requirement"},
    )
    spec_attrs = el(spec_type, "SPEC-ATTRIBUTES")
    attr_ids = {name: _ident("attr") for name in fields}
    for name, attr_id in attr_ids.items():
        definition = el(
            spec_attrs,
            "ATTRIBUTE-DEFINITION-STRING",
            {"IDENTIFIER": attr_id, "LAST-CHANGE": now, "LONG-NAME": name},
        )
        el(el(definition, "TYPE"), "DATATYPE-DEFINITION-STRING-REF").text = datatype_id
    objects = el(content, "SPEC-OBJECTS")
    for row in rows:
        obj = el(objects, "SPEC-OBJECT", {"IDENTIFIER": _ident("req"), "LAST-CHANGE": now})
        el(el(obj, "TYPE"), "SPEC-OBJECT-TYPE-REF").text = type_id
        values = el(obj, "VALUES")
        for name, attr_id in attr_ids.items():
            value = el(values, "ATTRIBUTE-VALUE-STRING", {"THE-VALUE": str(row.get(name, ""))})
            el(el(value, "DEFINITION"), "ATTRIBUTE-DEFINITION-STRING-REF").text = attr_id
    el(content, "SPECIFICATIONS")
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def looks_like_reqif(path: Path) -> bool:
    return path.suffix.lower() in (".reqif", ".reqifz") or bool(
        re.search(r"<REQ-IF\b", path.read_text(encoding="utf-8", errors="ignore")[:2000])
    )
