"""Pinned, checksummed downloads of the public datasets."""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from spec_review.config import RAW

_PROMISE = (
    "https://raw.githubusercontent.com/AleksandarMitrevski/se-requirements-classification/"
    "1f5dc4501a1956f21011083594d925bac49f198c/0-datasets/PROMISE_exp/"
)
_REQEVAL = (
    "https://raw.githubusercontent.com/frieden84/nlp4re-reqeval/"
    "05f42f83cca53c1d46e433e54b21e7f852cef7a5/data/"
)
_COEST = "http://sarec.nd.edu/coest/datasets/"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str
    unzip: bool = False


SOURCES = (
    # PROMISE_exp (Lima et al. 2019): 969 requirements from 48 projects, CC BY-SA 3.0.
    Source(
        "PROMISE_exp.arff",
        _PROMISE + "PROMISE_exp.arff",
        "7475c2904648912ef08bd1b6149f505f7ce6ab26ee9b187c2ddee28d1af97d75",
    ),
    # NLP4RE 2020 ReqEval shared task: anaphoric ambiguity, CC BY 4.0.
    Source(
        "reqeval_training.tsv",
        _REQEVAL + "training.tsv",
        "3b2f85149caf64daebf2eca68f4e1f931f1c83d64d8a27696b9501c6673fb06f",
    ),
    Source(
        "reqeval_test.tsv",
        _REQEVAL + "test.tsv",
        "9a567a0b06e7eb6d4353819a4e56c06f2d412bc8f68da2a75d782604cfcac508",
    ),
    # CoEST traceability datasets with ground-truth trace links.
    Source(
        "CM1-NASA.zip",
        _COEST + "CM1-NASA.zip",
        "b7838320a869a674d9a28c47d6d01c2478d3557c1dc4d7c562136b1b9a2827ba",
        unzip=True,
    ),
    Source(
        "EasyClinic.zip",
        _COEST + "EasyClinic.zip",
        "c7661abbe5ca69d5155d1cdc67edf54b99e6f645e6bce4bf9582425675e81715",
        unzip=True,
    ),
    Source(
        "eTOUR.zip",
        _COEST + "eTOUR.zip",
        "7a8c974deb36c20171795eba475080e8232097e9586ec26242855480b1da8079",
        unzip=True,
    ),
    Source(
        "iTrust.zip",
        _COEST + "iTrust.zip",
        "ff052eea03c561b144ff89a66dd3c34e4294c42fefaea88b84abfd18a6c8603b",
        unzip=True,
    ),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(source: Source, raw: Path = RAW) -> Path:
    """Download one file unless a copy with the right checksum is already there."""
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / source.name
    if not (path.exists() and sha256(path) == source.sha256):
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(source.url, tmp)
        got = sha256(tmp)
        if got != source.sha256:
            tmp.unlink()
            raise RuntimeError(f"{source.name}: checksum {got} != {source.sha256}")
        tmp.replace(path)
    if source.unzip:
        target = raw / "coest" / path.stem
        if not target.exists():
            with zipfile.ZipFile(path) as z:
                z.extractall(target)
    return path


def fetch_all(raw: Path = RAW) -> dict[str, Path]:
    return {s.name: fetch(s, raw) for s in SOURCES}
