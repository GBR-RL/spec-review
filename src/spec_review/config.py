"""Filesystem layout. Everything generated lives under one data root."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SPEC_REVIEW_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SPEC_REVIEW_DATA", ROOT / "data"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RUNS = Path(os.environ.get("SPEC_REVIEW_RUNS", ROOT / "runs"))

SEED = 13
