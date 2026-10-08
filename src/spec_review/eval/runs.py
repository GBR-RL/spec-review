"""Sharded, resumable LLM runs: one JSON line per item, so shards can run on parallel runners."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from spec_review.config import RUNS
from spec_review.data import promise, reqeval
from spec_review.llm.client import LLM, Completion
from spec_review.quality import reviewer, rules

OUT = RUNS / "llm"


def run_dir(name: str) -> Path:
    return OUT / name


def shard(ids: list[str], index: int, count: int) -> set[str]:
    return {i for i in ids if int(hashlib.md5(i.encode()).hexdigest(), 16) % count == index}


def _usage(c: Completion) -> dict[str, Any]:
    return {
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens,
        "latency_s": round(c.latency_s, 2),
    }


def loop(
    items: Iterable[tuple[str, Any]],
    fn: Callable[[Any], dict[str, Any]],
    path: Path,
    index: int = 0,
    count: int = 1,
) -> Path:
    items = list(items)
    wanted = shard([i for i, _ in items], index, count)
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item_id, item in items:
            if item_id not in wanted or item_id in done:
                continue
            t0 = time.perf_counter()
            try:
                row = {"id": item_id, **fn(item)}
            except Exception as err:  # counted in the report
                row = {"id": item_id, "error": f"{type(err).__name__}: {err}"[:500]}
            row["wall_s"] = round(time.perf_counter() - t0, 2)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    return path


def merge(name: str) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for p in sorted(run_dir(name).glob("shard*.jsonl"))
        for line in p.read_text(encoding="utf-8").splitlines()
    ]
    df = pd.DataFrame(rows).drop_duplicates("id", keep="last")
    df.to_json(run_dir(name) / "merged.jsonl", orient="records", lines=True, force_ascii=False)
    return df


def pronoun_run(llm: LLM, name: str, index: int = 0, count: int = 1) -> Path:
    """Every ReqEval sentence: nothing is trained on them, so all 212 are scored."""
    df = reqeval.load()

    def fn(r: dict[str, Any]) -> dict[str, Any]:
        out, c = reviewer.pronoun(llm, r["marked"])
        return {**out, **_usage(c)}

    path = run_dir(name) / f"shard{index}of{count}.jsonl"
    return loop(((r["id"], r) for r in df.to_dict("records")), fn, path, index, count)


def review_run(llm: LLM, name: str, index: int = 0, count: int = 1, split: str = "test") -> Path:
    """Review and rewrite the PROMISE requirements of one split, with the rule findings as hints."""
    df = promise.load()
    df = df[df.split == split]

    def fn(r: dict[str, Any]) -> dict[str, Any]:
        found = rules.check(r["text"])
        out, c = reviewer.review(llm, r["text"], found)
        return {"rule_findings": [f.rule for f in found], **out, **_usage(c)}

    path = run_dir(name) / f"shard{index}of{count}.jsonl"
    return loop(((r["id"], r) for r in df.to_dict("records")), fn, path, index, count)
