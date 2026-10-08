"""Scores for the quality checks: pronoun ambiguity on ReqEval, rewrites re-checked by the rules."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from spec_review.config import ROOT
from spec_review.data import promise, reqeval
from spec_review.eval.runs import run_dir
from spec_review.quality import rules

RESULTS = ROOT / "docs" / "results" / "quality"
_NP = re.compile(r"\b(?:the|a|an|each|every|all|any|its|their)\s+(?:[a-z-]+\s+){0,2}[a-z-]+",
                 re.IGNORECASE)  # fmt: skip


def candidate_heuristic(marked: str) -> bool:
    """Ambiguous if at least two noun phrases precede the pronoun (a common simple baseline)."""
    before = marked.split("<referential>", 1)[0]
    return len(_NP.findall(before)) >= 2


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def _ci(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        stats.append(f1_score(y[i], p[i], zero_division=0))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y, p)),
        "f1_ambiguous": float(f1_score(y, p, zero_division=0)),
        "f1_ambiguous_ci": _ci(y, p),
        "cohen_kappa": float(cohen_kappa_score(y, p)),
        "predicted_ambiguous": float(p.mean()),
    }


def _write(name: str, report: dict[str, Any]) -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(_round(report), indent=2) + "\n")
    return report


def _round(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_round(x) for x in v]
    return v


def pronoun_report(run: str) -> dict[str, Any]:
    gold = reqeval.load().set_index("id")
    df = pd.read_json(run_dir(run) / "merged.jsonl", lines=True, dtype={"id": str})
    ok = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()].set_index("id")
    g = gold.loc[ok.index]
    y = g.ambiguous.to_numpy()
    llm = ok.ambiguous.astype(bool).to_numpy()
    clear = ~y
    agreed = [
        bool(_norm(a)) and (_norm(a) in _norm(p) or _norm(p) in _norm(a))
        for a, p in zip(g.antecedent[clear], ok.antecedent[clear], strict=True)
    ]
    return _write(
        run,
        {
            "run": run,
            "sentences": len(ok),
            "errors": len(df) - len(ok),
            "base_rate_ambiguous": float(y.mean()),
            "llm": _scores(y, llm),
            "always_ambiguous": _scores(y, np.ones_like(y)),
            "two_noun_phrases_heuristic": _scores(
                y, np.array([candidate_heuristic(m) for m in g.marked])
            ),
            # On the sentences people agreed are clear, does the model pick their antecedent?
            "antecedent_match_on_clear": float(np.mean(agreed)) if agreed else None,
        },
    )


def review_report(run: str) -> dict[str, Any]:
    df = pd.read_json(run_dir(run) / "merged.jsonl", lines=True, dtype={"id": str})
    ok = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()]
    texts = promise.load().set_index("id").text
    before, after, new_rules, unchanged, placeholders = [], [], 0, 0, 0
    for r in ok.to_dict("records"):
        b = {f.rule for f in rules.check(str(texts[r["id"]]))}
        a = {f.rule for f in rules.check(str(r["rewrite"]))}
        before.append(len(b))
        after.append(len(a))
        new_rules += bool(a - b)
        unchanged += _norm(str(r["rewrite"])) == _norm(str(texts[r["id"]]))
        placeholders += "[value]" in str(r["rewrite"])
    b_arr, a_arr = np.array(before), np.array(after)
    flagged = b_arr > 0
    issue_types = pd.Series([i["type"] for issues in ok.issues for i in issues]).value_counts()
    return _write(
        run,
        {
            "run": run,
            "requirements": len(ok),
            "errors": len(df) - len(ok),
            "with_rule_findings_before": float(flagged.mean()),
            "with_rule_findings_after": float((a_arr > 0).mean()),
            "rule_findings_before": int(b_arr.sum()),
            "rule_findings_after": int(a_arr.sum()),
            "flagged_now_clean": float((a_arr[flagged] == 0).mean()) if flagged.any() else None,
            "rewrites_adding_a_new_rule": float(new_rules / max(1, len(ok))),
            "returned_unchanged": float(unchanged / max(1, len(ok))),
            "rewrites_with_value_placeholder": float(placeholders / max(1, len(ok))),
            "llm_issue_types": {str(k): int(v) for k, v in issue_types.items()},
            "llm_issues_per_requirement": float(ok.issues.map(len).mean()),
        },
    )
