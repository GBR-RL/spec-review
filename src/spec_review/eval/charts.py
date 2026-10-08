"""README charts from the committed result files, in a light and a dark variant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spec_review.config import ROOT

RESULTS = ROOT / "docs" / "results"
ASSETS = ROOT / "docs" / "assets"

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "bar": "#cfcec9", "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "muted": "#c3c2b7", "grid": "#33322f",
             "bar": "#4a4945", "s1": "#3987e5", "s2": "#d95926"},
}  # fmt: skip


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _style(ax: Any, t: dict[str, str]) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: Any, name: str, theme: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"{name}-{theme}.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return path


CLASSIFIERS = (
    ("tfidf_svm", "TF-IDF + linear SVM"),
    ("lora_qwen3-0.6b", "LoRA Qwen3-0.6B"),
    ("fewshot_qwen3.5-4b", "Few-shot Qwen3.5-4B"),
)


def classification(theme: str) -> Path:
    """Grouped bars per task, plus the random-fold score of TF-IDF as a marked reference."""
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0), sharey=True, facecolor=t["surface"])
    panels = (("all-12", "12 classes"), ("fr-nfr", "functional vs non-functional"))
    for ax, (task, title) in zip(axes, panels, strict=True):
        _style(ax, t)
        names, vals, los, his = [], [], [], []
        for key, label in CLASSIFIERS:
            r = _json(RESULTS / "classification" / f"{task}_{key}.json")
            names.append(label)
            vals.append(r["macro_f1"])
            los.append(r["macro_f1"] - r["macro_f1_ci"][0])
            his.append(r["macro_f1_ci"][1] - r["macro_f1"])
        ys = list(range(len(names)))
        ax.barh(ys, vals, height=0.55, color=t["s1"])
        ax.errorbar(vals, ys, xerr=[los, his], fmt="none", ecolor=t["muted"], elinewidth=1.2,
                    capsize=3)  # fmt: skip
        for y, v in zip(ys, vals, strict=True):
            ax.text(0.02, y, f"{v:.2f}", va="center", fontsize=9, color="#ffffff",
                    fontweight="bold" if v == max(vals) else "normal")  # fmt: skip
        leak = _json(RESULTS / "classification" / f"{task}_tfidf_svm_random-folds.json")
        ax.axvline(leak["macro_f1"], color=t["s2"], linewidth=2, linestyle=(0, (4, 3)),
                   label=f"TF-IDF + SVM with random folds ({leak['macro_f1']:.2f})")  # fmt: skip
        leg = ax.legend(frameon=False, fontsize=8, loc="upper center",
                        bbox_to_anchor=(0.5, -0.12))  # fmt: skip
        for text in leg.get_texts():
            text.set_color(t["ink"])
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left", color=t["ink"], fontsize=10)
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(CLASSIFIERS)), [n for _, n in CLASSIFIERS])
    fig.suptitle("Requirement classification on projects the model never saw (macro-F1)",
                 x=0.01, ha="left", color=t["ink"], fontsize=11, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    path = _save(fig, "classification", theme)
    plt.close(fig)
    return path


TRACE_METHODS = (("tfidf", "TF-IDF"), ("bge-m3", "bge-m3"), ("hybrid", "hybrid"))
PAIR_LABELS = {
    "CM1:HLR-LLR": "CM1 requirements",
    "eTOUR:UC-CODE": "eTOUR use case → code",
    "iTrust:UC-CODE": "iTrust use case → code",
    "EasyClinic:UC-ID": "EasyClinic use case → diagram",
    "EasyClinic:UC-TC": "EasyClinic use case → test",
    "EasyClinic:UC-CODE": "EasyClinic use case → code",
    "EasyClinic:ID-CODE": "EasyClinic diagram → code",
    "EasyClinic:TC-CODE": "EasyClinic test → code",
}


def traceability(theme: str) -> Path:
    """Dot plot: one row per pair, TF-IDF and bge-m3 as dots, hybrid as a bar."""
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    runs = {k: _json(RESULTS / "traceability" / f"{k}.json") for k, _ in TRACE_METHODS}
    pairs = list(PAIR_LABELS)
    fig, ax = plt.subplots(figsize=(7.6, 3.9), facecolor=t["surface"])
    _style(ax, t)
    ys = range(len(pairs))
    hybrid = [runs["hybrid"]["pairs"][p]["map"] for p in pairs]
    ax.barh(list(ys), hybrid, height=0.55, color=t["bar"], label="hybrid")
    for key, label, color in (("tfidf", "TF-IDF", t["s1"]), ("bge-m3", "bge-m3", t["s2"])):
        vals = [runs[key]["pairs"][p]["map"] for p in pairs]
        ax.scatter(vals, list(ys), s=60, color=color, label=label, zorder=3,
                   edgecolor=t["surface"], linewidth=1.5)  # fmt: skip
    for y, v in zip(ys, hybrid, strict=True):
        ax.text(1.03, y, f"{v:.2f}", va="center", fontsize=8.5, color=t["ink"])
    ax.set_yticks(list(ys), [PAIR_LABELS[p] for p in pairs])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.text(1.03, -0.85, "hybrid", fontsize=8, color=t["muted"])
    ax.set_xlabel("mean average precision", color=t["muted"], fontsize=9)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3,
                    bbox_to_anchor=(0.45, -0.13))  # fmt: skip
    for text in leg.get_texts():
        text.set_color(t["ink"])
    ax.set_title("Recovering trace links (bar: hybrid of TF-IDF and bge-m3)", loc="left",
                 color=t["ink"], fontsize=11, fontweight="bold")  # fmt: skip
    path = _save(fig, "traceability", theme)
    plt.close(fig)
    return path


def all_charts() -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    return [fn(theme) for fn in (classification, traceability) for theme in THEMES]
