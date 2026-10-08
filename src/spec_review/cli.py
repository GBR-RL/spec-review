"""Command line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from spec_review import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Requirements as code: an LLM-assisted reviewer for engineering requirements."""


@app.command()
def data() -> None:
    """Download the pinned datasets and prepare PROMISE_exp, ReqEval and CoEST."""
    from spec_review.data import coest, download, promise, reqeval

    typer.echo(f"downloaded {len(download.fetch_all())} files")
    p = promise.build()
    split = p.groupby("split").agg(requirements=("id", "size"), projects=("project", "nunique"))
    typer.echo(f"PROMISE_exp: {len(p)} requirements\n{split}")
    r = reqeval.build()
    typer.echo(f"ReqEval: {len(r)} sentences, {int(r.ambiguous.sum())} ambiguous")
    arts, links = coest.build()
    summary = arts.groupby(["dataset", "kind"]).size().to_string()
    typer.echo(f"CoEST: {len(arts)} artifacts, {len(links)} links\n{summary}")


@app.command()
def lint(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Requirements file.")],
    fmt: str = typer.Option("text", "--format", help="text, json or github"),
    fail_on_findings: bool = typer.Option(False, help="Exit with 1 if anything is found."),
) -> None:
    """Check every requirement in a file against the quality rules."""
    from spec_review.quality.lint import lint as run
    from spec_review.quality.lint import render

    reqs = run(path)
    out = render(reqs, path, fmt)
    if out:
        typer.echo(out)
    n = sum(len(r.findings) for r in reqs)
    if fmt != "json":
        flagged = sum(bool(r.findings) for r in reqs)
        typer.echo(f"{n} findings in {flagged} of {len(reqs)} requirements", err=True)
    if fail_on_findings and n:
        raise typer.Exit(1)


@app.command()
def classify(
    task: str = typer.Option("all", help="fr-nfr, nfr-11, all-12 or all"),
    method: str = typer.Option("all", help="tfidf_lr, tfidf_svm or all"),
    random_folds: bool = typer.Option(False, help="Plain stratified folds (leakage check)."),
) -> None:
    """Cross-validate classifiers over projects on PROMISE_exp."""
    from spec_review.classify import baselines, cv

    tasks = list(cv.TASKS) if task == "all" else [task]
    methods = list(baselines.METHODS) if method == "all" else [method]
    for t in tasks:
        for m in methods:
            r = cv.evaluate(t, m, baselines.METHODS[m], grouped=not random_folds)
            lo, hi = r["macro_f1_ci"]
            typer.echo(
                f"{t:7s} {r['method']:24s} macro-F1 {r['macro_f1']:.3f} [{lo:.3f}, {hi:.3f}]"
            )


@app.command("trace-eval")
def trace_eval(
    method: str = typer.Option("all", help="tfidf, e5-small, bge-m3, hybrid or all"),
) -> None:
    """Recover trace links on the CoEST pairs and score the rankings (MAP, recall@k)."""
    from spec_review.trace import recover as r

    dense = {m: r.dense(m) for m in r.DENSE_MODELS}
    methods = {"tfidf": r.tfidf, **dense, "hybrid": r.hybrid(r.tfidf, dense["bge-m3"])}
    for name, fn in methods.items():
        if method in ("all", name):
            out = r.evaluate(name, fn)
            typer.echo(f"{name:10s} mean MAP {out['mean_map']:.3f}")


@app.command("trace-verify-score")
def trace_verify_score(
    run: str = typer.Option("trace-verify-qwen3.5-4b"),
    method: str = typer.Option("hybrid"),
) -> None:
    """Re-rank the recovered candidates by the LLM verdicts and score them again."""
    import json

    import pandas as pd

    from spec_review.config import RUNS
    from spec_review.eval.runs import run_dir
    from spec_review.trace import recover, verify

    verdicts = pd.read_json(run_dir(run) / "merged.jsonl", lines=True, dtype={"id": str})
    ok = verdicts[verdicts.get("error", pd.Series(index=verdicts.index, dtype=object)).isna()]
    rankings = json.loads((RUNS / "trace" / f"{method}_rankings.json").read_text())
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for key, score in zip(ok.id, ok.score, strict=True):
        pair, source, target = key.split("|")
        scores.setdefault(pair, {}).setdefault(source, {})[target] = float(score)
    out: dict[str, Any] = {"method": f"{method}+llm-verify", "run": run, "pairs": {}}
    yes_right, yes_total = 0, 0
    for pair in recover.PAIRS:
        _, _, gold = recover.pair_data(pair)
        reranked = {
            s: verify.rerank(r, scores.get(pair.name, {}).get(s, {}))
            for s, r in rankings.get(pair.name, {}).items()
        }
        out["pairs"][pair.name] = recover.score_rankings(reranked, gold)
        for s, ts in scores.get(pair.name, {}).items():
            for t, v in ts.items():
                if v > 0:
                    yes_total += 1
                    yes_right += (s, t) in gold
    out["mean_map"] = sum(p["map"] for p in out["pairs"].values()) / len(out["pairs"])
    out["precision_of_yes"] = yes_right / max(1, yes_total)
    out["verdicts"] = len(ok)
    out["errors"] = len(verdicts) - len(ok)
    path = recover.RESULTS / f"{method}_llm-verify.json"
    path.write_text(json.dumps(recover._round(out), indent=2) + "\n")
    typer.echo(f"mean MAP {out['mean_map']:.3f}, precision of yes {out['precision_of_yes']:.3f}")


@app.command()
def impact(
    method: str = typer.Option("hybrid", help="Recovery method whose rankings build the graph."),
    k: int = typer.Option(3, help="Top candidates per source kept as predicted links."),
    hops: int = typer.Option(3),
) -> None:
    """Impact analysis on EasyClinic: predicted impact sets against the true ones."""
    import json

    from spec_review.data import coest
    from spec_review.trace import graph as g
    from spec_review.trace.recover import RESULTS

    arts, _ = coest.load()
    starts = arts[(arts.dataset == "EasyClinic") & (arts.kind == "use case")].id.tolist()
    gold = g.Graph(g.gold_edges("EasyClinic"))
    by_k: dict[str, dict[str, Any]] = {}
    for top in sorted({1, 2, 3, 5, k}):
        pred = g.Graph(g.predicted_edges(method, "EasyClinic", top))
        s = by_k[str(top)] = g.impact_scores(gold, pred, starts, hops)
        sizes = f"true {s['mean_true_impact']:.1f}, predicted {s['mean_predicted_impact']:.1f}"
        typer.echo(f"top-{top}: precision {s['precision']:.3f} recall {s['recall']:.3f} ({sizes})")
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"method": method, "hops": hops, "by_k": by_k}
    (RESULTS / f"impact_{method}.json").write_text(json.dumps(report, indent=2) + "\n")


@app.command("lora-train")
def lora_train(
    fold: int = typer.Option(..., help="Fold to hold out (0-4)."),
    epochs: int = typer.Option(3),
) -> None:
    """Fine-tune Qwen3-0.6B with LoRA on four folds and predict the fifth."""
    from spec_review.classify import lora

    typer.echo(f"wrote {lora.train_fold(fold, epochs=epochs)}")


@app.command("classify-score")
def classify_score(
    method: str = typer.Option(..., help="Name for the results, e.g. lora_qwen3-0.6b"),
    lora_dir: Annotated[
        Path | None, typer.Option(help="Folder with the five LoRA fold files.")
    ] = None,
    llm_run: str = typer.Option("", help="Name of a merged few-shot LLM run."),
) -> None:
    """Score out-of-fold predictions made on CI (12 classes, and functional vs NFR)."""
    import pandas as pd

    from spec_review.classify import cv, lora
    from spec_review.eval.runs import run_dir

    if lora_dir is not None:
        preds = lora.merge(lora_dir)
    else:
        preds = pd.read_json(run_dir(llm_run) / "merged.jsonl", lines=True, dtype={"id": str})
        preds = preds.assign(pred=preds["pred"].fillna("F"))
    for task in ("all-12", "fr-nfr"):
        r = cv.score_predictions(task, method, preds)
        lo, hi = r["macro_f1_ci"]
        typer.echo(f"{task:7s} {method}: macro-F1 {r['macro_f1']:.3f} [{lo:.3f}, {hi:.3f}]")


@app.command("llm-run")
def llm_run(
    task: str = typer.Option(
        ..., help="pronoun, pronoun-candidates, review, classify or trace-verify"
    ),
    model: str = typer.Option("qwen3.5-4b"),
    name: str = typer.Option("", help="Run name (default: task-model)."),
    shard: int = typer.Option(0),
    shards: int = typer.Option(1),
) -> None:
    """Run an LLM task over one shard of its items (JSON lines, resumable)."""
    from spec_review.eval import runs
    from spec_review.llm.client import LLM

    llm = LLM(model)
    run = name or f"{task}-{model}"
    if task in ("pronoun", "pronoun-candidates"):
        path = runs.pronoun_run(llm, run, shard, shards, candidates=task != "pronoun")
    elif task == "review":
        path = runs.review_run(llm, run, shard, shards)
    elif task == "classify":
        path = runs.classify_run(llm, run, shard, shards)
    elif task == "trace-verify":
        path = runs.verify_run(llm, run, shard, shards)
    else:
        raise typer.BadParameter(f"unknown task {task!r}")
    typer.echo(f"wrote {path}")


@app.command("llm-merge")
def llm_merge(name: str = typer.Option(...)) -> None:
    """Join the shard files of an LLM run."""
    from spec_review.eval import runs

    df = runs.merge(name)
    errors = int(df["error"].notna().sum()) if "error" in df else 0
    typer.echo(f"{name}: {len(df)} items, {errors} errors")


@app.command("quality-report")
def quality_report(
    pronoun_run: str = typer.Option("pronoun-qwen3.5-4b"),
    review_run: str = typer.Option("review-qwen3.5-4b"),
) -> None:
    """Score the pronoun-ambiguity and review runs found on disk."""
    from spec_review.eval import quality
    from spec_review.eval.runs import run_dir

    for name, fn in ((pronoun_run, quality.pronoun_report), (review_run, quality.review_report)):
        if (run_dir(name) / "merged.jsonl").exists():
            typer.echo(fn(name))


@app.command("model-url")
def model_url(name: str) -> None:
    """Download URL of a GGUF model."""
    from spec_review.llm.models import MODELS

    typer.echo(MODELS[name].url)


@app.command("data-card")
def data_card() -> None:
    """Regenerate docs/data.md from the prepared datasets."""
    from spec_review.data import card

    typer.echo(f"wrote {card.write()}")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
