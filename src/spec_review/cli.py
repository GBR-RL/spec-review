"""Command line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

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
