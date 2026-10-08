"""Command line entry point."""

from __future__ import annotations

import typer

from spec_review import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Requirements as code: an LLM-assisted reviewer for engineering requirements."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
