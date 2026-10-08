from typer.testing import CliRunner

from spec_review import __version__
from spec_review.cli import app


def test_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
