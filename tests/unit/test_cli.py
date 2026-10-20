from typer.testing import CliRunner

from understudy import __version__
from understudy.cli import app


def test_version_command_prints_installed_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.output.strip() == __version__
