"""The discover command's offline paths: what it refuses before a browser or a model runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.conftest import REPO
from understudy import cli
from understudy.cli import app

GOAL = str(REPO / "goals" / "open_account.yaml")
CREDS = {"PARABANK_USERNAME": "john", "PARABANK_PASSWORD": "pw-for-cli-test"}


class NoModel:
    def __call__(self, request: dict[str, Any]) -> Any:
        raise AssertionError("the model should not have been called")


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No ./.env, the repo's app folders, and tenants in a folder this test owns."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "APPS", REPO / "apps")
    tenants = tmp_path / "tenants"
    tenants.mkdir()
    for path in (REPO / "tenants").glob("*.yaml"):
        (tenants / path.name).write_text(path.read_text())
    (tenants / "parabank-prod.yaml").write_text(
        "schema_version: understudy.tenant/v1\nid: parabank-prod\nvendor_app: parabank\n"
        "base_url: https://bank.example/parabank\nsandbox: false\n"
    )
    monkeypatch.setattr(cli, "TENANTS", tenants)
    monkeypatch.setattr(cli, "MODEL_FACTORY", NoModel)
    return tmp_path


def invoke(*args: str, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    result = CliRunner().invoke(app, ["discover", GOAL, "--console-port", "0", *args], env=env)
    return result.exit_code, result.stdout, result.stderr


def test_a_production_tenant_is_rejected_before_any_browser_or_model(isolated: Path) -> None:
    runs = isolated / "runs"
    code, out, _ = invoke("--tenant", "parabank-prod", "--runs-dir", str(runs), env=CREDS)

    assert code == 3
    result = json.loads(out)
    assert result["status"] == "rejected"
    assert "not a sandbox" in result["reason"]
    assert (runs / result["run_id"] / "result.json").is_file()
    assert not (runs / result["run_id"] / "trajectory.json").exists()


def test_a_bad_param_value_is_rejected_by_name(isolated: Path) -> None:
    code, out, _ = invoke(
        "--tenant", "parabank-local", "--param", "account_type=GOLD",
        "--runs-dir", str(isolated / "runs"), env=CREDS,
    )  # fmt: skip

    assert code == 3
    assert "param 'account_type' must be one of" in json.loads(out)["reason"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--param", "account_type"), "--param takes name=value"),
        (("--param", "a=1", "--param", "a=2"), "given twice"),
        (("--effort", "extreme"), "unknown --effort"),
    ],
)
def test_usage_errors_exit_2(isolated: Path, args: tuple[str, ...], message: str) -> None:
    code, _, err = invoke("--tenant", "parabank-local", *args, env=CREDS)

    assert code == 2
    assert message in err


def test_missing_credentials_stop_before_anything_runs(isolated: Path) -> None:
    runs = isolated / "runs"
    code, _, err = invoke(
        "--tenant", "parabank-local", "--runs-dir", str(runs),
        env={"PARABANK_USERNAME": "", "PARABANK_PASSWORD": ""},
    )  # fmt: skip

    assert code == 2 and "PARABANK_USERNAME" in err
    assert not runs.exists()


def test_a_missing_goal_file_is_a_usage_error(isolated: Path) -> None:
    result = CliRunner().invoke(
        app, ["discover", str(isolated / "nope.yaml"), "--tenant", "parabank-local"], env=CREDS
    )

    assert result.exit_code == 2
    assert "cannot discover" in result.stderr


def test_a_malformed_goal_is_a_usage_error_not_a_traceback(isolated: Path) -> None:
    broken = isolated / "broken.yaml"
    broken.write_text("capability: x.y\n  title: [unclosed\n")

    result = CliRunner().invoke(
        app, ["discover", str(broken), "--tenant", "parabank-local"], env=CREDS
    )

    assert result.exit_code == 2
    assert "cannot discover" in result.stderr
