"""The replay command's offline paths: what it refuses before a browser starts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import REPO
from understudy import cli
from understudy.cli import app, load_env_file

DRAFT = str(REPO / "capabilities" / "drafts" / "parabank.open_account@1.0.0.json")
CREDS = {"PARABANK_USERNAME": "john", "PARABANK_PASSWORD": "pw-for-cli-test"}


@pytest.fixture(autouse=True)
def away_from_this_machines_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command reads ./.env; run it where there is none, with the repo's folders."""
    monkeypatch.chdir(tmp_path)
    for name in ("APPS", "TENANTS", "CAPABILITIES"):
        monkeypatch.setattr(cli, name, REPO / getattr(cli, name))


def invoke(*args: str, env: dict[str, str] | None = None) -> tuple[int, str, str]:
    result = CliRunner().invoke(
        app, ["replay", DRAFT, "--tenant", "parabank-local", *args], env=env
    )
    return result.exit_code, result.stdout, result.stderr


def test_bad_input_is_rejected_with_its_own_exit_code(tmp_path: Path) -> None:
    code, out, _ = invoke(
        "--input", "account_type=GOLD", "--input", "funding_account_id=12345",
        "--allow-draft", "--approve", "--runs-dir", str(tmp_path), env=CREDS,
    )  # fmt: skip
    assert code == 3
    result = json.loads(out)
    assert result["status"] == "rejected"
    assert result["rejection"]["code"] == "invalid_input"
    assert (tmp_path / result["run_id"] / "result.json").is_file()


def test_without_approve_a_commit_is_rejected(tmp_path: Path) -> None:
    code, out, _ = invoke(
        "--input", "account_type=SAVINGS", "--input", "funding_account_id=12345",
        "--allow-draft", "--runs-dir", str(tmp_path), env=CREDS,
    )  # fmt: skip
    assert code == 3
    assert json.loads(out)["rejection"]["code"] == "approval_required"


def test_missing_credentials_stop_before_anything_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    code, _, err = invoke(
        "--input", "account_type=SAVINGS", "--runs-dir", str(runs),
        env={"PARABANK_USERNAME": "", "PARABANK_PASSWORD": ""},
    )  # fmt: skip
    assert code == 2
    assert "PARABANK_USERNAME" in err
    assert not runs.exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--inject", "teleport"), "unknown fault"),
        (("--input", "account_type"), "name=value"),
        (("--input", "a=1", "--input", "a=2"), "given twice"),
        (("--on-stuck", "panic"), "unknown --on-stuck"),
        (("--approval", "maybe"), "unknown --approval"),
        (("--approval", "interactive", "--approve"), "exclude each other"),
    ],
)
def test_usage_errors_exit_2(tmp_path: Path, args: tuple[str, ...], message: str) -> None:
    code, _, err = invoke(*args, "--runs-dir", str(tmp_path), env=CREDS)
    assert code == 2
    assert message in err


def test_a_malformed_input_is_not_echoed(tmp_path: Path) -> None:
    code, _, err = invoke("--input", "54321", "--runs-dir", str(tmp_path), env=CREDS)
    assert code == 2
    assert "54321" not in err


def test_the_env_file_fills_only_unset_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nUS_A=from-file\nexport US_B='quoted'\nUS_C=\nUS_D=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("US_D", "from-env")
    for name in ("US_A", "US_B", "US_C"):
        monkeypatch.setenv(name, "")  # empty counts as unset; monkeypatch restores it
    load_env_file(env_file)
    assert os.environ["US_A"] == "from-file"
    assert os.environ["US_B"] == "quoted"
    assert os.environ["US_C"] == ""
    assert os.environ["US_D"] == "from-env"


def test_with_an_operator_the_console_address_is_printed_and_a_rejection_still_returns(
    tmp_path: Path,
) -> None:
    """A run rejected in preflight never opens a browser, with or without an operator."""
    code, out, err = invoke(
        "--input", "account_type=GOLD", "--input", "funding_account_id=12345", "--allow-draft",
        "--on-stuck", "escalate", "--console-port", "0", "--runs-dir", str(tmp_path),
        env=CREDS,
    )  # fmt: skip
    assert code == 3
    assert "operator console: http://127.0.0.1:" in err
    assert json.loads(out)["rejection"]["code"] == "invalid_input"
