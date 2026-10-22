"""The artifact CLI commands, end to end over real files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.fixtures.example_capability import example_capability
from understudy.artifact.hashing import content_hash, load, save
from understudy.cli import app

runner = CliRunner()


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "draft.json"
    save(example_capability(), path)
    return path


def test_lint_passes_the_example(artifact: Path) -> None:
    result = runner.invoke(app, ["lint", str(artifact)])

    assert result.exit_code == 0
    assert "clean" in result.output


def test_lint_exits_nonzero_on_an_error(artifact: Path) -> None:
    broken = json.loads(artifact.read_text())
    del broken["success"]["negative_control"]
    artifact.write_text(json.dumps(broken))

    result = runner.invoke(app, ["lint", str(artifact)])

    assert result.exit_code == 1
    assert "L001" in result.output


def test_review_renders_a_summary(artifact: Path) -> None:
    result = runner.invoke(app, ["review", str(artifact)])

    assert result.exit_code == 0
    assert "CONTRACT" in result.output
    assert "STEPS" in result.output


def test_describe_as_tool_emits_a_tool_definition(artifact: Path) -> None:
    result = runner.invoke(app, ["describe", str(artifact), "--as", "tool"])

    assert result.exit_code == 0
    assert json.loads(result.output)["name"] == "parabank_open_account"


def test_describe_rejects_an_unknown_format(artifact: Path) -> None:
    result = runner.invoke(app, ["describe", str(artifact), "--as", "yaml"])

    assert result.exit_code == 2


def test_approve_binds_the_approval_to_the_content_hash(artifact: Path, tmp_path: Path) -> None:
    out = tmp_path / "approved.json"

    result = runner.invoke(
        app, ["approve", str(artifact), "--reviewer", "A Reviewer", "--out", str(out)]
    )

    assert result.exit_code == 0
    approved = load(out)
    assert approved.lifecycle.status == "approved"
    assert approved.lifecycle.reviewed_by == "A Reviewer"
    assert approved.lifecycle.approved_hash == content_hash(approved)


def test_approve_out_may_be_a_directory(artifact: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["approve", str(artifact), "--reviewer", "A Reviewer", "--out", str(tmp_path)]
    )

    assert result.exit_code == 0
    approved_files = sorted(tmp_path.glob("*@*.json"))
    assert len(approved_files) == 1
    assert load(approved_files[0]).lifecycle.status == "approved"


def test_approve_refuses_an_artifact_with_lint_errors(artifact: Path, tmp_path: Path) -> None:
    broken = json.loads(artifact.read_text())
    broken["steps"][3]["retry"] = {"max_attempts": 3, "backoff_ms": [1, 2, 3]}
    artifact.write_text(json.dumps(broken))
    out = tmp_path / "approved.json"

    result = runner.invoke(
        app, ["approve", str(artifact), "--reviewer", "A Reviewer", "--out", str(out)]
    )

    assert result.exit_code == 1
    assert "L003" in result.output
    assert not out.exists()


def test_schema_command_emits_json_schema() -> None:
    result = runner.invoke(app, ["schema"])

    assert result.exit_code == 0
    assert json.loads(result.output)["title"] == "Capability"


def test_a_missing_artifact_is_reported_not_traced(tmp_path: Path) -> None:
    result = runner.invoke(app, ["lint", str(tmp_path / "nope.json")])

    assert result.exit_code == 2
    assert "no such artifact" in result.output


def test_an_invalid_artifact_is_reported_not_traced(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"id": "x.y"}')

    result = runner.invoke(app, ["lint", str(path)])

    assert result.exit_code == 2
    assert "not a valid capability" in result.output
