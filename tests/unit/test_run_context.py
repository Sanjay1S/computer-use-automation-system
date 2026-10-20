"""The run directory: nothing reaches disk unredacted."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.example_capability import example_capability
from understudy.evidence.run import RunContext
from understudy.redaction.redactor import Redactor


@pytest.fixture
def run(tmp_path: Path) -> RunContext:
    return RunContext(tmp_path, kind="replay", redactor=Redactor(b"k"), meta={"tenant": "fixture"})


def lines(run: RunContext) -> list[dict[str, object]]:
    text = (run.dir / "events.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines()]


def test_run_json_is_written(run: RunContext) -> None:
    meta = json.loads((run.dir / "run.json").read_text())
    assert meta["kind"] == "replay" and meta["tenant"] == "fixture"
    assert meta["run_id"] == run.run_id and run.run_id.startswith("replay-")


def test_run_ids_do_not_collide(tmp_path: Path) -> None:
    RunContext(tmp_path, kind="replay", redactor=Redactor(b"k"), run_id="same")
    with pytest.raises(FileExistsError):
        RunContext(tmp_path, kind="replay", redactor=Redactor(b"k"), run_id="same")


def test_an_unclassified_value_is_hashed_not_printed(run: RunContext) -> None:
    run.event("act", actor="engine", value="Jane Q Public")
    run.event("act", actor="engine", value="SAVINGS", value_sensitivity="public")
    run.event("act", actor="engine", value="13344", value_sensitivity="financial_id")
    run.event("act", actor="engine", value="hunter2", value_sensitivity="secret")
    values = [e["value"] for e in lines(run)]
    assert str(values[0]).startswith("[PII:")
    assert values[1:] == ["SAVINGS", "••••4", "[SECRET]"]


def test_urls_free_text_and_data_are_redacted(run: RunContext) -> None:
    run.know({"13344": "financial_id"})
    target = example_capability().steps[3].target
    run.event(
        "act",
        actor="engine",
        step_id="s3",
        target=target,
        url="http://localhost:8089/parabank/openaccount.htm;jsessionid=C66C519E79FF448D",
        locator_used="text='13344'",
        error="no option 13344 near 123-45-6789",
        rationale="picked 13344",
        attempts=[{"detail": "row 13344 not found"}],
    )
    (event,) = lines(run)
    dumped = json.dumps(event)
    assert "13344" not in dumped and "123-45-6789" not in dumped
    assert "jsessionid" not in dumped
    assert event["url"] == "http://localhost:8089/parabank/openaccount.htm"
    assert event["target"]["fingerprint"] == {"role": "button", "name": "Open New Account"}


def test_events_are_sequenced_and_recent_is_bounded(tmp_path: Path) -> None:
    run = RunContext(tmp_path, kind="replay", redactor=Redactor(b"k"), recent=3)
    for i in range(5):
        run.event("observe", actor="engine", step_id=f"s{i}")
    assert [e["seq"] for e in lines(run)] == [1, 2, 3, 4, 5]
    assert [e.step_id for e in run.recent(10)] == ["s2", "s3", "s4"]
    assert [e.step_id for e in run.recent(2)] == ["s3", "s4"]


def test_write_json_redacts(run: RunContext) -> None:
    path = run.write_json("result.json", {"outputs": {"ssn": "123-45-6789"}, "status": "ok"})
    assert "123-45-6789" not in path.read_text()
    assert json.loads(path.read_text())["status"] == "ok"


def test_screenshot_paths_are_numbered_and_safe(run: RunContext) -> None:
    first = run.screenshot_path("s1_open form")
    second = run.screenshot_path("../../escape")
    assert first.name == "001_s1_open_form.png"
    assert second.name == "002_______escape.png"
    assert second.parent == run.dir / "screenshots"
