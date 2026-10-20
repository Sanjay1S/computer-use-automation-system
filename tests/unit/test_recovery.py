from __future__ import annotations

import pytest

from tests.fixtures.example_capability import example_capability
from understudy.artifact.schema import RecoveryPolicy, RetryPolicy
from understudy.replay.recovery import RecoveryLedger, backoff_ms, reentry_index
from understudy.replay.result import RecoveryEvent


def event(handler: str = "retry_transient") -> RecoveryEvent:
    return RecoveryEvent(handler=handler, step_id="s1", reason="r")  # type: ignore[arg-type]


def test_the_budget_is_per_run_and_runs_out() -> None:
    ledger = RecoveryLedger(RecoveryPolicy(allowed=["retry_transient", "reauthenticate"], budget=2))
    assert ledger.refusal("retry_transient") is None
    ledger.spend(event())
    ledger.spend(event("reauthenticate"))
    assert ledger.refusal("retry_transient") == "the recovery budget for this run is spent"
    with pytest.raises(RuntimeError):
        ledger.spend(event())
    assert [e.handler for e in ledger.events] == ["retry_transient", "reauthenticate"]


def test_a_handler_the_artifact_does_not_allow_never_runs() -> None:
    ledger = RecoveryLedger(RecoveryPolicy(allowed=["retry_transient"], budget=3))
    assert ledger.refusal("reauthenticate") == "the artifact does not allow reauthenticate"


def test_backoff_uses_the_steps_table_then_the_default() -> None:
    assert [backoff_ms(RetryPolicy(max_attempts=3, backoff_ms=[250]), n) for n in range(3)] == [
        250,
        250,
        250,
    ]
    assert [backoff_ms(RetryPolicy(max_attempts=3), n) for n in range(4)] == [
        500,
        1_000,
        2_000,
        2_000,
    ]


def test_the_restart_point_is_the_latest_reentry_point_at_or_before_the_step() -> None:
    steps = example_capability().steps
    assert reentry_index(steps, 2) == 0
    assert reentry_index(steps, 0) == 0
    no_entry = [s.model_copy(update={"reentry_point": False}) for s in steps]
    assert reentry_index(no_entry, 3) is None
