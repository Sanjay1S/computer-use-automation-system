"""The control plane on its own: the lease and its fencing token, the state machine,
who may decide what, and what is recorded when. No browser, no server."""

from __future__ import annotations

import itertools
import threading
from typing import Any

import pytest

from understudy.control.interventions import Intervention, Subject
from understudy.control.plane import ControlLost, ControlPlane, Lease, Refused, Transition

SUBJECT = Subject(kind="capability", id="parabank.open_account", version="1.0.0")


class Clock:
    def __init__(self) -> None:
        self.ticks = itertools.count(1)
        self.now = 0.0

    def __call__(self) -> float:
        self.now = float(next(self.ticks))
        return self.now


class Queue:
    def __init__(self) -> None:
        self.published: list[Intervention] = []

    def publish(self, intervention: Intervention) -> None:
        self.published.append(intervention)


def started(**kwargs: Any) -> tuple[ControlPlane, Lease]:
    plane = ControlPlane(**kwargs)
    return plane, plane.start("replay-test-run", automation="engine")


def open_takeover(plane: ControlPlane, lease: Lease, **kwargs: Any) -> Intervention:
    return plane.open(
        lease,
        kind="takeover",
        subject=SUBJECT,
        reason="unknown_blocking_state",
        detail="a modal covers the target",
        step_id="s3_funding_account",
        resume_points=["s3_funding_account", "s1_open_form"],
        **kwargs,
    )


# --- the lease and fencing -------------------------------------------------------------


def test_automation_starts_holding_the_lease() -> None:
    plane, lease = started()

    assert plane.state == "AGENT"
    assert lease == Lease("engine", 1)
    plane.check(lease)


def test_no_lease_or_a_made_up_one_cannot_act() -> None:
    plane, lease = started()

    with pytest.raises(ControlLost):
        plane.check(None)
    with pytest.raises(ControlLost):
        plane.check(Lease("engine", lease.epoch + 1))
    with pytest.raises(ControlLost):
        plane.check(Lease("human", lease.epoch))


def test_opening_an_intervention_takes_the_lease_from_automation() -> None:
    plane, lease = started()
    open_takeover(plane, lease)

    assert plane.state == "AWAITING_HUMAN"
    assert plane.lease.holder is None
    with pytest.raises(ControlLost, match="not the current one"):
        plane.check(lease)


def test_a_stale_lease_never_acts_again_even_after_control_comes_back() -> None:
    plane, old = started()
    intervention = open_takeover(plane, old)
    plane.act(intervention.id, "claim", operator="dana")
    plane.act(intervention.id, "resume", operator="dana")
    new = plane.close(intervention.id, "resumed", to="AGENT")

    assert new.holder == "engine" and new.epoch > old.epoch
    plane.check(new)
    with pytest.raises(ControlLost):
        plane.check(old)


def test_every_change_of_hands_raises_the_epoch() -> None:
    plane, lease = started()
    intervention = open_takeover(plane, lease)
    epochs = [lease.epoch, plane.lease.epoch]
    plane.act(intervention.id, "claim", operator="dana")
    epochs.append(plane.lease.epoch)
    plane.act(intervention.id, "resume", operator="dana")
    epochs.append(plane.lease.epoch)  # a decision alone does not change hands
    plane.close(intervention.id, "resumed", to="AGENT")
    epochs.append(plane.lease.epoch)

    assert epochs == [1, 2, 3, 3, 4]


def test_only_the_lease_holder_can_open_an_intervention() -> None:
    plane, lease = started()

    with pytest.raises(ControlLost):
        open_takeover(plane, Lease("engine", 99))
    open_takeover(plane, lease)
    with pytest.raises(ControlLost):
        open_takeover(plane, lease)  # automation no longer holds it


# --- the state machine ---------------------------------------------------------------------


def test_a_takeover_goes_claim_then_resume_then_back_to_automation() -> None:
    plane, lease = started()
    intervention = open_takeover(plane, lease)

    claimed = plane.act(intervention.id, "claim", operator="dana")
    assert plane.state == "HUMAN"
    assert plane.lease == Lease("human", 3, "dana")
    assert claimed.status == "claimed" and claimed.claimed_by == "dana"

    decided = plane.act(intervention.id, "resume", operator="dana", note="closed the notice")
    assert decided.status == "decided"
    assert plane.state == "HUMAN"  # automation changes state when it acts on the decision
    assert plane.decision(intervention.id) is not None

    plane.close(intervention.id, "resumed", to="AGENT")
    closed = plane.get(intervention.id)
    assert plane.state == "AGENT"
    assert (closed.status, closed.resolution) == ("closed", "resumed")
    assert closed.decision is not None and closed.decision.note == "closed the notice"


def test_complete_and_abort_end_in_their_own_terminal_states() -> None:
    plane, lease = started()
    intervention = open_takeover(plane, lease)
    plane.act(intervention.id, "claim", operator="dana")
    plane.act(intervention.id, "complete", operator="dana")
    assert plane.close(intervention.id, "completed", to="COMPLETED_BY_HUMAN").holder is None
    assert plane.state == "COMPLETED_BY_HUMAN"
    with pytest.raises(RuntimeError):
        open_takeover(plane, plane.lease)

    plane, lease = started()
    intervention = open_takeover(plane, lease)
    plane.act(intervention.id, "abort", operator="dana")  # without claiming the session
    assert plane.close(intervention.id, "aborted", to="ABORTED").holder is None
    assert plane.state == "ABORTED"


@pytest.mark.parametrize("action", ["resume", "complete"])
def test_resume_and_complete_need_the_session_claimed_first(action: str) -> None:
    plane, lease = started()
    intervention = open_takeover(plane, lease)

    with pytest.raises(Refused, match="claim the session"):
        plane.act(intervention.id, action, operator="dana")  # type: ignore[arg-type]


def test_only_the_operator_holding_the_session_decides() -> None:
    plane, lease = started()
    intervention = open_takeover(plane, lease)
    plane.act(intervention.id, "claim", operator="dana")

    for action in ("resume", "complete", "abort"):
        with pytest.raises(Refused, match="dana holds this session"):
            plane.act(intervention.id, action, operator="lee")  # type: ignore[arg-type]
    with pytest.raises(Refused):
        plane.act(intervention.id, "claim", operator="lee")


def test_a_decision_is_made_once() -> None:
    plane, lease = started()
    intervention = open_takeover(plane, lease)
    plane.act(intervention.id, "abort", operator="dana")

    with pytest.raises(Refused, match="already decided"):
        plane.act(intervention.id, "claim", operator="dana")
    plane.close(intervention.id, "aborted", to="ABORTED")
    with pytest.raises(Refused, match="already closed"):
        plane.act(intervention.id, "abort", operator="dana")


def test_an_approval_is_approved_or_denied_and_never_claimed() -> None:
    plane, lease = started()
    intervention = plane.open(
        lease, kind="approval", subject=SUBJECT, reason="commit_approval", detail="open account"
    )

    assert intervention.allowed == ["approve", "deny"]
    for action in ("claim", "resume", "complete", "abort"):
        with pytest.raises(Refused, match="not possible for a approval"):
            plane.act(intervention.id, action, operator="dana")  # type: ignore[arg-type]
    plane.act(intervention.id, "approve", operator="dana")
    assert plane.state == "AWAITING_HUMAN"
    plane.close(intervention.id, "approved", to="AGENT")
    assert plane.state == "AGENT"


def test_a_commit_with_an_unknown_outcome_is_never_resumed() -> None:
    plane, lease = started()
    intervention = plane.open(
        lease,
        kind="verify_commit",
        subject=SUBJECT,
        reason="commit_outcome_unknown",
        detail="no answer after the commit",
        resume_points=["s4_submit"],
    )

    assert intervention.allowed == ["claim", "complete", "abort"]
    assert intervention.resume_points == []  # never offered, whatever was asked
    plane.act(intervention.id, "claim", operator="dana")
    with pytest.raises(Refused):
        plane.act(intervention.id, "resume", operator="dana")
    plane.act(intervention.id, "complete", operator="dana")


def test_resume_from_must_be_one_of_the_offered_points() -> None:
    plane, lease = started()
    intervention = open_takeover(plane, lease)
    plane.act(intervention.id, "claim", operator="dana")

    with pytest.raises(Refused, match="resume from one of: s3_funding_account, s1_open_form"):
        plane.act(intervention.id, "resume", operator="dana", resume_from="s4_submit")
    decided = plane.act(intervention.id, "resume", operator="dana", resume_from="s1_open_form")
    assert decided.decision is not None and decided.decision.resume_from == "s1_open_form"


@pytest.mark.parametrize(
    ("operator", "note", "problem"),
    [
        ("", None, "operator name"),
        ("   ", None, "operator name"),
        ("x" * 81, None, "operator"),
        ("dana", "n" * 2_001, "note"),
    ],
)
def test_operator_input_is_bounded(operator: str, note: str | None, problem: str) -> None:
    plane, lease = started()
    intervention = open_takeover(plane, lease)

    with pytest.raises(ValueError, match=problem):
        plane.act(intervention.id, "claim", operator=operator, note=note)
    assert plane.state == "AWAITING_HUMAN"


def test_an_unknown_intervention_is_a_key_error() -> None:
    plane, _ = started()

    with pytest.raises(KeyError):
        plane.act("iv-nosuchone", "claim", operator="dana")


def test_a_plane_serves_one_run() -> None:
    plane, _ = started()

    with pytest.raises(RuntimeError):
        plane.start("replay-other-run", automation="engine")


# --- time, records, threads --------------------------------------------------------------


def test_the_state_at_a_past_moment_is_known() -> None:
    """Input reported late is judged by who held the session when it happened."""
    clock = Clock()
    plane = ControlPlane(clock=clock)
    lease = plane.start("replay-test-run", automation="engine")  # t=1
    intervention = open_takeover(plane, lease)  # t=2
    plane.act(intervention.id, "claim", operator="dana")  # t=3
    plane.act(intervention.id, "resume", operator="dana")
    plane.close(intervention.id, "resumed", to="AGENT")  # t=4

    assert [plane.state_at(t) for t in (0.5, 1.5, 2.5, 3.5, 4.5)] == [
        "AGENT",
        "AGENT",
        "AWAITING_HUMAN",
        "HUMAN",
        "AGENT",
    ]


def test_every_change_is_reported_and_new_interventions_are_published() -> None:
    changes: list[tuple[str, str, Transition | None]] = []
    queue = Queue()
    plane = ControlPlane(queue=queue)
    lease = plane.start(
        "replay-test-run",
        automation="engine",
        on_change=lambda change, iv, t: changes.append((change, iv.status, t)),
    )
    intervention = open_takeover(plane, lease)
    plane.act(intervention.id, "claim", operator="dana")
    plane.act(intervention.id, "resume", operator="dana")
    plane.update_live(intervention.id, "interventions/x.live.png")
    plane.close(intervention.id, "resumed", to="AGENT")

    assert [(c, s) for c, s, _ in changes] == [
        ("opened", "open"),
        ("claimed", "claimed"),
        ("decided", "decided"),
        ("updated", "decided"),
        ("closed", "closed"),
    ]
    states = [t.state for _, _, t in changes if t is not None]
    assert states == ["AWAITING_HUMAN", "HUMAN", "AGENT"]
    assert [i.id for i in queue.published] == [intervention.id]


def test_the_snapshot_shows_the_lease_and_the_newest_intervention_first() -> None:
    plane, lease = started()
    first = open_takeover(plane, lease)
    plane.act(first.id, "abort", operator="dana")
    lease = plane.close(first.id, "aborted", to="AGENT")
    second = open_takeover(plane, lease)

    snapshot = plane.snapshot()
    assert snapshot["state"] == "AWAITING_HUMAN"
    assert snapshot["lease"] == {"holder": None, "epoch": lease.epoch + 1, "operator": None}
    assert [i["id"] for i in snapshot["interventions"]] == [second.id, first.id]


def test_only_one_of_two_operators_racing_to_claim_wins() -> None:
    plane, lease = started()
    intervention = open_takeover(plane, lease)
    results: list[str] = []
    gate = threading.Barrier(8)

    def claim(name: str) -> None:
        gate.wait()
        try:
            plane.act(intervention.id, "claim", operator=name)
            results.append(name)
        except Refused:
            results.append("refused")

    threads = [threading.Thread(target=claim, args=(f"op{n}",)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    winners = [r for r in results if r != "refused"]
    assert len(winners) == 1
    assert plane.lease.operator == winners[0]


def test_intervention_ids_survive_redaction() -> None:
    """Letters only: the digit detectors would otherwise mask them in the run folder."""
    from understudy.redaction.redactor import Redactor

    redactor = Redactor(b"k" * 32)
    plane, lease = started()
    ids = []
    for _ in range(50):
        intervention = open_takeover(plane, lease)
        plane.act(intervention.id, "abort", operator="dana")
        lease = plane.close(intervention.id, "aborted", to="AGENT")
        ids.append(intervention.id)
    assert all(redactor.text(i) == i for i in ids)
