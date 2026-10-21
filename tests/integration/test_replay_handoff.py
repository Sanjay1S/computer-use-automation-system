"""Replay with people in the loop, end to end and offline: the engine on the fixture
site, the real console over HTTP, an operator on a thread of its own, and a scripted
person acting in the live page.

The person's input reaches the page through the handoff loop's relay, on the thread
that drives the browser (Playwright is single-threaded), as a remote operator's
would. Their clicks are real input events, so the human recorder sees them exactly
as it would see a person's in a headed window.
"""

from __future__ import annotations

import http.client
import json
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Page

from tests.conftest import REPO
from tests.fixtures.example_capability import example_capability
from tests.fixtures.site import PASSWORD, USERNAME, FixtureSite
from understudy.chaos import MODAL_ID, Chaos, parse_fault
from understudy.control.console import Console
from understudy.control.handoff import Control
from understudy.control.interventions import Intervention
from understudy.control.plane import ControlPlane
from understudy.policy.gate import PolicyGate
from understudy.policy.models import ApprovalModes
from understudy.profile import load_profile
from understudy.redaction.redactor import Redactor
from understudy.replay.engine import Replayer
from understudy.replay.preflight import grant
from understudy.replay.result import ReplayResult
from understudy.session import Credentials

CREATE = "/parabank/services_proxy/bank/createAccount"
PROFILE = load_profile(REPO / "apps" / "parabank" / "profile.yaml")
INPUTS = {"account_type": "SAVINGS", "funding_account_id": "54321"}
MODAL_OK = f"#{MODAL_ID} button"


class Desk:
    """A simulated operator: reads the console and decides through it, over HTTP."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.operator = "dana"

    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        connection = http.client.HTTPConnection("127.0.0.1", self.console.port, timeout=5)
        headers = {"Host": f"127.0.0.1:{self.console.port}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        connection.request(method, f"/{self.console.token}{path}", body=data, headers=headers)
        response = connection.getresponse()
        payload = json.loads(response.read() or b"null")
        connection.close()
        return response.status, payload

    def next_open(
        self, kind: str, *, seen: Sequence[str] = (), timeout: float = 15
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, state = self.call("GET", "/api/state")
            for item in state["interventions"]:
                if item["status"] == "open" and item["kind"] == kind and item["id"] not in seen:
                    return dict(item)
            time.sleep(0.05)
        raise AssertionError(f"no open {kind} intervention appeared")

    def post(self, intervention: dict[str, Any], action: str, **fields: Any) -> tuple[int, Any]:
        body = {"operator": self.operator, **fields}
        return self.call("POST", f"/api/interventions/{intervention['id']}/{action}", body)

    def ok(self, intervention: dict[str, Any], action: str, **fields: Any) -> dict[str, Any]:
        status, body = self.post(intervention, action, **fields)
        assert status == 200, (action, status, body)
        return dict(body)


class Person:
    """Stands in for a person at the live window."""

    def __init__(self) -> None:
        self._todo: queue.Queue[Callable[[Page], None]] = queue.Queue()
        self._done = threading.Event()
        self.relayed = 0

    def relay(self, page: Page, intervention: Intervention) -> None:
        while not self._todo.empty():
            self._todo.get_nowait()(page)
            self.relayed += 1
            self._done.set()

    def does(self, action: Callable[[Page], None], timeout: float = 10) -> None:
        """From the operator's thread: wait until the input has reached the page."""
        self._done.clear()
        self._todo.put(action)
        assert self._done.wait(timeout), "the relay never delivered the person's input"


class Rig:
    def __init__(self, page: Page, gate: PolicyGate, site: FixtureSite, runs: Path) -> None:
        self.page = page
        self.gate = gate
        self.site = site
        self.runs = runs
        self.person = Person()
        self.last: ReplayResult | None = None

    def run(
        self,
        script: Callable[[Desk, Person], None] | None = None,
        *,
        faults: Sequence[str] = (),
        inputs: dict[str, str] | None = None,
        approve: bool = True,
        interactive: bool = False,
        escalate: bool = True,
        max_wait_s: float = 15,
        max_handoffs: int = 3,
        timeout_ms: int = 2_000,
        gate: PolicyGate | None = None,
    ) -> ReplayResult:
        capability = example_capability().model_copy(deep=True)
        for step in capability.steps:
            step.timeout_ms = min(step.timeout_ms, timeout_ms)
        values = inputs or INPUTS
        gate = gate or self.gate
        plane = ControlPlane()
        control = Control(
            plane=plane,
            escalate=escalate,
            interactive_approval=interactive,
            relay=self.person.relay,
            poll_ms=50,
            max_wait_s=max_wait_s,
            max_handoffs=max_handoffs,
            live_every_s=0.5,
        )
        replayer = Replayer(
            capability,
            gate=gate,
            profile=PROFILE,
            credentials=Credentials(USERNAME, PASSWORD),
            runs_dir=self.runs,
            redactor=Redactor(b"k" * 32),
            screenshots=False,
        )
        errors: list[BaseException] = []
        with Console(plane, self.runs, port=0) as console:
            desk = Desk(console)

            def operate() -> None:
                try:
                    if script is not None:
                        script(desk, self.person)
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=operate, daemon=True)
            thread.start()
            self.last = replayer.run(
                values,
                open_page=lambda: self.page,
                approval=grant(capability, values, gate.tenant.id, "test") if approve else None,
                allow_draft=True,
                chaos=Chaos([parse_fault(f, capability) for f in faults], gate) if faults else None,
                control=control,
            )
            thread.join(timeout=20)
        if errors:
            raise errors[0]
        return self.last

    @property
    def run_dir(self) -> Path:
        assert self.last is not None
        return self.runs / self.last.run_id

    def events(self) -> list[dict[str, Any]]:
        lines = (self.run_dir / "events.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    def handoffs(self) -> list[dict[str, Any]]:
        return list(json.loads((self.run_dir / "handoff.json").read_text())["handoffs"])


@pytest.fixture
def rig(page: Page, site: FixtureSite, gate: PolicyGate, tmp_path: Path) -> Iterator[Rig]:
    yield Rig(page, gate, site, tmp_path / "runs")


def control_events(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(e["actor"], e["data"]["change"]) for e in events if "change" in e.get("data", {})]


# --- takeover ------------------------------------------------------------------------------


def test_a_person_clears_an_unknown_modal_and_the_run_resumes_to_success(
    rig: Rig, site: FixtureSite
) -> None:
    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("takeover")
        assert (iv["reason"], iv["step_id"]) == ("unknown_blocking_state", "s3_funding_account")
        assert iv["resume_points"] == ["s3_funding_account", "s1_open_form"]
        assert iv["screenshot"] and iv["recent_events"]
        assert site.reached(CREATE) == 0
        desk.ok(iv, "claim")
        person.does(lambda page: page.click(MODAL_OK))
        desk.ok(iv, "resume", note="Closed the maintenance notice.")

    result = rig.run(script, faults=["unknown_modal_at:s3"])

    assert result.status == "succeeded", result
    assert result.outputs == {"new_account_id": "13566"}
    assert result.completed_by == "automation"
    assert result.side_effects.commit_state == "committed"
    assert site.reached(CREATE) == 1
    assert result.escalation is None
    (summary,) = result.handoffs
    assert (summary.kind, summary.resolution, summary.operator) == ("takeover", "resumed", "dana")
    assert summary.human_actions == 1

    (handoff,) = rig.handoffs()
    assert handoff["operator"] == "dana"
    assert handoff["note"] == "Closed the maintenance notice."
    assert handoff["claimed_at"] and handoff["duration_ms"] > 0
    (action,) = handoff["actions"]
    assert (action["kind"], action["description"]) == ("click", "button 'OK'")
    assert action["verified"] is False  # the modal removed itself before it could be checked

    events = rig.events()
    assert control_events(events) == [
        ("engine", "opened"),
        ("human", "claimed"),
        ("human", "decided"),
        ("engine", "closed"),
    ]
    human = [e for e in events if e["actor"] == "human" and e["kind"] == "act"]
    assert [e["member"] for e in human] == ["left_click"]
    claimed = next(e["seq"] for e in events if e.get("data", {}).get("change") == "claimed")
    closed = next(e["seq"] for e in events if e.get("data", {}).get("change") == "closed")
    assert claimed < human[0]["seq"] < closed
    engine_after = [e for e in events if e["seq"] > closed and e["actor"] == "engine"]
    assert [e["step_id"] for e in engine_after if e["kind"] == "act"][:1] == ["s3_funding_account"]
    verified = [e["data"] for e in events if "inputs_after_handoff" in e.get("data", {})]
    assert [v["inputs_after_handoff"] for v in verified] == ["verified"]
    stored = json.loads(
        (rig.run_dir / "interventions" / f"{summary.intervention_id}.json").read_text()
    )
    assert (stored["status"], stored["resolution"]) == ("closed", "resumed")
    assert (rig.run_dir / stored["screenshot"]).is_file()


def test_a_step_the_person_did_themselves_is_not_done_twice(rig: Rig, site: FixtureSite) -> None:
    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("takeover")
        desk.ok(iv, "claim")

        def clear_and_choose(page: Page) -> None:
            page.click(MODAL_OK)
            page.focus("select#fromAccountId")
            for _ in range(3):  # 12345 -> 12456 -> 13344 -> 54321
                page.keyboard.press("ArrowDown")

        person.does(clear_and_choose)
        desk.ok(iv, "resume")

    result = rig.run(script, faults=["unknown_modal_at:s3"])

    assert result.status == "succeeded", result
    events = rig.events()
    looked = [e for e in events if e.get("data", {}).get("after_resume") == "done_by_person"]
    assert [e["step_id"] for e in looked] == ["s3_funding_account"]
    engine_s3 = [
        e["ok"]
        for e in events
        if e["actor"] == "engine" and e["kind"] == "act" and e["step_id"] == "s3_funding_account"
    ]
    assert engine_s3 == [False]  # blocked by the modal, then never again: the person chose
    # Each arrow key on a closed select is a change of its own, as it is for a person.
    assert [e["member"] for e in events if e["actor"] == "human" and e["kind"] == "act"] == [
        "left_click",
        *["form_input"] * 3,
    ]
    assert site.reached(CREATE) == 1


def test_an_abort_fails_the_run_with_its_escalation_and_commits_nothing(
    rig: Rig, site: FixtureSite
) -> None:
    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("takeover")
        desk.ok(iv, "abort", note="Maintenance window; try again tomorrow.")

    result = rig.run(script, faults=["unknown_modal_at:s3"])

    assert result.status == "failed", result
    assert result.failure is None
    assert result.escalation is not None
    assert result.escalation.reason == "unknown_blocking_state"
    assert result.escalation.resolution == "aborted"
    assert result.escalation.intervention_id == result.handoffs[0].intervention_id
    assert result.side_effects.commit_state == "not_attempted"
    assert site.reached(CREATE) == 0
    (handoff,) = rig.handoffs()
    assert (handoff["resolution"], handoff["claimed_at"]) == ("aborted", None)


def test_handing_back_a_page_that_still_blocks_escalates_again(rig: Rig, site: FixtureSite) -> None:
    def script(desk: Desk, person: Person) -> None:
        first = desk.next_open("takeover")
        desk.ok(first, "claim")
        desk.ok(first, "resume")  # the modal is still there
        second = desk.next_open("takeover", seen=[first["id"]])
        assert second["reason"] == "unknown_blocking_state"
        desk.ok(second, "abort")

    result = rig.run(script, faults=["unknown_modal_at:s3"])

    assert result.status == "failed"
    assert [h.resolution for h in result.handoffs] == ["resumed", "aborted"]
    assert site.reached(CREATE) == 0


def test_resuming_from_the_reentry_point_reloads_the_form(rig: Rig, site: FixtureSite) -> None:
    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("takeover")
        desk.ok(iv, "claim")
        status, body = desk.post(iv, "resume", resume_from="s4_submit")
        assert status == 409 and "resume from one of" in body["error"]
        desk.ok(iv, "resume", resume_from="s1_open_form")

    result = rig.run(script, faults=["unknown_modal_at:s3"])

    assert result.status == "succeeded", result
    assert [t.step_id for t in result.timings.steps].count("s1_open_form") == 2
    assert result.handoffs[0].resolution == "resumed from s1_open_form"
    assert site.reached(CREATE) == 1


def test_nobody_answering_leaves_the_run_needing_a_human(rig: Rig, site: FixtureSite) -> None:
    result = rig.run(None, faults=["unknown_modal_at:s3"], max_wait_s=0.6)

    assert result.status == "needs_human"
    assert result.escalation is not None
    assert result.escalation.resolution == "timed_out"
    assert result.escalation.intervention_id is not None
    assert site.reached(CREATE) == 0


def test_a_run_that_keeps_coming_back_stops_at_the_handoff_limit(rig: Rig) -> None:
    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("takeover")
        desk.ok(iv, "claim")
        desk.ok(iv, "resume")

    result = rig.run(script, faults=["unknown_modal_at:s3"], max_handoffs=1)

    assert result.status == "needs_human"
    assert result.escalation is not None
    assert result.escalation.resolution == "handoff_limit"
    assert [h.resolution for h in result.handoffs] == ["resumed"]


def test_without_escalate_an_unknown_state_is_still_a_result(rig: Rig) -> None:
    result = rig.run(None, faults=["unknown_modal_at:s3"], escalate=False)

    assert result.status == "needs_human"
    assert result.escalation is not None and result.escalation.resolution is None
    assert result.handoffs == []


# --- a commit whose outcome is unknown ------------------------------------------------------


def test_a_late_answer_after_the_commit_is_verified_and_completed_by_a_person(
    rig: Rig, site: FixtureSite
) -> None:
    site.result_delay_ms = 2_500

    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("verify_commit")
        assert iv["reason"] == "commit_outcome_unknown"
        assert iv["resume_points"] == []
        desk.ok(iv, "claim")
        status, _ = desk.post(iv, "resume")
        assert status == 409  # a commit whose outcome is unknown is never resumed
        time.sleep(2.0)  # the app's answer reaches the page meanwhile
        desk.ok(iv, "complete", note="The page shows Account Opened!")

    result = rig.run(script, timeout_ms=1_000)

    assert result.status == "succeeded", result
    assert result.completed_by == "human"
    assert result.outputs == {"new_account_id": "13566"}
    assert result.side_effects.commit_state == "committed"
    assert site.reached(CREATE) == 1
    assert [(h.kind, h.resolution) for h in result.handoffs] == [("verify_commit", "completed")]


def test_complete_is_refused_until_the_success_checkpoint_holds(
    rig: Rig, site: FixtureSite
) -> None:
    site.result_delay_ms = 2_500

    def script(desk: Desk, person: Person) -> None:
        first = desk.next_open("verify_commit")
        desk.ok(first, "claim")
        desk.ok(first, "complete")  # too early: the page shows nothing yet
        second = desk.next_open("verify_commit", seen=[first["id"]])
        assert second["reason"] == "completion_not_verified"
        assert "success checkpoint does not hold" in second["detail"]
        desk.ok(second, "claim")
        time.sleep(2.0)
        desk.ok(second, "complete")

    result = rig.run(script, timeout_ms=1_000)

    assert result.status == "succeeded", result
    assert [h.resolution for h in result.handoffs] == ["completion_not_verified", "completed"]
    assert site.reached(CREATE) == 1


def test_complete_before_any_commit_is_refused(rig: Rig, site: FixtureSite) -> None:
    def script(desk: Desk, person: Person) -> None:
        first = desk.next_open("takeover")
        desk.ok(first, "claim")
        desk.ok(first, "complete")
        second = desk.next_open("takeover", seen=[first["id"]])
        assert "has not sent its commit" in second["detail"]
        desk.ok(second, "abort")

    result = rig.run(script, faults=["unknown_modal_at:s3"])

    assert result.status == "failed"
    assert result.completed_by == "automation"
    assert site.reached(CREATE) == 0


def test_an_aborted_verification_leaves_the_commit_unknown(rig: Rig, site: FixtureSite) -> None:
    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("verify_commit")
        desk.ok(iv, "abort", note="No new account in the list; not committed.")

    result = rig.run(script, faults=["drop_commit_response"])

    assert result.status == "failed"
    assert result.escalation is not None
    assert (result.escalation.reason, result.escalation.resolution) == (
        "commit_outcome_unknown",
        "aborted",
    )
    assert result.side_effects.commit_state == "unknown"
    assert result.side_effects.commit_requests == 1


# --- interactive approval -------------------------------------------------------------------


def test_an_operator_approves_the_commit_when_the_run_reaches_it(
    rig: Rig, site: FixtureSite
) -> None:
    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("approval")
        assert iv["step_id"] == "s4_submit"
        assert "Inputs: account_type=SAVINGS, funding_account_id=••••1" in iv["detail"]
        assert site.reached(CREATE) == 0  # nothing is sent before the approval
        desk.ok(iv, "approve")

    result = rig.run(script, approve=False, interactive=True, escalate=False)

    assert result.status == "succeeded", result
    assert site.reached(CREATE) == 1
    assert [(h.kind, h.resolution, h.operator) for h in result.handoffs] == [
        ("approval", "approved", "dana")
    ]


def test_a_denied_commit_is_rejected_before_anything_is_sent(rig: Rig, site: FixtureSite) -> None:
    def script(desk: Desk, person: Person) -> None:
        desk.ok(desk.next_open("approval"), "deny", note="Wrong customer.")

    result = rig.run(script, approve=False, interactive=True, escalate=False)

    assert result.status == "rejected"
    assert result.rejection is not None and result.rejection.code == "approval_denied"
    assert result.side_effects.commit_state == "not_attempted"
    assert site.reached(CREATE) == 0


def test_a_tenant_that_wants_interactive_approval_refuses_a_token(
    rig: Rig, gate: PolicyGate, site: FixtureSite
) -> None:
    policy = gate.policy.model_copy(update={"approval": ApprovalModes(replay="interactive")})
    strict = PolicyGate(policy, gate.tenant)

    result = rig.run(None, approve=True, escalate=False, gate=strict)

    assert result.status == "rejected"
    assert result.rejection is not None and result.rejection.code == "approval_required"
    assert "accepts only interactive approval" in result.rejection.problems[0]
    assert site.requests == []


# --- evidence ------------------------------------------------------------------------------


def test_nothing_sensitive_reaches_the_run_folder_through_a_handoff(
    rig: Rig, site: FixtureSite
) -> None:
    def script(desk: Desk, person: Person) -> None:
        take = desk.next_open("takeover")
        desk.ok(take, "claim")
        person.does(lambda page: page.click(MODAL_OK))
        desk.ok(take, "resume", note=f"account 54321 for {USERNAME}")
        desk.ok(desk.next_open("approval"), "approve")

    result = rig.run(script, approve=False, interactive=True, faults=["unknown_modal_at:s3"])

    assert result.status == "succeeded", result
    files = [p for p in rig.run_dir.rglob("*") if p.suffix in (".json", ".jsonl", ".txt")]
    text = "\n".join(p.read_text() for p in files)
    assert (rig.run_dir / "handoff.json").is_file()
    for secret in (PASSWORD, site.session_id, "54321", "13566"):
        assert secret not in text, secret


# --- review round: regressions -------------------------------------------------------------


def test_an_input_the_person_changed_is_entered_again_before_the_commit(
    rig: Rig, site: FixtureSite
) -> None:
    """The commit and its approval are for the inputs given, not the person's edits."""

    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("takeover")
        desk.ok(iv, "claim")

        def clear_and_change_type(page: Page) -> None:
            page.click(MODAL_OK)
            page.focus("select#type")
            page.keyboard.press("ArrowUp")  # SAVINGS -> CHECKING

        person.does(clear_and_change_type)
        desk.ok(iv, "resume")

    result = rig.run(script, faults=["unknown_modal_at:s3"])

    assert result.status == "succeeded", result
    (check,) = [e["data"] for e in rig.events() if "inputs_after_handoff" in e.get("data", {})]
    assert check["inputs_after_handoff"] == "re_entered"
    assert check["unverified"] == ["s2_account_type"]
    assert check["restarted_from"] == "s1_open_form"
    assert [t.step_id for t in result.timings.steps].count("s2_account_type") == 2
    assert site.reached(CREATE) == 1


def test_a_page_the_person_took_off_policy_is_handed_back_again_not_failed(
    rig: Rig, site: FixtureSite
) -> None:
    def script(desk: Desk, person: Person) -> None:
        first = desk.next_open("takeover")
        desk.ok(first, "claim")

        def wander(page: Page) -> None:
            page.click(MODAL_OK)
            page.evaluate("location.href = 'admin.htm'")  # a denied route

        person.does(wander)
        time.sleep(0.3)
        desk.ok(first, "resume")
        second = desk.next_open("takeover", seen=[first["id"]])
        assert second["reason"] == "resume_state_mismatch"
        assert "off policy" in second["detail"]
        desk.ok(second, "claim")
        desk.ok(second, "resume", resume_from="s1_open_form")

    result = rig.run(script, faults=["unknown_modal_at:s3"])

    assert result.status == "succeeded", result
    assert [h.resolution for h in result.handoffs] == ["resumed", "resumed from s1_open_form"]
    assert site.reached("/parabank/admin.htm") == 0
    assert site.reached(CREATE) == 1


def test_a_page_that_dies_during_a_handoff_still_closes_its_intervention(
    rig: Rig, site: FixtureSite
) -> None:
    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("takeover")
        desk.ok(iv, "claim")
        person.does(lambda page: page.close())

    result = rig.run(script, faults=["unknown_modal_at:s3"])

    assert result.status == "failed"
    assert result.failure is not None and result.failure.category == "infra"
    (summary,) = result.handoffs
    assert summary.resolution == "interrupted (InfraError)"
    (handoff,) = rig.handoffs()
    assert handoff["resolution"] == "interrupted (InfraError)"
    assert site.reached(CREATE) == 0


# --- review round 2: regressions -----------------------------------------------------------


def test_input_while_an_approval_is_open_sends_the_inputs_through_again(
    rig: Rig, site: FixtureSite
) -> None:
    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("approval")

        def change_type(page: Page) -> None:  # someone at the window, nobody's session
            page.focus("select#type")
            page.keyboard.press("ArrowUp")  # SAVINGS -> CHECKING

        person.does(change_type)
        desk.ok(iv, "approve")

    result = rig.run(script, approve=False, interactive=True, escalate=False)

    assert result.status == "succeeded", result
    events = rig.events()
    assert [e["data"]["warning"] for e in events if "warning" in e.get("data", {})] == [
        "unexpected_input"
    ]
    (check,) = [e["data"] for e in events if "inputs_after_handoff" in e.get("data", {})]
    assert (check["inputs_after_handoff"], check["unverified"]) == (
        "re_entered",
        ["s2_account_type"],
    )
    assert [h.kind for h in result.handoffs] == ["approval"]  # asked once
    assert site.reached(CREATE) == 1


def test_a_console_approval_is_asked_again_after_a_person_held_the_session(
    rig: Rig, site: FixtureSite
) -> None:
    def script(desk: Desk, person: Person) -> None:
        first = desk.next_open("approval")
        desk.ok(first, "approve")
        take = desk.next_open("takeover")
        assert take["step_id"] == "s4_submit"
        desk.ok(take, "claim")
        person.does(lambda page: page.click(MODAL_OK))
        desk.ok(take, "resume")
        assert site.reached(CREATE) == 0
        desk.ok(desk.next_open("approval", seen=[first["id"]]), "approve")

    result = rig.run(script, approve=False, interactive=True, faults=["unknown_modal_at:s4"])

    assert result.status == "succeeded", result
    assert [(h.kind, h.resolution) for h in result.handoffs] == [
        ("approval", "approved"),
        ("takeover", "resumed"),
        ("approval", "approved"),
    ]
    assert any(e.get("data", {}).get("approval") == "withdrawn_after_handoff" for e in rig.events())
    assert site.reached(CREATE) == 1


def test_the_page_is_looked_at_again_after_a_long_wait_for_approval(
    rig: Rig, site: FixtureSite
) -> None:
    def script(desk: Desk, person: Person) -> None:
        iv = desk.next_open("approval")

        def expire(page: Page) -> None:  # the session runs out while nobody answers
            page.context.clear_cookies()
            page.reload()

        person.does(expire)
        desk.ok(iv, "approve")

    result = rig.run(script, approve=False, interactive=True, escalate=False)

    assert result.status == "succeeded", result
    assert [r.handler for r in result.recoveries] == ["reauthenticate"]
    assert site.reached(CREATE) == 1


def test_a_commit_step_handed_back_off_policy_is_handed_back_again(
    rig: Rig, site: FixtureSite
) -> None:
    def script(desk: Desk, person: Person) -> None:
        first = desk.next_open("takeover")
        desk.ok(first, "claim")

        def wander(page: Page) -> None:
            page.click(MODAL_OK)
            page.evaluate("location.href = 'admin.htm'")

        person.does(wander)
        time.sleep(0.3)
        desk.ok(first, "resume")
        second = desk.next_open("takeover", seen=[first["id"]])
        assert (second["reason"], second["step_id"]) == ("resume_state_mismatch", "s4_submit")
        desk.ok(second, "claim")
        desk.ok(second, "resume", resume_from="s1_open_form")

    result = rig.run(script, faults=["unknown_modal_at:s4"])

    assert result.status == "succeeded", result
    assert site.reached(CREATE) == 1
