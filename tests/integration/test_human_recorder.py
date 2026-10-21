"""The human recorder in a real page: whose input each report is, what it says about
the element, and what it never says."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Page

from tests.fixtures.site import BASE_URL, FixtureSite
from understudy.control.handoff import Control, Handoffs
from understudy.control.human_recorder import HumanInput, HumanRecorder
from understudy.control.interventions import Intervention, Subject
from understudy.control.plane import ControlLost, ControlPlane, Lease
from understudy.evidence.run import RunContext
from understudy.policy.gate import PolicyGate
from understudy.redaction.redactor import Redactor
from understudy.surface.web import WebSurface

HTML = """<html><body>
<h1>Form</h1>
<button id="keep" type="button">Keep</button>
<button id="gone" type="button" onclick="this.remove()">OK</button>
<label for="type">Type</label>
<select id="type" name="type">
  <option value="0">CHECKING</option><option value="1">SAVINGS</option>
</select>
<input id="pw" type="password" name="password">
<input id="note" name="note">
</body></html>"""

SECRET = "hunter2-very-secret"


class Session:
    def __init__(self, page: Page, gate: PolicyGate) -> None:
        self.page = page
        self.plane = ControlPlane()
        self.surface = WebSurface(page, gate, mode="replay", control=self.plane)
        self.lease: Lease = self.plane.start("replay-test-run", automation="engine")
        self.recorder = HumanRecorder(self.surface, self.plane)
        self.intervention: Intervention | None = None

    def hand_over(self) -> None:
        self.intervention = self.plane.open(
            self.lease,
            kind="takeover",
            subject=Subject(kind="capability", id="x"),
            reason="test",
            detail="test",
        )
        self.plane.act(self.intervention.id, "claim", operator="dana")

    def hand_back(self) -> None:
        assert self.intervention is not None
        self.plane.act(self.intervention.id, "resume", operator="dana")
        self.lease = self.plane.close(self.intervention.id, "resumed", to="AGENT")

    def inputs(self) -> list[HumanInput]:
        return self.recorder.flush()


@pytest.fixture
def session(page: Page, gate: PolicyGate) -> Iterator[Session]:
    page.set_content(HTML)  # before the recorder: it installs itself in the open document
    yield Session(page, gate)


def test_automation_s_own_input_is_its_own(session: Session) -> None:
    session.surface.act("click", session.page.locator("#keep"), lease=session.lease)

    reports = session.inputs()
    assert [r.attribution for r in reports] == ["automation"]


def test_input_while_automation_holds_the_session_is_unexpected(session: Session) -> None:
    session.page.click("#keep")  # not through the surface: nobody's action window

    (report,) = session.inputs()
    assert report.attribution == "unexpected"
    assert report.kind == "click"


def test_a_persons_click_is_theirs_and_located_for_the_synthesizer(session: Session) -> None:
    session.hand_over()
    session.page.click("#keep")

    (report,) = session.inputs()
    assert report.attribution == "human"
    assert report.member == "left_click"
    assert report.locator is not None and report.locator.count() == 1
    assert report.facts.id_attr == "keep"
    assert report.describe() == "button 'Keep'"


def test_an_element_gone_by_the_time_it_is_read_is_described_from_the_report(
    session: Session,
) -> None:
    session.hand_over()
    session.page.click("#gone")

    (report,) = session.inputs()
    assert report.attribution == "human"
    assert report.locator is None
    assert report.describe() == "button 'OK'"


def test_events_a_page_script_dispatches_are_not_a_persons(session: Session) -> None:
    session.hand_over()
    session.page.evaluate("document.getElementById('keep').click()")
    session.page.select_option("#type", "1")  # Playwright sets selects by script

    assert session.inputs() == []


def test_a_password_field_never_reports_its_value(session: Session) -> None:
    session.hand_over()
    session.page.click("#pw")
    session.page.keyboard.type(SECRET)
    session.page.keyboard.press("Tab")  # change fires on blur

    reports = session.inputs()
    changes = [r for r in reports if r.kind == "change"]
    assert len(changes) == 1
    assert changes[0].value is None
    assert all(SECRET not in repr(r) for r in reports)


def test_a_select_chosen_by_keyboard_reports_its_option(session: Session) -> None:
    session.hand_over()
    session.page.focus("#type")
    session.page.keyboard.press("ArrowDown")

    (report,) = [r for r in session.inputs() if r.kind == "change"]
    assert report.member == "form_input"
    assert (report.value, report.option_label) == ("1", "SAVINGS")
    assert report.facts.label == "Type"


def test_a_report_read_after_the_hand_back_still_belongs_to_the_person(session: Session) -> None:
    session.hand_over()
    session.page.click("#keep")
    session.hand_back()  # nothing drained yet

    (report,) = session.inputs()
    assert report.attribution == "human"


def test_a_click_into_a_field_is_part_of_the_change_that_follows(session: Session) -> None:
    session.hand_over()
    session.page.click("#note")
    session.page.keyboard.type("hello")
    session.page.click("#keep")

    reports = session.inputs()
    assert [(r.kind, r.member) for r in reports] == [
        ("click", None),
        ("change", "form_input"),
        ("click", "left_click"),
    ]
    assert reports[1].value == "hello"


def test_the_surface_refuses_an_actuation_under_a_stale_lease(session: Session) -> None:
    stale = session.lease
    session.hand_over()

    with pytest.raises(ControlLost):
        session.surface.act("click", session.page.locator("#gone"), lease=stale)
    with pytest.raises(ControlLost):
        session.surface.navigate(f"{BASE_URL}/index.htm", lease=stale)
    assert session.page.locator("#gone").count() == 1  # nothing was delivered
    session.hand_back()
    session.surface.act("click", session.page.locator("#gone"), lease=session.lease)
    assert session.page.locator("#gone").count() == 0


def test_the_recorder_follows_the_session_into_new_documents(
    page: Page, site: FixtureSite, gate: PolicyGate
) -> None:
    session = Session(page, gate)  # installed on about:blank, before any navigation
    session.surface.navigate(f"{BASE_URL}/index.htm", lease=session.lease)
    session.hand_over()
    page.click("input[value='Log In']")
    page.wait_for_load_state("load")

    reports = session.inputs()
    assert [(r.kind, r.attribution) for r in reports][:1] == [("click", "human")]


def test_while_a_person_holds_the_session_no_rule_answers_a_dialog_for_them(
    page: Page, gate: PolicyGate
) -> None:
    page.set_content(HTML)
    plane = ControlPlane()
    surface = WebSurface(page, gate, mode="replay", control=plane, dialog_rules=[(None, "accept")])
    lease = plane.start("replay-test-run", automation="engine")

    assert page.evaluate("confirm('Close this account?')") is True  # the rule, for automation
    intervention = plane.open(
        lease, kind="takeover", subject=Subject(kind="capability", id="x"), reason="r", detail="d"
    )
    plane.act(intervention.id, "claim", operator="dana")
    assert page.evaluate("confirm('Close this account?')") is False  # only the default
    assert [d.answer for d in surface.dialog_history] == ["accept", "dismiss"]


# --- the handoff loop's bookkeeping (review round 2) ------------------------------------------


@pytest.fixture
def handoffs(page: Page, gate: PolicyGate, tmp_path: Path) -> Handoffs:
    page.set_content(HTML)
    plane = ControlPlane()
    surface = WebSurface(page, gate, mode="replay", control=plane)
    run = RunContext(tmp_path, kind="replay", redactor=Redactor(b"k" * 32))
    return Handoffs(
        Control(plane=plane, escalate=True),
        run,
        surface,
        automation="engine",
        subject=Subject(kind="capability", id="x"),
    )


def test_an_intervention_left_open_is_closed_before_the_error(handoffs: Handoffs) -> None:
    with (
        pytest.raises(RuntimeError, match="was left open"),
        handoffs.opened("takeover", "r", "d") as intervention,
    ):
        pass

    assert handoffs.plane.get(intervention.id).resolution == "interrupted (left open)"
    assert handoffs.plane.state == "ABORTED"


def test_a_report_read_after_its_handoff_closed_still_reaches_handoff_json(
    handoffs: Handoffs, page: Page
) -> None:
    with handoffs.opened("takeover", "r", "d") as intervention:
        handoffs.plane.act(intervention.id, "claim", operator="dana")
        page.click("#keep")
        held = time.time()
        reports, handoffs.human._queued = handoffs.human._queued, []  # not read yet
        handoffs.plane.act(intervention.id, "resume", operator="dana")
        handoffs.close(intervention, "resumed", to="AGENT")
    assert handoffs.records[0].actions == []
    handoffs.human._queued = reports  # the page's report, read late

    handoffs.poll()

    (record,) = json.loads((handoffs.run.dir / "handoff.json").read_text())["handoffs"]
    assert [a["description"] for a in record["actions"]] == ["button 'Keep'"]
    assert handoffs.summaries()[0].human_actions == 1
    assert held > 0
