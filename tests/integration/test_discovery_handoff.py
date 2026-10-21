"""Discovery with people in the loop, offline: a scripted model asks for help, an
operator takes the live session over the console, a scripted person chooses what
the model could not, and the model carries on. The person's step is compiled into
the draft, flagged for review, and the draft replays.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page

from tests.conftest import REPO
from tests.fixtures.fake_model import (
    Script,
    ScriptedModel,
    browser,
    comboboxes,
    numeric_link,
    open_account,
    ref_of,
    ref_target,
    tool,
)
from tests.fixtures.site import BASE_URL, PASSWORD, USERNAME, FixtureSite
from tests.integration.test_replay_handoff import Desk, Person
from understudy.artifact import hashing
from understudy.artifact.schema import ParamValue
from understudy.control.console import Console
from understudy.control.handoff import Control
from understudy.control.plane import ControlPlane
from understudy.discovery.goal import load_goal
from understudy.discovery.run import DiscoveryResult, discover
from understudy.policy.gate import PolicyGate
from understudy.profile import load_profile
from understudy.redaction.redactor import Redactor
from understudy.replay.engine import Replayer
from understudy.replay.preflight import grant
from understudy.session import Credentials
from understudy.surface.web import CONTEXT_OPTIONS

GOAL = load_goal(REPO / "goals" / "open_account.yaml")
PROFILE = load_profile(REPO / "apps" / "parabank" / "profile.yaml")
CREATE = "/parabank/services_proxy/bank/createAccount"
DEMO = {"account_type": "SAVINGS", "funding_account_id": "13344"}


def asks_for_the_account_type(told: list[str]) -> Callable[[], Script]:
    """Fills the funding account, then asks a person to choose the account type."""

    def script() -> Script:
        text = yield browser("navigate", url=f"{BASE_URL}/openaccount.htm")
        text = yield browser("read_page")
        while len(comboboxes(text)) < 2 or not comboboxes(text)[1][1]:
            text = yield browser("wait", duration=0.3)
            text = yield browser("read_page")
        funding = comboboxes(text)[1][0]
        text = yield browser("form_input", target=ref_target(funding), value="13344")
        text = yield tool(
            "request_human",
            reason="I cannot tell which account type the task means.",
            need="Choose the account type.",
        )
        told.append(text)
        text = yield browser("read_page")
        text = yield browser(
            "left_click", target=ref_target(ref_of(text, 'button "Open New Account"'))
        )
        if "did not grant permission" in text:
            yield tool("give_up", reason="A person did not approve opening the account.")
            return
        text = yield browser("read_page")
        heading = ref_of(text, 'heading "Account Opened!"')
        link = numeric_link(text, "Account Opened!")
        text = yield tool("record_output", name="new_account_id", ref=link, description="number")
        yield tool("declare_success", summary="Opened it.", evidence_refs=[heading, link])

    return script


def choose_savings(page: Page) -> None:
    page.focus("select#type")
    page.keyboard.press("ArrowDown")  # CHECKING -> SAVINGS, a change a person makes


class Rig:
    def __init__(self, page: Page, gate: PolicyGate, tmp_path: Path) -> None:
        self.page = page
        self.gate = gate
        self.tmp = tmp_path
        self.person = Person()
        self.result: DiscoveryResult | None = None

    def run(
        self, model: ScriptedModel, script: Callable[[Desk, Person], None], *, escalate: bool = True
    ) -> DiscoveryResult:
        plane = ControlPlane()
        control = Control(
            plane=plane,
            escalate=escalate,
            relay=self.person.relay,
            poll_ms=50,
            max_wait_s=15,
            live_every_s=0.5,
        )
        errors: list[BaseException] = []
        with Console(plane, self.tmp / "runs", port=0) as console:
            desk = Desk(console)

            def operate() -> None:
                try:
                    script(desk, self.person)
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=operate, daemon=True)
            thread.start()
            self.result = discover(
                GOAL,
                GOAL.values(DEMO),
                gate=self.gate,
                profile=PROFILE,
                credentials=Credentials(USERNAME, PASSWORD),
                model=model,
                open_page=lambda: self.page,
                runs_dir=self.tmp / "runs",
                draft_path=self.tmp / "drafts" / "parabank.open_account.json",
                redactor=Redactor(b"k" * 32),
                control=control,
            )
            thread.join(timeout=20)
        if errors:
            raise errors[0]
        return self.result

    @property
    def run_dir(self) -> Path:
        assert self.result is not None
        return self.tmp / "runs" / self.result.run_id


@pytest.fixture
def rig(page: Page, site: FixtureSite, gate: PolicyGate, tmp_path: Path) -> Rig:
    return Rig(page, gate, tmp_path)


@pytest.fixture
def fresh_page(browser: Browser, site: FixtureSite) -> Iterator[Callable[[], Page]]:
    contexts = []

    def make() -> Page:
        context = browser.new_context(**CONTEXT_OPTIONS)
        site.install(context)
        contexts.append(context)
        return context.new_page()

    yield make
    for context in contexts:
        context.close()


def test_a_person_does_the_step_the_model_asked_for_and_it_compiles_and_replays(
    rig: Rig, site: FixtureSite, gate: PolicyGate, fresh_page: Callable[[], Page], tmp_path: Path
) -> None:
    told: list[str] = []

    def script(desk: Desk, person: Person) -> None:
        take = desk.next_open("takeover")
        assert take["reason"] == "human_requested"
        assert take["subject"] == {
            "kind": "goal",
            "id": "parabank.open_account",
            "version": "1.0.0",
        }
        desk.ok(take, "claim")
        person.does(choose_savings)
        desk.ok(take, "resume", note="Savings, as the customer asked.")
        approval = desk.next_open("approval")
        assert "button 'Open New Account'" in approval["detail"]
        assert site.reached(CREATE) == 0
        desk.ok(approval, "approve")

    result = rig.run(ScriptedModel(asks_for_the_account_type(told)), script)

    assert result.status == "compiled", result
    assert result.lint_ok, result.lint
    assert [(h.kind, h.resolution) for h in result.handoffs] == [
        ("takeover", "resumed"),
        ("approval", "approved"),
    ]
    # The model heard what the person did, in the turn that answered its request.
    (account,) = told
    assert "A person has been asked." in account
    assert "A person took over the browser and has handed it back." in account
    assert "to the value of account_type" in account
    assert "Savings, as the customer asked." in account

    draft = hashing.load(Path(result.draft or ""))
    (human_step,) = draft.provenance.human_steps
    step = next(s for s in draft.steps if s.id == human_step)
    assert step.action == "select"
    assert step.value == ParamValue(param="account_type")
    assert [s.action for s in draft.steps] == ["navigate", "select", "select", "click", "extract"]
    review = (rig.run_dir / "review.txt").read_text()
    assert "recorded from a person's action" in review
    assert f"by a person {human_step}" in review

    trajectory = json.loads((rig.run_dir / "trajectory.json").read_text())
    assert [c["member"] for c in trajectory["calls"] if c["actor"] == "human"] == ["form_input"]

    # The step a person recorded replays like any other, with other inputs.
    inputs = {"account_type": "CHECKING", "funding_account_id": "12456"}
    replayed = Replayer(
        draft,
        gate=gate,
        profile=PROFILE,
        credentials=Credentials(USERNAME, PASSWORD),
        runs_dir=tmp_path / "replays",
        redactor=Redactor(b"k" * 32),
        screenshots=False,
    ).run(
        inputs,
        open_page=fresh_page,
        approval=grant(draft, inputs, gate.tenant.id, "test"),
        allow_draft=True,
    )
    assert replayed.status == "succeeded", replayed
    assert site.reached(CREATE) == 2


def test_a_commit_the_model_attempts_waits_for_the_console(rig: Rig, site: FixtureSite) -> None:
    def script(desk: Desk, person: Person) -> None:
        desk.ok(desk.next_open("approval"), "approve")

    result = rig.run(
        ScriptedModel(open_account(BASE_URL, "SAVINGS", "13344")), script, escalate=False
    )

    assert result.status == "compiled", result
    assert [(h.kind, h.resolution, h.operator) for h in result.handoffs] == [
        ("approval", "approved", "dana")
    ]
    assert site.reached(CREATE) == 1


def test_a_denied_commit_is_never_sent(rig: Rig, site: FixtureSite) -> None:
    def script(desk: Desk, person: Person) -> None:
        desk.ok(desk.next_open("approval"), "deny")

    result = rig.run(
        ScriptedModel(open_account(BASE_URL, "SAVINGS", "13344")), script, escalate=False
    )

    assert result.status == "gave_up", result
    assert site.reached(CREATE) == 0


def test_an_aborted_takeover_ends_the_run_as_stuck(rig: Rig, site: FixtureSite) -> None:
    told: list[str] = []

    def script(desk: Desk, person: Person) -> None:
        desk.ok(desk.next_open("takeover"), "abort", note="Not today.")

    result = rig.run(ScriptedModel(asks_for_the_account_type(told)), script)

    assert result.status == "stuck"
    assert result.code == "human_requested"
    assert result.reason is not None and "aborted by dana" in result.reason
    assert told == []
    assert site.reached(CREATE) == 0


def test_without_escalate_a_request_for_help_ends_the_run(rig: Rig) -> None:
    told: list[str] = []
    result = rig.run(
        ScriptedModel(asks_for_the_account_type(told)), lambda d, p: None, escalate=False
    )

    assert (result.status, result.code) == ("stuck", "human_requested")
    assert result.handoffs == []


def test_a_persons_commit_is_not_sent_not_compiled_and_the_model_is_told(
    rig: Rig, site: FixtureSite
) -> None:
    told: list[str] = []

    def script_model() -> Script:
        text = yield browser("navigate", url=f"{BASE_URL}/openaccount.htm")
        text = yield browser("read_page")
        text = yield tool("request_human", reason="unsure", need="look at the form")
        told.append(text)
        yield tool("give_up", reason="The person tried to submit; I stop here.")

    def script(desk: Desk, person: Person) -> None:
        take = desk.next_open("takeover")
        desk.ok(take, "claim")
        person.does(lambda page: page.click("input[value='Open New Account']"))
        desk.ok(take, "resume")

    result = rig.run(ScriptedModel(script_model), script)

    assert result.status == "gave_up"
    assert site.reached(CREATE) == 0  # stopped at the network: no approved window
    (account,) = told
    assert "(a commit, which was not sent)" in account
    trajectory = json.loads((rig.run_dir / "trajectory.json").read_text())
    (call,) = [c for c in trajectory["calls"] if c["actor"] == "human"]
    assert (call["member"], call["effect"], call["ok"]) == ("left_click", "commit", False)
