"""The discovery driver against the fixture site: one browser toolset call at a time,
through the SDK's own `tool_result`, so the SDK pipeline (url_policy, confirm,
execute, browser_state) runs exactly as it does under the real model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from anthropic.types.beta import BetaToolResultBlockParam, BetaToolUseBlock
from playwright.sync_api import Page

from tests.conftest import FIXTURE_TENANT, PARABANK_POLICY, REPO, sign_in
from tests.fixtures.fake_model import comboboxes, ref_of
from tests.fixtures.site import BASE_URL, PASSWORD, FixtureSite
from understudy.discovery.driver import ApprovalRequest, DiscoveryBrowser
from understudy.discovery.goal import load_goal
from understudy.discovery.trajectory import Recorder
from understudy.evidence.run import RunContext
from understudy.policy.gate import PolicyGate
from understudy.policy.models import PolicyOverlay
from understudy.redaction.redactor import Redactor
from understudy.surface.web import WebSurface

GOAL = load_goal(REPO / "goals" / "open_account.yaml")
CREATE = "/parabank/services_proxy/bank/createAccount"


class Operator:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.asked: list[ApprovalRequest] = []

    def __call__(self, request: ApprovalRequest) -> bool:
        self.asked.append(request)
        return self.answer


@dataclass
class Harness:
    browser: DiscoveryBrowser
    recorder: Recorder
    operator: Operator
    site: FixtureSite
    page: Page
    run: RunContext
    _n: int = 0

    def call(self, member: str, **input: Any) -> BetaToolResultBlockParam:
        self._n += 1
        tool_use = BetaToolUseBlock.model_validate(
            {"type": "tool_use", "id": f"toolu_{self._n}", "name": member, "input": input}
            | {"toolset_name": "browser"}
        )
        return self.browser.tool_result(tool_use)

    def read(self) -> str:
        return text(self.call("read_page"))

    def events(self) -> list[dict[str, Any]]:
        lines = (self.run.dir / "events.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]


def text(result: BetaToolResultBlockParam) -> str:
    content = result["content"]
    if isinstance(content, str):
        return content
    return "\n".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text")


def ref(target: str) -> dict[str, str]:
    return {"type": "ref", "ref": target}


def harness(
    page: Page, site: FixtureSite, gate: PolicyGate, tmp_path: Path, answer: bool
) -> Harness:
    surface = WebSurface(page, gate, mode="discovery", run_id="driver-test")
    sign_in(page)
    run = RunContext(tmp_path / "runs", kind="discovery", redactor=Redactor(b"k" * 32))
    recorder = Recorder(run, surface, GOAL, GOAL.values(), screenshots=False)
    recorder.start()
    operator = Operator(answer)
    browser = DiscoveryBrowser(
        surface, recorder, approve=operator, settle_max_ms=500, commit_wait_ms=1_000
    )
    return Harness(browser, recorder, operator, site, page, run)


@pytest.fixture
def h(page: Page, site: FixtureSite, gate: PolicyGate, tmp_path: Path) -> Harness:
    return harness(page, site, gate, tmp_path, answer=True)


@pytest.fixture
def h_refusing(page: Page, site: FixtureSite, gate: PolicyGate, tmp_path: Path) -> Harness:
    return harness(page, site, gate, tmp_path, answer=False)


def open_form(h: Harness) -> str:
    h.call("navigate", url=f"{BASE_URL}/openaccount.htm")
    snapshot = h.read()
    for _ in range(20):
        if len(comboboxes(snapshot)) >= 2 and comboboxes(snapshot)[1][1]:
            return snapshot
        h.call("wait", duration=0.1)
        snapshot = h.read()
    raise AssertionError("the funding list never loaded")


# --- what the model is offered ----------------------------------------------------------


def test_members_outside_the_policy_and_the_driver_are_off_on_the_wire(h: Harness) -> None:
    configs = h.browser.to_dict()["configs"]
    off = {name for name, cfg in configs.items() if cfg == {"enabled": False}}

    assert {"find", "new_tab", "switch_tab", "javascript_exec", "file_upload", "hold_key"} <= off
    assert {"left_click_drag", "right_click", "mouse_move", "read_network"} <= off
    assert h.browser.served.isdisjoint(off)
    result = h.call("find", query="the submit button")
    assert result.get("is_error")


def test_a_tenant_overlay_switches_members_off_too(
    page: Page, site: FixtureSite, tmp_path: Path
) -> None:
    tenant = FIXTURE_TENANT.model_copy(update={"policy": PolicyOverlay(remove_actions=["hover"])})
    narrowed = harness(page, site, PolicyGate.load(PARABANK_POLICY, tenant), tmp_path, answer=True)

    assert narrowed.browser.to_dict()["configs"]["hover"] == {"enabled": False}


# --- refusals ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "rule"),
    [
        (f"{BASE_URL}/admin.htm", "routes.deny"),
        ("https://elsewhere.test/parabank/index.htm", "host"),
        ("admin.htm", "routes.deny"),  # relative to the current page
    ],
)
def test_navigation_off_policy_is_refused_before_any_request(
    h: Harness, url: str, rule: str
) -> None:
    before = len(h.site.requests)

    result = h.call("navigate", url=url)

    assert result.get("is_error") and f"Refused by policy ({rule})" in text(result)
    assert len(h.site.requests) == before
    assert h.recorder.trajectory.calls[-1].ok is False


def test_a_link_to_a_denied_page_is_refused_before_the_click(h: Harness) -> None:
    snapshot = h.read()

    result = h.call("left_click", target=ref(ref_of(snapshot, 'link "Admin Page"')))

    assert result.get("is_error") and "routes.deny" in text(result)
    assert h.site.reached("/parabank/admin.htm") == 0


def test_the_model_can_never_type_a_password(h: Harness) -> None:
    h.call("navigate", url=f"{BASE_URL}/index.htm")
    snapshot = h.read()
    password_box = [line for line in snapshot.splitlines() if "- textbox" in line][1]
    box = password_box.split("[ref=")[1].split("]")[0]

    filled = h.call("form_input", target=ref(box), value="guess-123")
    h.call("left_click", target=ref(box))
    typed = h.call("type", text="guess-456")

    assert filled.get("is_error") and "session provider" in text(filled)
    assert typed.get("is_error") and "session provider" in text(typed)
    assert h.page.locator("input[name='password']").input_value() == ""
    log = (h.run.dir / "events.jsonl").read_text()
    assert "guess-123" not in log and "guess-456" not in log and PASSWORD not in log


def test_a_stale_ref_is_refused_with_a_way_forward(h: Harness) -> None:
    snapshot = h.read()
    link = ref_of(snapshot, 'link "Accounts Overview"')
    h.call("navigate", url=f"{BASE_URL}/openaccount.htm")
    h.read()

    result = h.call("left_click", target=ref(link))

    assert result.get("is_error") and "read_page" in text(result)


# --- commits -----------------------------------------------------------------------------


def test_an_approved_commit_runs_once_and_is_recorded_as_one(h: Harness) -> None:
    snapshot = open_form(h)

    result = h.call("left_click", target=ref(ref_of(snapshot, 'button "Open New Account"')))

    assert not result.get("is_error"), text(result)
    assert [(r.member, r.target) for r in h.operator.asked] == [
        ("left_click", "button 'Open New Account'")
    ]
    assert h.site.reached(CREATE) == 1
    call = h.recorder.trajectory.calls[-1]
    assert (call.effect, call.approved, call.ok) == ("commit", True, True)
    assert call.element is not None and call.element.role == "button"
    assert "Account Opened!" in (call.after.snapshot if call.after else "")


def test_a_refused_commit_never_reaches_the_app(h_refusing: Harness) -> None:
    snapshot = open_form(h_refusing)

    result = h_refusing.call(
        "left_click", target=ref(ref_of(snapshot, 'button "Open New Account"'))
    )

    assert result.get("is_error") and "did not grant permission" in text(result)
    assert h_refusing.site.reached(CREATE) == 0
    control = [e for e in h_refusing.events() if e["kind"] == "control"]
    assert [e["data"]["approval"] for e in control] == ["refused"]


def test_enter_inside_a_commit_form_counts_as_the_commit(h_refusing: Harness) -> None:
    snapshot = open_form(h_refusing)
    funding = comboboxes(snapshot)[1][0]
    h_refusing.call("left_click", target=ref(funding))

    result = h_refusing.call("key", text="Return")

    assert result.get("is_error") and "did not grant permission" in text(result)
    assert [r.member for r in h_refusing.operator.asked] == ["key"]
    assert h_refusing.site.reached(CREATE) == 0


# --- recording ---------------------------------------------------------------------------


def test_choosing_from_a_list_binds_the_param_by_label_or_value(h: Harness) -> None:
    snapshot = open_form(h)
    kind, funding = (box for box, _ in comboboxes(snapshot)[:2])

    h.call("form_input", target=ref(kind), value="SAVINGS")
    h.call("form_input", target=ref(funding), value="13344")

    first, second = h.recorder.trajectory.calls[-2:]
    assert first.value is not None and first.value.param == "account_type"
    assert first.element is not None and first.element.selected == ("1", "SAVINGS")
    assert second.value is not None and second.value.param == "funding_account_id"
    assert second.value.sensitivity == "financial_id"
    acts = [e for e in h.events() if e["kind"] == "act" and e.get("member") == "form_input"]
    assert [e["value"] for e in acts] == ["SAVINGS", "••••4"]


def test_a_value_that_is_neither_a_param_nor_public_is_withheld(h: Harness) -> None:
    snapshot = open_form(h)
    funding = comboboxes(snapshot)[1][0]

    h.call("form_input", target=ref(funding), value="54321")

    value = h.recorder.trajectory.calls[-1].value
    assert value is not None and value.param is None and value.literal is None
    assert "54321" not in (h.run.dir / "events.jsonl").read_text()


def test_a_coordinate_click_is_recorded_by_the_element_it_hit(h: Harness) -> None:
    box = h.page.get_by_role("link", name="Open New Account").bounding_box()
    assert box is not None

    result = h.call(
        "left_click",
        target={
            "type": "coordinate",
            "x": int(box["x"] + 5),
            "y": int(box["y"] + box["height"] / 2),
        },
    )

    assert not result.get("is_error"), text(result)
    call = h.recorder.trajectory.calls[-1]
    assert call.element is not None
    assert (call.element.role, call.element.name) == ("link", "Open New Account")
    assert call.element.target is not None and call.after is not None
    assert call.after.route == "/openaccount.htm"


def test_a_dialog_is_answered_and_the_model_is_told(h: Harness) -> None:
    h.page.evaluate("() => { setTimeout(() => alert('Statement ready'), 0); }")

    result = h.call("wait", duration=0.3)

    assert not result.get("is_error")
    assert "Statement ready" in text(result) and "dismissed" in text(result)
    assert h.read()  # and the page can still be driven


def test_looking_after_an_action_becomes_what_the_action_is_judged_by(h: Harness) -> None:
    h.call("navigate", url=f"{BASE_URL}/openaccount.htm")
    h.call("wait", duration=0.2)
    h.read()

    navigate = [c for c in h.recorder.trajectory.calls if c.member == "navigate"][-1]
    assert navigate.observed_after is not None
    assert "combobox" in navigate.observed_after.snapshot


# --- review round 1 regressions ------------------------------------------------------------


def test_a_page_taken_off_policy_can_still_be_navigated_back(h: Harness) -> None:
    h.page.evaluate("() => { location.href = 'admin.htm'; }")
    h.page.wait_for_timeout(300)

    blocked = h.call("read_page")
    back = h.call("navigate", url=f"{BASE_URL}/openaccount.htm")

    assert blocked.get("is_error") and "navigate to an allowed address" in text(blocked)
    assert not back.get("is_error"), text(back)
    assert h.site.reached("/parabank/admin.htm") == 0
    assert h.recorder.trajectory.calls[-1].ok


def test_a_commit_is_recorded_even_when_the_page_is_taken_off_policy_right_after(
    h: Harness,
) -> None:
    snapshot = open_form(h)
    h.page.evaluate(
        """() => document.querySelector("input[type=button]").addEventListener("click",
             () => setTimeout(() => { location.href = 'admin.htm'; }, 50))"""
    )

    h.call("left_click", target=ref(ref_of(snapshot, 'button "Open New Account"')))

    assert h.site.reached(CREATE) == 1
    call = h.recorder.trajectory.calls[-1]
    assert (call.member, call.effect, call.approved, call.ok) == (
        "left_click",
        "commit",
        True,
        True,
    )
    assert [
        e["data"]["approved"]
        for e in h.events()
        if e["kind"] == "act" and e.get("member") == "left_click"
    ] == [True]


def test_a_committing_key_is_never_pressed_twice(h: Harness) -> None:
    snapshot = open_form(h)
    h.call("left_click", target=ref(comboboxes(snapshot)[1][0]))

    result = h.call("key", text="Enter", repeat=2)

    assert result.get("is_error") and "pressed once" in text(result)
    assert h.site.reached(CREATE) == 0
