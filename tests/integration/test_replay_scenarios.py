"""The replay scenario matrix (REPORT.md §3) against the routed fixture site.

Every row of the taxonomy table has a test that asserts the status, the code, the
commit state and what reached the app, so a fault that fails in the wrong way, or
a pass that skipped the commit, cannot look green.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Page

from tests.conftest import FIXTURE_TENANT, PARABANK_POLICY, REPO
from tests.fixtures.example_capability import VERIFIED, example_capability
from tests.fixtures.site import BASE_URL, PASSWORD, SSN, USERNAME, FixtureSite
from understudy.artifact.schema import (
    Capability,
    Disposition,
    Expectation,
    Fingerprint,
    Locator,
    RetryPolicy,
    Step,
    Target,
)
from understudy.artifact.signatures import DialogCondition, VisibleCondition
from understudy.chaos import MODAL_ID, Chaos, parse_fault
from understudy.policy.gate import PolicyGate
from understudy.policy.models import PolicyOverlay
from understudy.profile import AppProfile, Interstitial, load_profile
from understudy.redaction.redactor import Redactor
from understudy.replay.engine import Replayer
from understudy.replay.preflight import grant
from understudy.replay.result import ReplayResult
from understudy.session import Credentials
from understudy.tenant import TenantBinding

CREATE = "/parabank/services_proxy/bank/createAccount"
PROFILE = load_profile(REPO / "apps" / "parabank" / "profile.yaml")
INPUTS = {"account_type": "SAVINGS", "funding_account_id": "12345"}
FIRST_NEW_ID = "13566"


def fast(capability: Capability, ms: int) -> Capability:
    """Caps every step's timeout, so the timeout scenarios take seconds, not minutes."""
    copy = capability.model_copy(deep=True)
    for step in copy.steps:
        step.timeout_ms = min(step.timeout_ms, ms)
    return copy


def never_open() -> Page:
    raise AssertionError("preflight should have stopped this run before the browser")


class Replay:
    def __init__(self, page: Page, gate: PolicyGate, runs: Path) -> None:
        self.page = page
        self.gate = gate
        self.runs = runs
        self.last: ReplayResult | None = None

    def __call__(
        self,
        capability: Capability | None = None,
        inputs: dict[str, str] | None = None,
        *,
        faults: Sequence[str] = (),
        approve: bool = True,
        allow_draft: bool = True,
        profile: AppProfile = PROFILE,
        gate: PolicyGate | None = None,
        timeout_ms: int = 2_000,
        open_page: Callable[[], Page] | None = None,
        screenshots: bool = False,
    ) -> ReplayResult:
        cap = fast(capability or example_capability(), timeout_ms)
        values = INPUTS if inputs is None else inputs
        gate = gate or self.gate
        replayer = Replayer(
            cap,
            gate=gate,
            profile=profile,
            credentials=Credentials(USERNAME, PASSWORD),
            runs_dir=self.runs,
            redactor=Redactor(b"k" * 32),
            screenshots=screenshots,
        )
        chaos = Chaos([parse_fault(f, cap) for f in faults], gate) if faults else None
        approval = grant(cap, values, gate.tenant.id, "test") if approve else None
        self.last = replayer.run(
            values,
            open_page=open_page or (lambda: self.page),
            approval=approval,
            allow_draft=allow_draft,
            chaos=chaos,
        )
        return self.last

    def run_dir(self) -> Path:
        assert self.last is not None
        return self.runs / self.last.run_id

    def events(self) -> list[dict[str, Any]]:
        lines = (self.run_dir() / "events.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]


@pytest.fixture
def replay(page: Page, site: FixtureSite, gate: PolicyGate, tmp_path: Path) -> Replay:
    # `site` first: its routes go on the context before the engine builds the surface.
    return Replay(page, gate, tmp_path / "runs")


def recovered(result: ReplayResult) -> list[tuple[str, str]]:
    return [(r.handler, r.step_id) for r in result.recoveries]


# --- the happy path -------------------------------------------------------------------


def test_the_flow_succeeds_with_typed_outputs_and_one_commit(
    replay: Replay, site: FixtureSite
) -> None:
    result = replay()

    assert result.status == "succeeded", result
    assert result.outputs == {"new_account_id": FIRST_NEW_ID}
    assert FIRST_NEW_ID in site.accounts
    assert result.side_effects.commit_state == "committed"
    assert result.side_effects.commit_step == "s4_submit"
    assert result.side_effects.commit_requests == 1
    assert site.reached(CREATE) == 1
    assert result.recoveries == [] and result.drift == []
    assert result.completed_by == "automation"
    assert [t.step_id for t in result.timings.steps] == [s.id for s in example_capability().steps]


def test_the_persisted_result_is_redacted_but_keeps_its_identifiers(replay: Replay) -> None:
    result = replay()
    stored = json.loads((replay.run_dir() / "result.json").read_text())

    assert stored["status"] == "succeeded"
    assert stored["artifact_hash"] == result.artifact_hash  # not mangled into [SECRET]
    assert stored["run_id"] == result.run_id
    assert stored["outputs"]["new_account_id"] != FIRST_NEW_ID
    assert stored["outputs"]["new_account_id"].startswith("••••")
    run = json.loads((replay.run_dir() / "run.json").read_text())
    assert run["inputs"] == {"account_type": "SAVINGS", "funding_account_id": "••••5"}


# --- rejected: nothing touches the UI --------------------------------------------------


@pytest.mark.parametrize(
    ("inputs", "problem"),
    [
        ({"account_type": "GOLD", "funding_account_id": "12345"}, "must be one of"),
        ({"account_type": "SAVINGS", "funding_account_id": "12a45"}, r"does not match ^\d+$"),
        ({"account_type": "SAVINGS"}, "'funding_account_id' is required"),
        ({**INPUTS, "note": "x"}, "unknown input 'note'"),
    ],
)
def test_bad_input_is_rejected_before_the_browser_opens(
    replay: Replay, site: FixtureSite, inputs: dict[str, str], problem: str
) -> None:
    result = replay(inputs=inputs, open_page=never_open)

    assert result.status == "rejected"
    assert result.rejection is not None and result.rejection.code == "invalid_input"
    assert any(problem in p for p in result.rejection.problems), result.rejection.problems
    assert "12a45" not in json.dumps(result.rejection.problems)  # values are never echoed
    assert site.requests == []


def test_a_draft_is_rejected_without_allow_draft(replay: Replay, site: FixtureSite) -> None:
    result = replay(allow_draft=False, open_page=never_open)

    assert result.status == "rejected"
    assert result.rejection is not None and result.rejection.code == "not_approved"
    assert site.requests == []


def test_a_commit_without_an_approval_is_rejected(replay: Replay, site: FixtureSite) -> None:
    result = replay(approve=False, open_page=never_open)

    assert result.rejection is not None and result.rejection.code == "approval_required"
    assert site.requests == []


def test_an_approval_for_other_inputs_does_not_carry_over(
    replay: Replay, site: FixtureSite, gate: PolicyGate, tmp_path: Path
) -> None:
    capability = fast(example_capability(), 2_000)
    replayer = Replayer(
        capability,
        gate=gate,
        profile=PROFILE,
        credentials=Credentials(USERNAME, PASSWORD),
        runs_dir=tmp_path / "runs",
    )
    approval = grant(capability, {**INPUTS, "funding_account_id": "54321"}, "fixture", "test")
    result = replayer.run(
        INPUTS, open_page=never_open, approval=approval, allow_draft=True, chaos=None
    )

    assert result.rejection is not None and result.rejection.code == "approval_required"
    assert result.rejection.problems == ["the approval was granted for a different inputs_hash"]


def test_a_tightened_tenant_policy_rejects_the_artifact(replay: Replay, site: FixtureSite) -> None:
    tenant = FIXTURE_TENANT.model_copy(
        update={"policy": PolicyOverlay(deny_routes=["/openaccount.htm"])}
    )
    result = replay(gate=PolicyGate.load(PARABANK_POLICY, tenant), open_page=never_open)

    assert result.rejection is not None and result.rejection.code == "policy"
    assert site.requests == []


def test_a_commit_step_with_retries_is_not_run(replay: Replay) -> None:
    capability = example_capability()
    capability.steps[3].retry = RetryPolicy(max_attempts=1, backoff_ms=[100])
    result = replay(capability, open_page=never_open)

    assert result.rejection is not None and result.rejection.code == "invalid_artifact"
    assert any("L003" in p for p in result.rejection.problems)


# --- business outcomes ---------------------------------------------------------------


def test_a_funding_account_the_customer_does_not_own_is_a_business_outcome(
    replay: Replay, site: FixtureSite
) -> None:
    result = replay(inputs={"account_type": "SAVINGS", "funding_account_id": "99999"})

    assert result.status == "business_outcome", result
    assert result.outcome is not None
    assert result.outcome.code == "funding_account_not_found"
    assert result.outcome.step_id == "s3_funding_account"
    assert result.side_effects.commit_state == "not_attempted"
    assert site.reached(CREATE) == 0
    assert result.failure is None and result.escalation is None


# --- slow but normal ---------------------------------------------------------------------


def test_a_slow_funding_list_is_waited_for_without_any_event(
    replay: Replay, site: FixtureSite
) -> None:
    site.accounts_delay_ms = 800
    result = replay()

    assert result.status == "succeeded", result
    assert result.recoveries == [] and result.drift == []
    assert not [e for e in replay.events() if e["kind"] in ("recovery", "drift", "escalation")]


def test_injected_latency_on_the_funding_list_is_just_slow(replay: Replay) -> None:
    result = replay(faults=["latency:/services_proxy/bank/customers/*/accounts:700"])

    assert result.status == "succeeded", result
    assert result.recoveries == []


# --- recovered ---------------------------------------------------------------------------


def test_a_503_on_the_form_page_is_retried_and_recovers(replay: Replay, site: FixtureSite) -> None:
    result = replay(faults=["http_503_once:/openaccount.htm"])

    assert result.status == "succeeded", result
    assert recovered(result) == [("retry_transient", "s1_open_form")]
    assert site.reached(CREATE) == 1


def test_an_expired_session_before_the_commit_logs_in_again_and_restarts(
    replay: Replay, site: FixtureSite
) -> None:
    result = replay(faults=["session_expired_at=s3"])

    assert result.status == "succeeded", result
    assert recovered(result) == [("reauthenticate", "s3_funding_account")]
    assert result.recoveries[0].restarted_from == "s1_open_form"
    assert site.reached(CREATE) == 1
    assert site.reached("/parabank/login.htm") == 2


def test_a_known_interstitial_is_dismissed(replay: Replay, site: FixtureSite) -> None:
    profile = PROFILE.model_copy(
        update={
            "interstitials": [
                Interstitial(
                    id="maintenance_notice",
                    description="The maintenance notice is showing",
                    when=VisibleCondition(css=f"#{MODAL_ID}"),
                    dismiss=Target(
                        description="The notice's OK button",
                        fingerprint=Fingerprint(role="button", name="OK"),
                        locators=[
                            Locator(
                                by="css",
                                value=f"#{MODAL_ID} button",
                                stability="high",
                                note="test profile",
                            )
                        ],
                    ),
                )
            ]
        }
    )
    result = replay(faults=["unknown_modal_at:s3"], profile=profile)

    assert result.status == "succeeded", result
    assert recovered(result) == [("dismiss_interstitial", "s3_funding_account")]
    assert site.reached(CREATE) == 1


def test_recovery_stops_when_the_budget_is_spent(replay: Replay, site: FixtureSite) -> None:
    capability = example_capability()
    capability.recovery.budget = 0
    result = replay(capability, faults=["http_503_once:/openaccount.htm"])

    assert result.status == "failed"
    assert result.failure is not None
    assert (result.failure.category, result.failure.code) == ("app", "app_error")
    assert "budget" in result.failure.observed
    assert site.reached(CREATE) == 0


# --- needs a human ---------------------------------------------------------------------


def test_an_unknown_modal_over_the_target_goes_to_a_human(
    replay: Replay, site: FixtureSite
) -> None:
    result = replay(
        inputs={**INPUTS, "funding_account_id": "54321"}, faults=["unknown_modal_at:s3"]
    )

    assert result.status == "needs_human", result
    assert result.escalation is not None
    assert result.escalation.reason == "unknown_blocking_state"
    assert result.escalation.step_id == "s3_funding_account"
    assert MODAL_ID in result.escalation.detail
    assert result.side_effects.commit_state == "not_attempted"
    assert site.reached(CREATE) == 0
    # The select under the modal was not touched: it still shows the first option.
    assert replay.page.locator("select#fromAccountId").input_value() == "12345"
    assert result.escalation.evidence is not None
    assert (replay.run_dir() / (result.escalation.evidence.screenshot or "")).is_file()


def test_no_answer_after_the_commit_is_unknown_and_never_retried(
    replay: Replay, site: FixtureSite
) -> None:
    result = replay(faults=["drop_commit_response"])

    assert result.status == "needs_human", result
    assert result.escalation is not None
    assert result.escalation.reason == "commit_outcome_unknown"
    assert result.escalation.step_id == "s4_submit"
    assert result.side_effects.commit_state == "unknown"
    assert result.side_effects.commit_requests == 1
    assert result.recoveries == []
    # The timeout reads in seconds, so the redactor's digit-run rule leaves it legible.
    stored = json.loads((replay.run_dir() / "result.json").read_text())
    assert stored["escalation"]["detail"].endswith(" s")
    assert "••••" not in stored["escalation"]["detail"]


def test_an_app_error_after_the_commit_goes_to_a_human(replay: Replay, site: FixtureSite) -> None:
    result = replay(faults=["app_error_at_commit"])

    assert result.status == "needs_human", result
    assert result.escalation is not None
    assert result.escalation.reason == "app_error_after_commit"
    assert result.side_effects.commit_state == "unknown"
    assert result.side_effects.commit_requests == 1
    assert site.reached(CREATE) == 0  # the fault answered in the server's place
    assert result.recoveries == []


# --- failed ------------------------------------------------------------------------------


def test_an_app_error_before_the_commit_fails_as_an_app_failure(
    replay: Replay, site: FixtureSite
) -> None:
    site.accounts_status = 500
    result = replay()

    assert result.status == "failed", result
    assert result.failure is not None
    assert (result.failure.category, result.failure.code) == ("app", "app_error")
    assert result.side_effects.commit_state == "not_attempted"
    assert site.reached(CREATE) == 0


def test_a_renamed_commit_button_fails_with_every_attempt_and_evidence(
    replay: Replay, site: FixtureSite
) -> None:
    result = replay(faults=["rename_target:s4"])

    assert result.status == "failed", result
    failure = result.failure
    assert failure is not None
    assert (failure.category, failure.code, failure.step_id) == (
        "automation",
        "target_not_found",
        "s4_submit",
    )
    assert [(a.by, a.outcome) for a in failure.attempts] == [
        ("role", "no_match"),
        ("css", "no_match"),
    ]
    assert "Open New Account" in failure.expected
    assert failure.evidence is not None
    for name in ("screenshot", "dom", "aria", "console"):
        path = getattr(failure.evidence, name)
        assert path is not None and (replay.run_dir() / path).is_file(), name
    assert result.side_effects.commit_state == "not_attempted"
    assert site.reached(CREATE) == 0


def test_a_success_checkpoint_that_already_holds_stops_the_commit(
    replay: Replay, site: FixtureSite
) -> None:
    """Lint can only demand a negative control; whether the checkpoint really is false
    before the commit is something only the running page can show."""
    capability = example_capability()
    capability.success.signature = VisibleCondition(role="heading", name="Open New Account")
    result = replay(capability)

    assert result.status == "failed", result
    assert result.failure is not None and result.failure.code == "checkpoint_vacuous"
    assert site.reached(CREATE) == 0


def test_a_click_that_leaves_the_allowlist_fails_on_policy(
    replay: Replay, site: FixtureSite
) -> None:
    capability = example_capability()
    capability.steps.insert(
        1,
        Step(
            id="s1b_admin",
            intent="Open the admin page",
            action="click",
            effect="read",
            target=Target(
                description="The Admin Page link",
                fingerprint=Fingerprint(role="link", name="Admin Page"),
                locators=[
                    Locator(
                        by="role",
                        value={"role": "link", "name": "Admin Page", "exact": True},
                        stability="high",
                        verified=VERIFIED,
                        note="test",
                    )
                ],
            ),
            expect=[
                Expectation(
                    description="The admin page",
                    when=VisibleCondition(role="heading", name="Administration"),
                    then=Disposition(do="continue"),
                )
            ],
        ),
    )
    result = replay(capability)

    assert result.status == "failed", result
    assert result.failure is not None
    assert (result.failure.category, result.failure.code) == ("policy", "policy_blocked")
    assert result.failure.step_id == "s1b_admin"
    assert site.reached("/parabank/admin.htm") == 0


def test_an_unreachable_host_is_an_infra_failure(replay: Replay, site: FixtureSite) -> None:
    tenant = TenantBinding(
        id="nowhere", vendor_app="parabank", base_url="https://nowhere.invalid/parabank"
    )
    result = replay(gate=PolicyGate.load(PARABANK_POLICY, tenant))

    assert result.status == "failed", result
    assert result.failure is not None
    assert (result.failure.category, result.failure.code) == ("infra", "infra_error")
    assert result.side_effects.commit_state == "not_attempted"


# --- drift --------------------------------------------------------------------------------


def test_a_fallback_locator_proceeds_and_reports_drift(replay: Replay, site: FixtureSite) -> None:
    capability = example_capability()
    target = capability.steps[3].target
    assert target is not None
    target.locators[0].value = {"role": "button", "name": "Open An Account", "exact": True}
    result = replay(capability)

    assert result.status == "succeeded", result
    assert [(d.step_id, d.locator_index, d.by) for d in result.drift] == [("s4_submit", 1, "css")]
    assert [e["step_id"] for e in replay.events() if e["kind"] == "drift"] == ["s4_submit"]
    assert site.reached(CREATE) == 1


# --- evidence ------------------------------------------------------------------------------


def test_nothing_sensitive_reaches_the_run_folder(replay: Replay, site: FixtureSite) -> None:
    replay(screenshots=True)
    succeeded = replay.run_dir()
    replay(faults=["rename_target:s4"], screenshots=True)
    failed = replay.run_dir()

    assert list((succeeded / "screenshots").glob("*.png"))
    texts = [
        p.read_text(encoding="utf-8")
        for run in (succeeded, failed)
        for p in run.rglob("*")
        if p.is_file() and p.suffix != ".png"
    ]
    blob = "\n".join(texts)
    assert (failed / "failure" / "dom.html").is_file()
    assert "[masked]" in (failed / "failure" / "dom.html").read_text()
    # The customer's name is in a panel the profile masks; it must not leak through
    # the DOM dump or the accessibility snapshot of the failure bundle.
    for secret in (
        PASSWORD,
        site.session_id,
        SSN,
        "12345",
        FIRST_NEW_ID,
        BASE_URL + ";",
        "John Smith",
    ):
        assert secret not in blob, secret


# --- review round: regressions ------------------------------------------------------------


def test_an_unknown_native_dialog_goes_to_a_human_without_hanging(
    replay: Replay, page: Page, site: FixtureSite
) -> None:
    """Screenshots and scripts block while a dialog is open, whatever their timeout,
    so the failure bundle must not read the page."""
    page.add_init_script(
        """window.addEventListener("load", () => {
          if (location.pathname.endsWith("openaccount.htm")) setTimeout(() => alert("Heads up"), 0);
        });"""
    )
    result = replay(screenshots=True)

    assert result.status == "needs_human", result
    assert result.escalation is not None
    assert result.escalation.reason == "unknown_blocking_state"
    assert "alert" in result.escalation.detail
    assert (replay.run_dir() / "failure" / "dialog.txt").read_text().startswith("alert: ")
    assert result.escalation.evidence is not None
    assert result.escalation.evidence.dialog == "failure/dialog.txt"
    assert site.reached(CREATE) == 0


def test_a_known_dialog_is_answered_by_its_rule(
    replay: Replay, page: Page, site: FixtureSite
) -> None:
    page.add_init_script(
        """window.addEventListener("load", () => {
          if (location.pathname.endsWith("openaccount.htm")) setTimeout(() => alert("Heads up"), 0);
        });"""
    )
    profile = PROFILE.model_copy(
        update={
            "interstitials": [
                Interstitial(
                    id="heads_up",
                    description="The heads-up alert",
                    when=DialogCondition(message_regex="^Heads up$"),
                    dialog="accept",
                )
            ]
        }
    )
    result = replay(profile=profile)

    assert result.status == "succeeded", result
    assert [r.handler for r in result.recoveries] == ["dismiss_interstitial"]
    assert site.reached(CREATE) == 1


def test_an_interstitial_that_fades_out_is_dismissed_once(
    replay: Replay, page: Page, site: FixtureSite
) -> None:
    page.add_init_script(
        """window.addEventListener("load", () => {
          if (!location.pathname.endsWith("openaccount.htm")) return;
          const d = document.createElement("div");
          d.id = "fade-notice";
          d.style = "position:fixed;inset:0;z-index:99999;background:#fff";
          d.innerHTML = "<button type='button'>Close</button>";
          d.querySelector("button").onclick = () => setTimeout(() => d.remove(), 300);
          document.body.appendChild(d);
        });"""
    )
    profile = PROFILE.model_copy(
        update={
            "interstitials": [
                Interstitial(
                    id="fade_notice",
                    description="A notice that fades out",
                    when=VisibleCondition(css="#fade-notice"),
                    dismiss=Target(
                        description="Its Close button",
                        fingerprint=Fingerprint(role="button", name="Close"),
                        locators=[
                            Locator(
                                by="css", value="#fade-notice button", stability="high", note="t"
                            )
                        ],
                    ),
                )
            ]
        }
    )
    result = replay(profile=profile)

    assert result.status == "succeeded", result
    assert recovered(result) == [("dismiss_interstitial", "s1_open_form")]


def test_an_app_error_panel_is_retried_from_the_reentry_point_and_recovers(
    replay: Replay, site: FixtureSite
) -> None:
    """Retrying a select in place would find the same error panel on screen; going back
    to the re-entry point reloads the page."""
    result = replay(faults=["http_503_once:/services_proxy/bank/customers/*/accounts"])

    assert result.status == "succeeded", result
    assert [r.handler for r in result.recoveries] == ["retry_transient"]
    restart = result.recoveries[0]
    # Seen at s1 it is a navigate retried in place; seen later it restarts from s1.
    assert restart.restarted_from == ("s1_open_form" if restart.step_id != "s1_open_form" else None)
    assert site.reached(CREATE) == 1


def test_refused_credentials_are_reported_outside_any_step(
    replay: Replay, gate: PolicyGate
) -> None:
    capability = fast(example_capability(), 2_000)
    replayer = Replayer(
        capability,
        gate=gate,
        profile=PROFILE,
        credentials=Credentials(USERNAME, "not-the-password"),
        runs_dir=replay.runs,
        screenshots=False,
    )
    result = replayer.run(
        INPUTS,
        open_page=lambda: replay.page,
        approval=grant(capability, INPUTS, "fixture", "t"),
        allow_draft=True,
    )

    assert result.status == "failed"
    assert result.failure is not None
    assert (result.failure.category, result.failure.code) == ("infra", "login_failed")
    assert result.failure.step_id is None
    assert "not-the-password" not in (replay.runs / result.run_id / "result.json").read_text()


def test_a_commit_step_without_expectations_is_not_run(replay: Replay) -> None:
    capability = example_capability()
    capability.steps[3].expect = []
    result = replay(capability, open_page=never_open)

    assert result.rejection is not None and result.rejection.code == "invalid_artifact"
    assert any("L003" in p and "no expectations" in p for p in result.rejection.problems)


def test_network_faults_wait_for_the_capability_not_the_login(
    replay: Replay, page: Page, site: FixtureSite
) -> None:
    """Live ParaBank's overview (where login lands) loads the accounts list too; a
    one-shot fault must not be spent there."""
    page.add_init_script(
        """window.addEventListener("load", () => {
          const path = location.pathname;  // may carry ;jsessionid=
          if (path.includes("/overview.htm") || path.includes("/login.htm"))
            fetch("/parabank/services_proxy/bank/customers/12212/accounts");
        });"""
    )
    result = replay(faults=["http_503_once:/services_proxy/bank/customers/*/accounts"])

    assert result.status == "succeeded", result
    assert [r.handler for r in result.recoveries] == ["retry_transient"]


HEADS_UP = """window.addEventListener("load", () => {
  if (location.pathname.endsWith("openaccount.htm")) setTimeout(() => alert("Heads up"), 0);
});"""


def heads_up_profile() -> AppProfile:
    rule = Interstitial(
        id="heads_up",
        description="The heads-up alert",
        when=DialogCondition(message_regex="^Heads up$"),
        dialog="accept",
    )
    return PROFILE.model_copy(update={"interstitials": [rule]})


def test_a_dialog_a_step_expects_is_acknowledged_and_does_not_block_the_next(
    replay: Replay, page: Page, site: FixtureSite
) -> None:
    page.add_init_script(HEADS_UP)
    capability = example_capability()
    capability.steps[0].expect = [
        Expectation(
            description="The page greets us with an alert",
            when=DialogCondition(message_regex="^Heads up$"),
            then=Disposition(do="continue"),
        )
    ]
    result = replay(capability)

    assert result.status == "succeeded", result
    assert site.reached(CREATE) == 1


def test_a_known_dialog_is_unknown_when_the_artifact_does_not_allow_dismissing(
    replay: Replay, page: Page, site: FixtureSite
) -> None:
    page.add_init_script(HEADS_UP)
    capability = example_capability()
    capability.recovery.allowed = ["retry_transient", "reauthenticate"]
    result = replay(capability, profile=heads_up_profile())

    assert result.status == "needs_human", result
    assert result.escalation is not None and result.escalation.reason == "unknown_blocking_state"
    assert "dismissed" in (replay.run_dir() / "failure" / "dialog.txt").read_text()
    assert site.reached(CREATE) == 0


def test_a_rule_answered_dialog_does_not_use_up_the_budget(
    replay: Replay, page: Page, site: FixtureSite
) -> None:
    page.add_init_script(HEADS_UP)
    capability = example_capability()
    capability.recovery.budget = 0
    result = replay(capability, profile=heads_up_profile())

    assert result.status == "succeeded", result
    assert [r.handler for r in result.recoveries] == ["dismiss_interstitial"]
