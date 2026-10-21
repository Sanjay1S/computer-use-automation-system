"""The compiler's rules, one at a time, on hand-built trajectories (plan §6)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import FIXTURE_TENANT, PARABANK_POLICY, REPO
from tests.fixtures.site import BASE_URL
from understudy.artifact import signatures as sig
from understudy.artifact.schema import (
    Fingerprint,
    LiteralValue,
    Locator,
    ParamValue,
    SecretValue,
    Target,
    Verification,
)
from understudy.discovery.compiler import CompileError, compile_trajectory
from understudy.discovery.goal import load_goal
from understudy.discovery.trajectory import (
    ElementRecord,
    PageState,
    RecordedCall,
    RecordedOutput,
    StepValue,
    SuccessClaim,
    Trajectory,
)
from understudy.policy.gate import PolicyGate

GOAL = load_goal(REPO / "goals" / "open_account.yaml")
GATE = PolicyGate.load(PARABANK_POLICY, FIXTURE_TENANT)
VERIFIED = Verification(unique=True, same_element=True, run_id="r")

FORM = """\
- heading "Open New Account" [level=1] [ref=ref_1]
- combobox [ref=ref_2]:
  - option "CHECKING" [selected]
- combobox [ref=ref_3]:
  - option "13344"
- textbox "Note" [ref=ref_5]
- button "Open New Account" [ref=ref_4]"""
DONE = """\
- heading "Account Opened!" [level=1] [ref=ref_6]
- paragraph [ref=ref_7]: Congratulations, your account is now open.
- link "13566" [ref=ref_8]"""


def page(route: str, snapshot: str = FORM) -> PageState:
    return PageState(url=f"{BASE_URL}{route}", route=route, title="t", snapshot=snapshot)


def element(
    css: str, *, tag: str = "select", role: str | None = "combobox", name: str | None = None
) -> ElementRecord:
    return ElementRecord(
        description=f"{role or tag} {css}",
        tag=tag,
        role=role,
        name=name,
        input_type="text" if tag == "input" else None,
        target=Target(
            description=f"{role or tag} {css}",
            fingerprint=Fingerprint(role=role, tag=tag),
            locators=[
                Locator(by="css", value=css, stability="medium", note="n", verified=VERIFIED)
            ],
        ),
    )


class Calls:
    def __init__(self) -> None:
        self.seq = 0
        self.items: list[RecordedCall] = []

    def add(self, member: str, **fields: Any) -> RecordedCall:
        self.seq += 1
        fields.setdefault("before", page("/openaccount.htm"))
        fields.setdefault("after", fields["before"])
        call = RecordedCall(seq=self.seq, turn=self.seq, member=member, ok=True, **fields)
        self.items.append(call)
        return call

    def insert(self, index: int, member: str, **fields: Any) -> None:
        """Insert a call and renumber, so call numbers stay in recorded order."""
        self.items.insert(index, self.add(member, **fields))
        self.items.pop()
        self.items = [c.model_copy(update={"seq": i + 1}) for i, c in enumerate(self.items)]


def the_flow(calls: Calls, *, start_with_navigate: bool = True) -> None:
    if start_with_navigate:
        calls.add(
            "navigate",
            url=f"{BASE_URL}/openaccount.htm;jsessionid=ABC123",
            before=page("/overview.htm", '- heading "Accounts Overview" [level=1]'),
            after=page("/openaccount.htm"),
        )
    calls.add(
        "form_input",
        element=element("select#type"),
        value=StepValue(param="account_type"),
        effect="input",
    )
    calls.add(
        "form_input",
        element=element("select#fromAccountId"),
        value=StepValue(param="funding_account_id", sensitivity="financial_id"),
        effect="input",
    )
    calls.add(
        "left_click",
        element=element(
            "input#submit", tag="input", role="button", name="Open New Account"
        ).model_copy(update={"input_type": "button"}),
        effect="commit",
        approved=True,
        after=page("/openaccount.htm", DONE),
    )


def trajectory(calls: Calls, *, evidence: list[ElementRecord] | None = None) -> Trajectory:
    link = element("a#newAccountId", tag="a", role="link")
    heading = element("h1.title", tag="h1", role="heading", name="Account Opened!")
    return Trajectory(
        run_id="discovery-x",
        tenant="fixture",
        capability=GOAL.capability,
        goal="g",
        params={"account_type": "SAVINGS", "funding_account_id": "13344"},
        start=page("/overview.htm"),
        calls=calls.items,
        outputs=[
            RecordedOutput(
                name="new_account_id",
                description="d",
                element=link,
                sample="13566",
                after_seq=calls.seq,
            )
        ],
        success=SuccessClaim(
            summary="Opened it",
            evidence=evidence or [heading, link],
            page=page("/openaccount.htm", DONE),
        ),
    )


def compiled(calls: Calls, **kw: Any) -> Any:
    return compile_trajectory(
        trajectory(calls, **kw), GOAL, GATE, models_served=["claude-opus-5-5"]
    )


def test_the_recorded_flow_compiles_lint_clean() -> None:
    calls = Calls()
    the_flow(calls)

    result = compiled(calls)

    assert result.report.ok, [str(f) for f in result.report.findings]
    steps = result.capability.steps
    assert [s.action for s in steps] == ["navigate", "select", "select", "click", "extract"]
    assert steps[0].url == "{base_url}/openaccount.htm"  # no session id, no host
    assert steps[0].reentry_point and not any(s.reentry_point for s in steps[1:])
    assert steps[3].effect == "commit" and steps[3].retry.max_attempts == 0
    assert steps[3].timeout_ms == 30_000
    assert steps[4].extract is not None and steps[4].extract.regex == r"^\s*(\d+)\s*$"


def test_each_step_waits_on_what_changed_after_it() -> None:
    calls = Calls()
    the_flow(calls)
    steps = compiled(calls).capability.steps

    arrival = steps[0].expect[0].when
    assert isinstance(arrival, sig.AllCondition)
    assert sig.UrlCondition(path_regex=r"/openaccount\.htm$") in arrival.all
    assert sig.VisibleCondition(role="heading", name="Open New Account") in arrival.all
    assert steps[1].expect[0].when == sig.SelectedCondition(
        target="select#type", param="account_type"
    )
    assert steps[1].wait_before == sig.OptionsLoadedCondition(target="select#type", min=1)
    assert steps[3].expect[0].when == sig.VisibleCondition(role="heading", name="Account Opened!")


def test_a_free_form_param_in_a_list_gets_a_not_found_outcome_and_an_enum_does_not() -> None:
    calls = Calls()
    the_flow(calls)
    capability = compiled(calls).capability

    assert [(o.code, o.detected_by) for o in capability.outcomes] == [
        ("funding_account_id_not_found", capability.steps[2].id)
    ]
    assert "Nothing was committed" in capability.outcomes[0].description
    assert capability.steps[1].on_target_absent is None


def test_a_flow_that_does_not_start_with_an_address_gets_one() -> None:
    calls = Calls()
    the_flow(calls, start_with_navigate=False)

    result = compiled(calls)

    first = result.capability.steps[0]
    assert (first.action, first.url, first.reentry_point) == (
        "navigate",
        "{base_url}/openaccount.htm",
        True,
    )
    assert any("re-entry point" in note for note in result.notes)


def test_a_click_into_a_field_then_typing_there_is_one_fill_and_only_the_last_value_counts() -> (
    None
):
    calls = Calls()
    the_flow(calls)
    note = element("input#note", tag="input", role="textbox", name="Note")
    calls.insert(1, "left_click", element=note, effect="input")
    calls.insert(2, "type", element=note, value=StepValue(literal="first"), effect="input")
    calls.insert(3, "form_input", element=note, value=StepValue(literal="final"), effect="input")

    steps = compiled(calls).capability.steps

    fills = [s for s in steps if s.action == "fill"]
    assert len(fills) == 1 and fills[0].value == LiteralValue(literal="final")
    assert not [s for s in steps if s.action == "click" and s.effect != "commit"]


def test_tab_presses_are_dropped_with_a_note() -> None:
    calls = Calls()
    the_flow(calls)
    calls.insert(2, "key", element=element("select#type"), key="Tab", effect="input")

    result = compiled(calls)

    assert "press" not in [s.action for s in result.capability.steps]
    assert any("Tab" in note for note in result.notes)


def test_a_value_that_is_not_a_param_and_not_public_is_withheld_not_written() -> None:
    calls = Calls()
    the_flow(calls)
    calls.items[2] = calls.items[2].model_copy(
        update={"value": StepValue(sensitivity="financial_id")}
    )

    result = compiled(calls)

    assert isinstance(result.capability.steps[2].value, SecretValue)
    assert any("withheld" in note for note in result.notes)
    assert "funding_account_id" in str([f for f in result.report.findings if f.code == "L004"])


def test_param_values_in_a_query_become_placeholders() -> None:
    calls = Calls()
    calls.add(
        "navigate",
        url=f"{BASE_URL}/activity.htm;jsessionid=ABC?id=13344&tab=summary",
        after=page("/activity.htm", '- heading "Account Details" [level=1]'),
    )
    the_flow(calls, start_with_navigate=False)

    url = compiled(calls).capability.steps[0].url

    assert url == "{base_url}/activity.htm?id={funding_account_id}&tab=summary"


def test_an_address_outside_the_tenant_does_not_compile() -> None:
    calls = Calls()
    calls.add("navigate", url="https://elsewhere.test/parabank/openaccount.htm", after=page("/x"))
    the_flow(calls, start_with_navigate=False)

    with pytest.raises(CompileError, match="outside the tenant"):
        compiled(calls)


def test_an_element_without_verified_locators_does_not_compile() -> None:
    calls = Calls()
    the_flow(calls)
    lost = calls.items[1].element
    assert lost is not None
    calls.items[1] = calls.items[1].model_copy(
        update={
            "element": lost.model_copy(update={"target": None, "rejected": [("css", "matched 2")]})
        }
    )

    with pytest.raises(CompileError, match=r"no locator was verified.*matched 2"):
        compiled(calls)


def test_a_run_that_never_declared_success_does_not_compile() -> None:
    calls = Calls()
    the_flow(calls)
    run = trajectory(calls).model_copy(update={"success": None})

    with pytest.raises(CompileError, match="never declared success"):
        compile_trajectory(run, GOAL, GATE)


def test_evidence_already_showing_at_the_start_is_left_out_of_success() -> None:
    calls = Calls()
    the_flow(calls)
    stale = element("h1.form", tag="h1", role="heading", name="Open New Account")
    fresh = element("h1.done", tag="h1", role="heading", name="Account Opened!")

    result = compiled(calls, evidence=[stale, fresh])

    assert result.capability.success.signature == sig.VisibleCondition(
        role="heading", name="Account Opened!"
    )
    assert any("already showing" in note for note in result.notes)
    control = result.capability.success.negative_control
    assert isinstance(control, sig.AllCondition)
    assert control.all[0] == sig.VisibleCondition(role="heading", name="Open New Account")
    assert control.all[1] == sig.NotCondition(not_=result.capability.success.signature)


def test_data_in_a_heading_is_never_a_checkpoint_unless_it_is_a_param() -> None:
    calls = Calls()
    the_flow(calls)
    data = page("/openaccount.htm", '- heading "Account 13566 opened" [level=1]\n- paragraph: Done')
    param = page("/openaccount.htm", '- heading "Funded from 13344" [level=1]')
    calls.items[3] = calls.items[3].model_copy(update={"after": data})

    data_step = compiled(calls).capability.steps[3]
    calls.items[3] = calls.items[3].model_copy(update={"after": param})
    param_step = compiled(calls).capability.steps[3]

    assert data_step.expect[0].when == sig.VisibleCondition(text="Done")
    assert param_step.expect[0].when == sig.VisibleCondition(
        role="heading", name="Funded from {funding_account_id}"
    )


def test_a_commit_with_no_visible_change_is_flagged_for_the_reviewer() -> None:
    calls = Calls()
    the_flow(calls)
    calls.items[3] = calls.items[3].model_copy(update={"after": calls.items[3].before})

    result = compiled(calls)

    assert result.capability.steps[3].expect == []
    assert "L003" in result.report.codes()
    assert any("nothing visibly changed" in note for note in result.notes)


def test_the_draft_contract_comes_from_the_goal() -> None:
    calls = Calls()
    the_flow(calls)
    capability = compiled(calls).capability

    assert capability.id == GOAL.capability and capability.lifecycle.status == "draft"
    assert [i.name for i in capability.inputs] == ["account_type", "funding_account_id"]
    assert capability.inputs[1].sensitivity == "financial_id"
    assert [(o.name, o.source_step) for o in capability.outputs] == [
        ("new_account_id", capability.steps[4].id)
    ]
    assert capability.side_effects.cls == "commit"
    assert capability.side_effects.approval == "per_invocation"
    assert capability.app.entry_route == "/openaccount.htm"
    assert capability.provenance.discovery_run_id == "discovery-x"
    assert capability.provenance.recorded_against == "fixture"
    assert capability.provenance.locator_stats.model_dump() == {
        "targets": 4,
        "locators": 4,
        "verified_unique": 4,
    }
    assert isinstance(capability.steps[1].value, ParamValue)


def test_a_key_press_on_the_page_itself_is_dropped_unless_it_commits() -> None:
    calls = Calls()
    the_flow(calls)
    page_body = ElementRecord(description="the page", tag="body", target=None)
    calls.insert(2, "key", element=page_body, key="Escape", effect="input")

    result = compiled(calls)

    assert "press" not in [s.action for s in result.capability.steps]
    assert any("Escape press on the page itself" in note for note in result.notes)


def test_literal_query_values_stay_encoded() -> None:
    calls = Calls()
    calls.add(
        "navigate",
        url=f"{BASE_URL}/search.htm?q=a%26b%20c&id=13344",
        after=page("/search.htm", '- heading "Results" [level=1]'),
    )
    the_flow(calls, start_with_navigate=False)

    url = compiled(calls).capability.steps[0].url

    assert url == "{base_url}/search.htm?q=a%26b%20c&id={funding_account_id}"


# --- a person's steps (M6) -------------------------------------------------------------------


def test_a_persons_step_compiles_like_the_models_and_is_listed_for_review() -> None:
    calls = Calls()
    the_flow(calls)
    calls.items[1] = calls.items[1].model_copy(update={"actor": "human"})

    result = compiled(calls)

    human = result.capability.provenance.human_steps
    assert len(human) == 1
    step = next(s for s in result.capability.steps if s.id == human[0])
    assert (step.action, step.value) == ("select", ParamValue(param="account_type"))
    assert any("recorded from a person's actions" in n for n in result.notes)
    assert result.report.ok


def test_a_persons_click_on_an_element_that_was_gone_is_a_note_not_a_failure() -> None:
    calls = Calls()
    the_flow(calls)
    gone = element("button#x", tag="button", role="button").model_copy(
        update={"description": "button 'Close account 13344'", "target": None}
    )
    calls.insert(2, "left_click", element=gone, actor="human", effect="input")

    result = compiled(calls)

    assert result.capability.provenance.human_steps == []
    (note,) = [n for n in result.notes if "was not compiled" in n]
    assert "13344" not in note  # page text stays out of review.txt
    assert len(result.capability.steps) == 5


def test_a_persons_commit_that_was_not_sent_is_not_compiled() -> None:
    calls = Calls()
    the_flow(calls)
    blocked = calls.items[3].model_copy(
        update={"actor": "human", "ok": False, "approved": False, "error": "not sent"}
    )
    calls.items.insert(3, blocked)
    calls.items = [c.model_copy(update={"seq": i + 1}) for i, c in enumerate(calls.items)]
    calls.seq = len(calls.items)

    result = compiled(calls)

    assert [s.effect for s in result.capability.steps].count("commit") == 1
    assert result.capability.provenance.human_steps == []
