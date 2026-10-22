"""WebSurface against the routed fixture site: refs, actions, probes, commit gating."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page

from tests.conftest import REPO, sign_in
from tests.fixtures.example_capability import example_capability
from tests.fixtures.site import BASE_URL, PASSWORD, FixtureSite
from understudy.artifact.schema import Extraction
from understudy.artifact.signatures import (
    AbsentCondition,
    AllCondition,
    DialogCondition,
    HttpStatusCondition,
    OptionsLoadedCondition,
    SelectedCondition,
    Signature,
    UrlCondition,
    VisibleCondition,
)
from understudy.policy.gate import PolicyGate
from understudy.profile import load_profile
from understudy.surface.base import (
    ActionBlocked,
    ActionFailed,
    DialogBlocking,
    OptionAbsent,
    PolicyBlocked,
    ProbeError,
    StaleRef,
    TargetChanged,
)
from understudy.surface.web import WebSurface

CREATE = "/parabank/services_proxy/bank/createAccount"
SUBMIT = "input[type='button'][value='Open New Account']"


def ref_for(snapshot: str, contains: str) -> str:
    for line in snapshot.splitlines():
        match = re.search(r"\[ref=(ref_\d+)\]", line)
        if match and contains in line:
            return match.group(1)
    raise AssertionError(f"no ref on a line containing {contains!r}:\n{snapshot}")


def open_form(page: Page, surface: WebSurface) -> None:
    sign_in(page)
    surface.navigate(f"{BASE_URL}/openaccount.htm")
    page.wait_for_function("document.querySelector('#fromAccountId').options.length > 0")


PROFILE = load_profile(REPO / "apps/parabank/profile.yaml")


def global_state(state_id: str) -> Signature:
    return next(s.when for s in PROFILE.global_states if s.id == state_id)


# --- the fixture site behaves like recon found ParaBank ------------------------------


def test_the_model_never_sees_the_session_id_or_a_typed_password(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    surface.navigate(f"{BASE_URL}/index.htm")
    # First visit: Tomcat-style URL rewriting puts the session id in every link.
    assert site.session_id in page.content()
    page.fill("input[name='password']", PASSWORD)
    snapshot = surface.observe().snapshot
    assert site.session_id not in snapshot and "jsessionid" not in snapshot
    assert PASSWORD not in snapshot
    assert 'link "Forgot login info?"' in snapshot


def test_an_expired_session_rerenders_the_login_form_in_place(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    sign_in(page)
    surface.navigate(f"{BASE_URL}/overview.htm")
    expired = global_state("session_expired")
    assert not surface.probe(expired)

    page.context.clear_cookies()
    surface.navigate(f"{BASE_URL}/overview.htm")
    assert page.url == f"{BASE_URL}/overview.htm"  # no redirect: the login form re-renders in place
    assert surface.probe(expired)


def test_a_bad_account_id_is_a_business_error(page: Page, surface: WebSurface) -> None:
    sign_in(page)
    surface.navigate(f"{BASE_URL}/activity.htm?id=99999")
    # Recorded under the profile's recon notes: a business answer on this page.
    not_found = AllCondition(
        all=[
            VisibleCondition(role="heading", name="Error!"),
            VisibleCondition(text_regex=r"Could not find account #\d+"),
        ]
    )
    assert surface.probe(not_found)
    assert not surface.probe(global_state("app_internal_error"))


# --- refs ----------------------------------------------------------------------------


def test_a_ref_resolves_to_the_element_it_named(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    snapshot = surface.observe().snapshot
    button = surface.element_for_ref(ref_for(snapshot, 'button "Open New Account"'))
    link = surface.element_for_ref(ref_for(snapshot, 'link "Open New Account"'))
    assert button.evaluate("e => e.type") == "button"
    assert link.evaluate("e => e.tagName") == "A"


def test_reading_facts_does_not_strand_the_models_refs(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    snapshot = surface.observe().snapshot
    heading = ref_for(snapshot, 'heading "Open New Account"')
    # An element-level snapshot replaces Playwright's page-level one.
    surface.facts(page.locator("select#type"))
    assert surface.element_for_ref(heading).evaluate("e => e.tagName") == "H1"


def test_acting_through_a_ref_survives_the_policy_check(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    account_type = surface.element_for_ref(ref_for(surface.observe().snapshot, "combobox"))
    # act() reads the element's facts for the gate before it selects.
    surface.act("select", account_type, "SAVINGS")
    assert page.locator("select#type").input_value() == "1"


def test_a_ref_from_before_a_navigation_is_stale_at_once(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    ref = ref_for(surface.observe().snapshot, 'button "Open New Account"')
    surface.navigate(f"{BASE_URL}/overview.htm")
    surface.observe()
    with pytest.raises(StaleRef):
        surface.element_for_ref(ref)


def test_refs_reach_into_frames(page: Page, surface: WebSurface) -> None:
    surface.navigate(f"{BASE_URL}/legacy/frameset.htm")
    page.frame_locator("frame[name='main']").locator("h2").first.wait_for()
    snapshot = surface.observe().snapshot
    payee = surface.element_for_ref(ref_for(snapshot, 'button "Send Payment"'))
    assert payee.evaluate("e => e.value") == "Send Payment"
    assert surface.facts(payee).frame_path == ('frame[name="main"]',)


# --- actions -------------------------------------------------------------------------


def test_select_by_label_then_by_value(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    account_type = page.locator("select#type")
    surface.act("select", account_type, "SAVINGS")
    assert account_type.input_value() == "1"  # labelled SAVINGS, valued 1
    funding = page.locator("select#fromAccountId")
    surface.act("select", funding, "13344")
    assert funding.input_value() == "13344"


def test_selecting_an_absent_option_reports_how_many_were_loaded(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    open_form(page, surface)
    with pytest.raises(OptionAbsent) as caught:
        surface.act("select", page.locator("select#fromAccountId"), "99999")
    assert caught.value.options_loaded == len(site.accounts)


def test_select_needs_a_select(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    with pytest.raises(ActionFailed, match="<select>"):
        surface.act("select", page.locator(SUBMIT), "x")


def test_a_commit_needs_an_open_commit_window(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    open_form(page, surface)
    submit = page.locator(SUBMIT)
    with pytest.raises(PolicyBlocked) as caught:
        surface.act("click", submit)
    assert caught.value.rule == "commit"
    assert site.reached(CREATE) == 0

    with surface.commit_window():
        surface.act("click", submit)
        page.get_by_text("Account Opened!").wait_for()
    assert site.reached(CREATE) == 1
    new_id = surface.read(page.locator("#newAccountId"), Extraction(output="new_account_id"))
    assert new_id == "13566"


def test_a_commit_request_from_page_script_is_stopped_at_the_network(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    open_form(page, surface)
    status = page.evaluate(
        "() => fetch('services_proxy/bank/createAccount?customerId=12212&newAccountType=0"
        "&fromAccountId=12345', {method: 'POST'}).then(r => r.status, () => 'blocked')"
    )
    assert status == "blocked"
    assert site.reached(CREATE) == 0
    assert [v.rule for v in surface.violations] == ["commit_window"]


def test_the_nav_link_sharing_the_commit_buttons_name_is_not_a_commit(
    page: Page, surface: WebSurface
) -> None:
    open_form(page, surface)
    surface.act("click", page.get_by_role("link", name="Open New Account"))
    page.wait_for_load_state("load")
    assert page.url.endswith("/openaccount.htm")


def test_an_overlay_over_the_target_is_reported_as_blocked(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    page.evaluate(
        """() => { const d = document.createElement('div');
        d.style = 'position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.3)';
        d.textContent = 'Please wait'; document.body.appendChild(d); }"""
    )
    with pytest.raises(ActionBlocked):
        surface.act("click", page.get_by_role("link", name="Accounts Overview"), timeout_ms=1_000)


def test_a_covered_select_is_refused_too_not_only_a_covered_click(
    page: Page, surface: WebSurface
) -> None:
    """Playwright checks what receives the pointer only for clicks: select_option and
    fill go straight through a modal. The surface refuses them as well."""
    open_form(page, surface)
    page.evaluate(
        """() => { const d = document.createElement('div'); d.id = 'cover';
        d.style = 'position:fixed;inset:0;z-index:9999'; document.body.appendChild(d); }"""
    )
    with pytest.raises(ActionBlocked, match="div#cover") as raised:
        surface.act("select", page.locator("select#fromAccountId"), "54321")
    assert not raised.value.dispatched
    assert page.locator("select#fromAccountId").input_value() == "12345"


def test_a_shadow_root_input_or_one_under_its_own_label_is_not_covered(
    page: Page, surface: WebSurface
) -> None:
    page.set_content(
        """<div style="position:relative">
          <input id="email" style="width:200px;height:24px">
          <label for="email" style="position:absolute;left:0;top:0;width:200px;height:24px">
            <span>Email</span></label>
        </div>
        <my-field></my-field>
        <script>customElements.define("my-field", class extends HTMLElement {
          constructor() {
            super(); this.attachShadow({mode: "open"}).innerHTML = "<input id=inner>";
          }
        });</script>"""
    )
    surface.act("fill", page.locator("#email"), "a")
    surface.act("fill", page.locator("my-field #inner"), "b")
    assert page.locator("#email").input_value() == "a"
    assert page.locator("my-field #inner").input_value() == "b"


def test_an_error_from_the_browser_call_itself_is_marked_dispatched(
    page: Page, surface: WebSurface
) -> None:
    open_form(page, surface)
    with pytest.raises(ActionFailed) as raised:
        surface.act("fill", page.get_by_role("link", name="Accounts Overview"), "x", timeout_ms=500)
    assert raised.value.dispatched


# --- probes --------------------------------------------------------------------------


def test_the_success_checkpoint_is_false_before_and_true_after_the_commit(
    page: Page, surface: WebSurface
) -> None:
    success = example_capability().success
    assert success.negative_control is not None
    open_form(page, surface)
    # Both success texts are in the DOM already, hidden: presence would pass here.
    assert page.locator("text=Account Opened!").count() == 1
    assert not surface.probe(success.signature)
    assert surface.probe(success.negative_control)
    assert surface.probe(AbsentCondition(text="Account Opened!"))

    with surface.commit_window():
        surface.act("click", page.locator(SUBMIT))
        page.get_by_text("Account Opened!").wait_for()
    assert surface.probe(success.signature)
    assert not surface.probe(success.negative_control)


def test_url_conditions_match_the_route_without_session_ids(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    surface.navigate(f"{BASE_URL}/index.htm;jsessionid={site.session_id}")
    assert surface.probe(UrlCondition(path_regex=r"^/index\.htm$"))
    assert not surface.probe(UrlCondition(path_regex=r"^/parabank/"))


def test_options_loaded_counts_real_options_on_a_unique_select(
    page: Page, surface: WebSurface
) -> None:
    open_form(page, surface)
    assert surface.probe(OptionsLoadedCondition(target="select#fromAccountId", min=4))
    assert not surface.probe(OptionsLoadedCondition(target="select#fromAccountId", min=50))
    assert not surface.probe(OptionsLoadedCondition(target="select", min=1))
    assert not surface.probe(OptionsLoadedCondition(target="select#nope", min=1))


def test_options_loaded_is_false_until_the_list_arrives(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    site.accounts_delay_ms = 1_500  # wide enough that the first probe cannot lose the race
    sign_in(page)
    surface.navigate(f"{BASE_URL}/openaccount.htm")
    loaded = OptionsLoadedCondition(target="select#fromAccountId", min=1)
    assert not surface.probe(loaded)
    page.wait_for_function("document.querySelector('#fromAccountId').options.length > 0")
    assert surface.probe(loaded)


def test_selected_matches_label_value_or_param(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    surface.act("select", page.locator("select#type"), "SAVINGS")
    assert surface.probe(SelectedCondition(target="select#type", value="SAVINGS"))
    assert surface.probe(SelectedCondition(target="select#type", value="1"))
    assert surface.probe(SelectedCondition(target="select#type", param="t"), {"t": "SAVINGS"})
    assert not surface.probe(SelectedCondition(target="select#type", value="CHECKING"))


def test_visible_matchers_are_anded_and_templated(page: Page, surface: WebSurface) -> None:
    sign_in(page)
    surface.navigate(f"{BASE_URL}/overview.htm")
    row = VisibleCondition(role="link", name="{acct}", within="#accountTable")
    assert surface.probe(row, {"acct": "13344"})
    assert not surface.probe(row, {"acct": "99999"})
    assert surface.probe(VisibleCondition(role="heading", text="Accounts Overview"))
    assert not surface.probe(VisibleCondition(role="link", css="h1"))
    assert surface.probe(VisibleCondition(text_regex=r"Could not|Accounts Over\w+"))


def test_http_status_tracks_the_main_frame(page: Page, surface: WebSurface) -> None:
    surface.navigate(f"{BASE_URL}/index.htm")
    assert surface.probe(HttpStatusCondition(min=200, max=299))
    surface.navigate(f"{BASE_URL}/missing.htm")
    assert surface.probe(HttpStatusCondition(min=400, max=499))


def test_an_unknown_dialog_is_dismissed_at_once_and_blocks_driving_until_acknowledged(
    page: Page, surface: WebSurface
) -> None:
    """A dialog left open freezes every page read, whatever its timeout, so the surface
    answers it on arrival (dismiss: the non-committal answer) and records it."""
    surface.navigate(f"{BASE_URL}/index.htm")
    page.evaluate("() => setTimeout(() => alert('Your session will expire'), 0)")
    page.wait_for_event("dialog")
    surface.pause(50)
    assert surface.probe(DialogCondition(message_regex="session will expire"))
    assert surface.pending_dialog == ("alert", "Your session will expire")
    assert surface.observe().dialog == "alert: Your session will expire (dismissed)"
    # The page is readable (nothing is open), but it may not be driven yet.
    assert surface.probe(VisibleCondition(role="heading", name="Customer Login"))
    with pytest.raises(DialogBlocking):
        surface.resolve(example_capability().steps[3].target, {})  # type: ignore[arg-type]
    assert [str(d) for d in surface.acknowledge_dialogs()] == ["alert: Your session will expire"]
    assert surface.pending_dialog is None
    assert not surface.probe(DialogCondition())


def test_a_dialog_with_a_rule_gets_that_answer_and_does_not_block(
    page: Page, site: FixtureSite, gate: PolicyGate
) -> None:
    surface = WebSurface(page, gate, mode="replay", dialog_rules=[("^Continue\\?", "accept")])
    surface.navigate(f"{BASE_URL}/index.htm")
    answer = page.evaluate("() => new Promise((r) => setTimeout(() => r(confirm('Continue?')), 0))")
    assert answer is True
    assert surface.pending_dialog is None  # known: it does not block
    assert [(d.answer, d.known) for d in surface.dialogs] == [("accept", True)]
    unknown = page.evaluate("() => new Promise((r) => setTimeout(() => r(confirm('Delete?')), 0))")
    assert unknown is False
    assert surface.pending_dialog == ("confirm", "Delete?")


# --- regressions from a code review ----------------------------------------------------------


def test_a_re_render_between_resolve_and_act_is_refused_not_clicked(
    page: Page, surface: WebSurface
) -> None:
    open_form(page, surface)
    target = example_capability().steps[3].target
    assert target is not None
    resolution = surface.resolve(target.model_copy(update={"locators": target.locators[1:]}), {})
    assert resolution.element is not None
    # The page swaps the button for a look-alike that the same locator still matches.
    page.evaluate(
        """() => { const old = document.querySelector("input.button");
        const fake = document.createElement("input"); fake.type = "button";
        fake.className = "button"; fake.value = "Open New Account";
        fake.setAttribute("aria-label", "Delete everything");
        fake.onclick = () => { window.clicked = true; }; old.replaceWith(fake); }"""
    )
    with pytest.raises(TargetChanged, match="Delete everything"):
        surface.act("click", resolution.element, fingerprint=target.fingerprint, timeout_ms=1_000)
    assert page.evaluate("() => window.clicked === undefined")


@pytest.mark.parametrize("value", ["x'], h1[class='title", "O'Brien"])
def test_a_probe_param_that_would_break_out_of_a_selector_is_refused(
    page: Page, surface: WebSurface, value: str
) -> None:
    sign_in(page)
    surface.navigate(f"{BASE_URL}/overview.htm")
    condition = VisibleCondition(css="#accountId[title='{acct}']")
    with pytest.raises(ProbeError, match="cannot go in a selector"):
        surface.probe(condition, {"acct": value})


def test_one_commit_window_allows_one_commit(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    open_form(page, surface)
    submit = page.locator(SUBMIT)
    with surface.commit_window():
        surface.act("click", submit)
        page.get_by_text("Account Opened!").wait_for()
        status = page.evaluate(
            "() => fetch('services_proxy/bank/createAccount?customerId=12212&newAccountType=0"
            "&fromAccountId=12345', {method: 'POST'}).then(r => r.status, () => 'blocked')"
        )
        assert status == "blocked"
        with pytest.raises(PolicyBlocked, match="no approved commit is left"):
            surface.act("click", submit, timeout_ms=500)
    assert site.reached(CREATE) == 1
