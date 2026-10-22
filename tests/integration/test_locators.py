"""Locator synthesis and resolution against the fixture site."""

from __future__ import annotations

from typing import Any

import pytest
from playwright.sync_api import Page

from tests.conftest import sign_in, surface_test_gate
from tests.fixtures.example_capability import example_capability
from tests.fixtures.site import BASE_URL, FixtureSite
from understudy.artifact.lint import lint
from understudy.artifact.schema import Fingerprint, Locator, Target, Verification
from understudy.policy.gate import PolicyGate
from understudy.surface.web import WebSurface

SUBMIT = "input[type='button'][value='Open New Account']"
VERIFIED = Verification(unique=True, same_element=True, run_id="t")


@pytest.fixture
def gate() -> PolicyGate:
    """This module drives fixture-only pages, so it adds them to the allowlist."""
    return surface_test_gate()


def open_form(page: Page, surface: WebSurface) -> None:
    sign_in(page)
    surface.navigate(f"{BASE_URL}/openaccount.htm")
    page.wait_for_function("document.querySelector('#fromAccountId').options.length > 0")


def kinds(target: Target | None) -> list[tuple[str, Any]]:
    assert target is not None
    return [(loc.by, loc.value) for loc in target.locators]


def loc(by: str, value: Any, **kw: Any) -> Locator:
    return Locator(by=by, value=value, stability="medium", note="t", verified=VERIFIED, **kw)  # type: ignore[arg-type]


# --- synthesis -------------------------------------------------------------------------


def test_the_submit_button_gets_role_first_and_never_a_name_only_locator(
    page: Page, surface: WebSurface
) -> None:
    open_form(page, surface)
    result = surface.synthesize(page.locator(SUBMIT))
    assert kinds(result.target) == [
        ("role", {"role": "button", "name": "Open New Account", "exact": True}),
        ("css", SUBMIT),
        (
            "css",
            "#openAccountForm > form:nth-of-type(1) > div:nth-of-type(1) > input:nth-of-type(1)",
        ),
    ]
    assert result.target is not None
    assert [loc.stability for loc in result.target.locators] == ["high", "medium", "low"]
    assert all(loc.verified and loc.verified.run_id == "test-run" for loc in result.target.locators)
    assert result.target.fingerprint == Fingerprint(
        role="button", name="Open New Account", tag="input", input_type="button"
    )


def test_unnamed_selects_are_anchored_on_their_neighbouring_text(
    page: Page, surface: WebSurface
) -> None:
    """ParaBank labels these selects with sibling <p> text, so there is no
    role+name or label locator; the derived text anchor is their only fallback."""
    open_form(page, surface)
    account_type = surface.synthesize(page.locator("select#type")).target
    funding = surface.synthesize(page.locator("select#fromAccountId")).target
    assert account_type is not None and funding is not None

    assert [loc.by for loc in account_type.locators] == ["css", "xpath", "css"]
    assert account_type.locators[0].value == "select#type"
    assert account_type.locators[1].value == (
        "//p[normalize-space(.)='What type of Account would you like to open?']"
        "/following-sibling::select[1]"
    )
    assert account_type.locators[1].stability == "medium"
    assert account_type.fingerprint == Fingerprint(role="combobox", tag="select")

    # The funding paragraph is long, so the anchor is a prefix match and low stability.
    assert funding.locators[1].by == "xpath"
    assert "starts-with(normalize-space(.), 'A minimum of $100.00" in str(funding.locators[1].value)
    assert funding.locators[1].stability == "low"


def test_a_row_link_is_templated_on_its_param_and_replays_for_another_row(
    page: Page, surface: WebSurface
) -> None:
    sign_in(page)
    surface.navigate(f"{BASE_URL}/overview.htm")
    result = surface.synthesize(
        page.get_by_role("link", name="13344"), params={"funding_account_id": "13344"}
    )
    target = result.target
    assert target is not None
    role, *rest = target.locators
    assert role.value == {"role": "link", "name": "{funding_account_id}", "exact": True}
    assert role.templated and role.stability == "medium"
    table = next(loc for loc in rest if loc.by == "table_cell")
    assert table.value == {
        "table": "table#accountTable",
        "column": "Account",
        "key_column": "Account",
        "row_key": "{funding_account_id}",
        "inner": "a",
    }
    assert target.fingerprint.name == "{funding_account_id}"

    other = surface.resolve(target, {"funding_account_id": "12456"})
    assert other.ok and other.locator_index == 0
    assert other.element is not None and other.element.inner_text() == "12456"

    only_by_table = target.model_copy(update={"locators": [table]})
    via_table = surface.resolve(only_by_table, {"funding_account_id": "54321"})
    assert via_table.ok and via_table.element is not None
    assert via_table.element.inner_text() == "54321"


def test_legacy_frame_fields_are_found_by_name_and_by_their_label_cell(
    page: Page, surface: WebSurface
) -> None:
    surface.navigate(f"{BASE_URL}/legacy/frameset.htm")
    main = page.frame_locator("frame[name='main']")
    main.locator("h2").first.wait_for()

    payee = surface.synthesize(main.locator("input[name='payee.name']")).target
    assert payee is not None
    assert payee.frame_path == ['frame[name="main"]']
    assert kinds(payee)[:2] == [
        ("css", "input[name='payee.name']"),
        ("xpath", "//td[normalize-space(.)='Payee Name:']/following-sibling::td[1]//input"),
    ]

    # A generated-looking id is not offered; the label cell carries the field.
    account = surface.synthesize(main.locator("#acct7731x")).target
    assert account is not None
    assert all("acct7731x" not in str(loc.value) for loc in account.locators)
    assert account.locators[0].value == (
        "//td[normalize-space(.)='Account #:']/following-sibling::td[1]//input"
    )


def test_repeated_row_links_fall_back_to_the_table_cell(page: Page, surface: WebSurface) -> None:
    surface.navigate(f"{BASE_URL}/legacy/frameset.htm")
    main = page.frame_locator("frame[name='main']")
    main.locator("h2").first.wait_for()
    result = surface.synthesize(main.locator("a[href='#t2']"))
    target = result.target
    assert target is not None
    assert ("role", "matched 3 elements, not 1") in result.rejected
    assert ("text", "matched 3 elements, not 1") in result.rejected
    # The action column has no header, so the cell is found by its position.
    assert target.locators[0].by == "table_cell"
    assert target.locators[0].value == {
        "table": "table#transactionTable",
        "column": 3,
        "key_column": "Date",
        "row_key": "10-02-2026",
        "inner": "a",
    }
    assert target.locators[0].stability == "low"
    resolved = surface.resolve(target.model_copy(update={"locators": target.locators[:1]}), {})
    assert resolved.ok and resolved.element is not None
    assert resolved.element.get_attribute("href") == "#t2"


def test_nothing_unverified_ever_reaches_the_target(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    result = surface.synthesize(page.get_by_role("link", name="Open New Account"))
    assert result.target is not None
    for candidate in result.target.locators:
        built = surface.resolve(result.target.model_copy(update={"locators": [candidate]}), {})
        assert built.ok, (candidate, built.attempts)


# --- resolution ------------------------------------------------------------------------


def test_every_locator_in_the_example_artifact_resolves_on_its_own(
    page: Page, surface: WebSurface
) -> None:
    """Each fallback is tried alone, so a locator that is wrong but hidden behind a
    working primary (as `input[type='submit']` was) cannot pass unnoticed."""
    capability = example_capability()
    open_form(page, surface)
    params = {"account_type": "SAVINGS", "funding_account_id": "13344"}
    for step in capability.steps:
        if step.target is None or step.action == "extract":
            continue
        for candidate in step.target.locators:
            alone = step.target.model_copy(update={"locators": [candidate]})
            resolution = surface.resolve(alone, params)
            assert resolution.ok, (step.id, candidate.value, resolution.attempts)

    # The control: the first-draft locator this test exists to catch does not resolve.
    s4 = capability.steps[3].target
    assert s4 is not None
    wrong = loc("css", "input[type='submit'][value='Open New Account']")
    assert not surface.resolve(s4.model_copy(update={"locators": [wrong]}), params).ok


def test_a_fallback_win_is_drift(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    target = Target(
        description="submit",
        fingerprint=Fingerprint(role="button", name="Open New Account"),
        locators=[loc("css", "#renamed-button"), loc("css", SUBMIT)],
    )
    resolution = surface.resolve(target, {})
    assert resolution.ok and resolution.drifted and resolution.locator_index == 1
    assert [a.outcome for a in resolution.attempts] == ["no_match", "matched"]


def test_a_look_alike_is_refused_not_clicked(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    open_form(page, surface)
    page.evaluate(f"() => document.querySelector(\"{SUBMIT}\").value = 'Open Account'")
    target = example_capability().steps[3].target
    assert target is not None
    loose = target.model_copy(update={"locators": [*target.locators, loc("css", "input.button")]})
    resolution = surface.resolve(loose, {})
    assert not resolution.ok
    assert [a.outcome for a in resolution.attempts] == [
        "no_match",
        "no_match",
        "fingerprint_mismatch",
    ]
    assert (
        "name: expected 'Open New Account', found 'Open Account'" in resolution.attempts[2].detail
    )
    assert site.reached("/parabank/services_proxy/bank/createAccount") == 0


def test_ambiguous_and_hidden_candidates_are_skipped(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    target = Target(
        description="new account id",
        fingerprint=Fingerprint(tag="a"),
        locators=[loc("css", "a"), loc("css", "#newAccountId")],
    )
    resolution = surface.resolve(target, {})
    assert [a.outcome for a in resolution.attempts] == ["ambiguous", "hidden"]
    assert surface.resolve(target, {}, require_visible=False).locator_index == 1


def test_a_scope_must_be_unique(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    target = Target(
        description="type",
        fingerprint=Fingerprint(tag="select"),
        scope=loc("css", "div"),
        locators=[loc("css", "select#type")],
    )
    resolution = surface.resolve(target, {})
    assert not resolution.ok
    assert resolution.attempts[0].by == "scope:css"
    assert resolution.attempts[0].outcome == "ambiguous"
    scoped = target.model_copy(update={"scope": loc("css", "#openAccountForm")})
    assert surface.resolve(scoped, {}).ok


@pytest.mark.parametrize("value", ["1' or '1'='1", "x]", 'a"b'])
def test_a_param_that_would_break_out_of_a_selector_is_refused(
    page: Page, surface: WebSurface, value: str
) -> None:
    open_form(page, surface)
    target = Target(
        description="row",
        fingerprint=Fingerprint(tag="a"),
        locators=[loc("xpath", "//a[text()='{acct}']", templated=True)],
    )
    resolution = surface.resolve(target, {"acct": value})
    assert resolution.attempts[0].outcome == "error"
    assert "cannot go in a xpath selector" in resolution.attempts[0].detail


def test_synthesized_targets_pass_lint_rules_on_locators(page: Page, surface: WebSurface) -> None:
    open_form(page, surface)
    capability = example_capability()
    synthesized = surface.synthesize(page.locator(SUBMIT)).target
    capability.steps[3] = capability.steps[3].model_copy(update={"target": synthesized})
    report = lint(capability)
    assert not ({"L005", "L010"} & report.codes()), [str(f) for f in report.findings]
