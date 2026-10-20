"""Playwright behaviours the design depends on; an upgrade that breaks one fails here first."""

import re

from playwright.sync_api import Page

FORM = """
<h1>Open New Account</h1>
<form>
  <label for="type">Type</label>
  <select id="type"><option>CHECKING</option><option>SAVINGS</option></select>
  <input type="button" value="Open New Account">
  <a href="#menu">Open New Account</a>
  <table><tr><td onclick="void 0" style="cursor:pointer">legacy clickable cell</td></tr></table>
  <input type="password" id="pw" value="hunter2">
</form>
"""


def _ref(snapshot: str, line_contains: str) -> str:
    for line in snapshot.splitlines():
        match = re.search(r"\[ref=(e\d+)\]", line)
        if match and line_contains in line:
            return match.group(1)
    raise AssertionError(f"no ref on a line containing {line_contains!r}:\n{snapshot}")


def test_ai_snapshot_refs_resolve_to_the_exact_element(page: Page) -> None:
    page.set_content(FORM)
    snapshot = page.locator("body").aria_snapshot(mode="ai")

    button = page.get_by_ref(_ref(snapshot, 'button "Open New Account"'))
    link = page.get_by_ref(_ref(snapshot, 'link "Open New Account"'))

    assert button.evaluate("e => e.type") == "button"
    assert link.evaluate("e => e.tagName") == "A"


def test_same_name_targets_are_disambiguated_by_role(page: Page) -> None:
    page.set_content(FORM)

    # Heading, button and nav link all read "Open New Account"; only role+name is unique.
    assert page.get_by_text("Open New Account", exact=True).count() == 3
    assert page.get_by_role("button", name="Open New Account", exact=True).count() == 1
    assert page.get_by_role("link", name="Open New Account", exact=True).count() == 1


def test_non_semantic_clickable_gets_a_ref(page: Page) -> None:
    page.set_content(FORM)
    snapshot = page.locator("body").aria_snapshot(mode="ai")

    cell = page.get_by_ref(_ref(snapshot, "legacy clickable cell"))
    assert cell.evaluate("e => e.tagName") == "TD"


def test_label_association_names_the_select(page: Page) -> None:
    page.set_content(FORM)

    assert page.get_by_label("Type").evaluate("e => e.id") == "type"
    assert page.get_by_role("combobox", name="Type").count() == 1


def test_screenshot_masking_is_available(page: Page) -> None:
    page.set_content(FORM)

    masked = page.screenshot(mask=[page.locator("#pw")])
    assert masked.startswith(b"\x89PNG")
