"""Evidence capture and the leak scan: nothing sensitive reaches the run directory."""

from __future__ import annotations

import base64
import re
from pathlib import Path

from playwright.sync_api import BrowserContext, Page

from tests.conftest import sign_in
from tests.fixtures.site import BASE_URL, PASSWORD, SSN, USERNAME, FixtureSite
from understudy.artifact.schema import Extraction
from understudy.evidence.run import RunContext
from understudy.redaction.redactor import Redactor
from understudy.surface import capture
from understudy.surface.web import WebSurface

MAGENTA = [255, 0, 255]  # Playwright's default mask colour


def pixel(context: BrowserContext, png: Path, x: float, y: float) -> list[int]:
    viewer = context.new_page()
    viewer.set_content("<canvas id='c'></canvas>")
    data = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
    rgba: list[int] = viewer.evaluate(
        """async ([src, x, y]) => {
          const img = new Image(); img.src = src; await img.decode();
          const c = document.getElementById("c"); c.width = img.width; c.height = img.height;
          const g = c.getContext("2d"); g.drawImage(img, 0, 0);
          return Array.from(g.getImageData(x, y, 1, 1).data);
        }""",
        [data, int(x), int(y)],
    )
    viewer.close()
    return rgba[:3]


def test_password_and_pii_fields_are_painted_over(
    page: Page, context: BrowserContext, surface: WebSurface, tmp_path: Path
) -> None:
    surface.navigate(f"{BASE_URL}/index.htm")
    page.fill("input[name='username']", USERNAME)
    page.fill("input[name='password']", PASSWORD)
    assert len(capture.sensitive_fields(page)) == 2

    shot = surface.screenshot(tmp_path / "login.png")
    password = page.locator("input[name='password']").bounding_box()
    heading = page.get_by_role("heading", name="Customer Login").bounding_box()
    assert password is not None and heading is not None
    centre = (password["x"] + password["width"] / 2, password["y"] + password["height"] / 2)
    assert pixel(context, shot, *centre) == MAGENTA
    assert pixel(context, shot, heading["x"] + 2, heading["y"] + 2) != MAGENTA


def test_the_open_account_form_has_no_field_needing_a_mask(page: Page, surface: WebSurface) -> None:
    sign_in(page)
    surface.navigate(f"{BASE_URL}/openaccount.htm")
    assert capture.sensitive_fields(page) == []


def test_extra_masks_cover_app_specific_panels(
    page: Page, context: BrowserContext, surface: WebSurface, tmp_path: Path
) -> None:
    sign_in(page)
    surface.navigate(f"{BASE_URL}/overview.htm")
    shot = surface.screenshot(tmp_path / "overview.png", extra_masks=["#accountTable"])
    table = page.locator("#accountTable").bounding_box()
    assert table is not None
    assert pixel(context, shot, table["x"] + 5, table["y"] + 5) == MAGENTA


def test_the_sanitized_dom_drops_values_session_ids_and_identifiers(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    surface.navigate(f"{BASE_URL}/index.htm")
    page.fill("input[name='password']", PASSWORD)
    # A server-rendered value lives in the attribute, not just the property.
    page.locator("input[name='username']").evaluate("e => e.setAttribute('value', 'prefill-u')")
    dom = capture.sanitized_dom(page, Redactor(b"k"))
    assert PASSWORD not in dom and "prefill-u" not in dom
    assert site.session_id not in dom and "jsessionid" not in dom.lower()
    assert 'action="login.htm"' in dom

    sign_in(page)
    surface.navigate(f"{BASE_URL}/overview.htm")
    dom = capture.sanitized_dom(page, Redactor(b"k"))
    assert SSN not in dom and "13344" not in dom
    assert "Accounts Overview" in dom


def test_a_failure_bundle_has_everything_and_leaks_nothing(
    page: Page, surface: WebSurface, site: FixtureSite, tmp_path: Path
) -> None:
    surface.navigate(f"{BASE_URL}/index.htm")
    page.fill("input[name='password']", PASSWORD)
    page.evaluate("() => console.error('login failed for 123-45-6789')")
    bundle = surface.failure_bundle(
        tmp_path / "failure", Redactor(b"k"), reason=f"step s1 failed near {SSN}"
    )
    names = sorted(p.name for p in bundle.iterdir())
    assert names == [
        "aria.txt",
        "bundle.json",
        "console.txt",
        "dom.html",
        "events.jsonl",
        "screenshot.png",
    ]
    assert "login failed for [PII:" in (bundle / "console.txt").read_text()
    assert leaks(bundle, site) == []


def test_leak_scan_over_a_whole_run(
    page: Page, surface: WebSurface, site: FixtureSite, tmp_path: Path
) -> None:
    """Plan §11: grep every file a run writes for the password, the session id and
    the SSN pattern; the count must be 0. The account numbers the run handled are
    checked whole, and every mask is checked to reveal at most one digit."""
    run = RunContext(tmp_path, kind="replay", redactor=Redactor(b"k"), meta={"tenant": "fixture"})
    run.know({"13344": "financial_id"})

    surface.navigate(f"{BASE_URL}/index.htm")
    first = surface.observe()
    run.event("observe", actor="engine", url=first.url)
    (run.subdir("snapshots") / "index.txt").write_text(run.redactor.snapshot(first.snapshot))
    sign_in(page)

    surface.navigate(f"{BASE_URL}/overview.htm")
    overview = surface.observe()
    run.event("observe", actor="engine", url=overview.url)
    (run.dir / "snapshots" / "overview.txt").write_text(run.redactor.snapshot(overview.snapshot))
    surface.screenshot(run.screenshot_path("overview"))

    surface.navigate(f"{BASE_URL}/openaccount.htm")
    page.wait_for_function("document.querySelector('#fromAccountId').options.length > 0")
    surface.act("select", page.locator("select#fromAccountId"), "13344")
    run.event("act", actor="engine", step_id="s3", value="13344", value_sensitivity="financial_id")
    with surface.commit_window():
        surface.act("click", page.locator("input[type='button'][value='Open New Account']"))
        page.get_by_text("Account Opened!").wait_for()
    new_id = surface.read(page.locator("#newAccountId"), Extraction(output="new_account_id"))
    run.know({new_id: "financial_id"})
    run.event("act", actor="engine", step_id="s5", value=new_id, value_sensitivity="financial_id")
    run.write_json("result.json", {"status": "succeeded", "outputs": {"new_account_id": new_id}})

    surface.navigate(f"{BASE_URL}/overview.htm")
    surface.failure_bundle(run.subdir("failure"), run.redactor, reason="probe", events=run.recent())

    assert leaks(run.dir, site, extra=[new_id]) == []
    assert (run.dir / "result.json").read_text().count("••••") == 1
    events = (run.dir / "events.jsonl").read_text()
    assert "••••4" in events and new_id[-1] in events


def leaks(directory: Path, site: FixtureSite, extra: list[str] | None = None) -> list[str]:
    needles = [PASSWORD, site.session_id, "13344", *(extra or [])]
    found = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix == ".png":
            continue
        text = path.read_text(encoding="utf-8")
        found += [f"{path.name}: {n}" for n in needles if n in text]
        found += [f"{path.name}: ssn" for _ in re.findall(r"\b\d{3}-\d{2}-\d{4}\b", text)]
        # A mask may keep one digit of a 5-digit account id; more would be most of it.
        found += [f"{path.name}: {m}" for m in re.findall(r"•{4}\d{2,}", text)]
        if "jsessionid" in text.lower():
            found.append(f"{path.name}: jsessionid")
    return found


def test_masked_panel_text_is_hidden_as_whole_values_only() -> None:
    from understudy.surface.capture import mask_snapshot

    snapshot = (
        '- paragraph: Welcome John Smith\n- columnheader "Account"\n'
        '- combobox [ref=e3]\n- link "Open New Account"\n- text: fromAccountId'
    )
    masked = mask_snapshot(snapshot, {"Welcome John Smith", "Account"})
    assert "John Smith" not in masked
    assert '- columnheader "[masked]"' in masked
    assert '- link "Open New Account"' in masked  # a whole value elsewhere is untouched
    assert "fromAccountId" in masked
