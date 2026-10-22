"""ParaBank reconnaissance: no LLM, read-only, logs in and records AI-mode
snapshots and screenshots of the pages the discovery/replay seams depend on.

Usage:
    uv run python scripts/recon.py local  --base-url http://localhost:8089/parabank
    uv run python scripts/recon.py public --base-url https://parabank.parasoft.com/parabank
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT_ROOT = Path(__file__).resolve().parent.parent / "runs" / "recon"


@dataclass
class Finding:
    page: str
    url: str
    status: int | None
    ai_snapshot: str
    notes: list[str]


def _dump(page: Page, out_dir: Path, name: str) -> Finding:
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = page.locator("body").aria_snapshot(mode="ai")
    (out_dir / f"{name}.snapshot.txt").write_text(snap, encoding="utf-8")
    page.screenshot(path=str(out_dir / f"{name}.png"), full_page=True)
    return Finding(page=name, url=page.url, status=None, ai_snapshot=snap, notes=[])


def recon(base_url: str, label: str, username: str | None, password: str | None) -> None:
    out_dir = OUT_ROOT / label
    findings: list[Finding] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        # --- index / login ---------------------------------------------------
        resp = page.goto(f"{base_url}/index.htm", wait_until="load")
        f = _dump(page, out_dir, "01_index")
        f.status = resp.status if resp else None
        f.notes.append(f"final url after redirect(s): {page.url}")
        findings.append(f)

        session_id_pattern = re.search(r";jsessionid=([0-9A-F]+)", page.url, re.IGNORECASE)
        if session_id_pattern:
            findings[-1].notes.append("confirmed: jsessionid is carried in the URL path")

        if username and password:
            page.fill("input[name='username']", username)
            page.fill("input[name='password']", password)
            page.click("input[value='Log In']")
            page.wait_for_load_state("load")
            f = _dump(page, out_dir, "02_overview")
            f.notes.append(f"login({username!r}) -> url={page.url}")
            findings.append(f)

            logged_in = "overview" in page.url or page.locator("text=Account Overview").count() > 0
            findings[-1].notes.append(f"login_succeeded={logged_in}")

            if logged_in:
                # --- open-account: before -----------------------------------
                page.goto(f"{base_url}/openaccount.htm", wait_until="load")
                f = _dump(page, out_dir, "03_openaccount_before")
                type_select = page.locator("select#type")
                from_select = page.locator("select#fromAccountId")
                f.notes.append(f"select#type present={type_select.count() > 0}")
                if type_select.count() > 0:
                    opts = type_select.locator("option").all_inner_texts()
                    f.notes.append(f"select#type options={opts}")
                f.notes.append(f"select#fromAccountId present={from_select.count() > 0}")
                if from_select.count() > 0:
                    page.wait_for_function(
                        "document.querySelector('select#fromAccountId').options.length > 0"
                    )
                    opts2 = from_select.locator("option").all_inner_texts()
                    f.notes.append(f"select#fromAccountId options(after AJAX load)={opts2}")
                btn = page.get_by_role("button", name="Open New Account")
                nav_link = page.get_by_role("link", name="Open New Account")
                f.notes.append(f"submit button role=button count={btn.count()}")
                f.notes.append(f"nav link role=link count={nav_link.count()}")
                findings.append(f)

                # --- open-account: submit + after ---------------------------
                if type_select.count() > 0 and from_select.count() > 0 and btn.count() > 0:
                    type_select.select_option(label="SAVINGS")
                    from_select.select_option(index=0)
                    btn.click()
                    page.wait_for_load_state("load")
                    f = _dump(page, out_dir, "04_openaccount_after")
                    opened = page.locator("text=Account Opened!").count() > 0
                    new_id_link = page.locator("#newAccountId")
                    f.notes.append(f"'Account Opened!' visible={opened}")
                    f.notes.append(f"#newAccountId present={new_id_link.count() > 0}")
                    if new_id_link.count() > 0:
                        f.notes.append(f"#newAccountId text={new_id_link.inner_text()!r}")
                    findings.append(f)

                # --- activity ------------------------------------------------
                page.goto(f"{base_url}/activity.htm", wait_until="load")
                findings.append(_dump(page, out_dir, "05_activity"))

                # --- a deliberate error page (bogus account id) --------------
                page.goto(f"{base_url}/activity.htm?id=999999999", wait_until="load")
                f = _dump(page, out_dir, "06_error")
                err = page.locator("text=An internal error has occurred and has been logged.")
                f.notes.append(f"error panel visible={err.count() > 0}")
                findings.append(f)

                # --- session-expired signature (clear cookies, revisit) ------
                context.clear_cookies()
                page.goto(f"{base_url}/overview.htm", wait_until="load")
                f = _dump(page, out_dir, "07_session_expired")
                f.notes.append(f"url_after_clearing_cookies={page.url}")
                login_form_count = page.locator("input[name='username']").count()
                f.notes.append(f"login form present={login_form_count > 0}")
                findings.append(f)

        browser.close()

    manifest = [
        {"page": fnd.page, "url": fnd.url, "status": fnd.status, "notes": fnd.notes}
        for fnd in findings
    ]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{label}] wrote {len(findings)} pages to {out_dir}")
    for fnd in findings:
        print(f"  - {fnd.page}: {fnd.url}")
        for note in fnd.notes:
            print(f"      {note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", choices=["local", "public"])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--username", default=None)
    ap.add_argument("--password", default=None)
    args = ap.parse_args()
    recon(args.base_url, args.label, args.username, args.password)


if __name__ == "__main__":
    main()
