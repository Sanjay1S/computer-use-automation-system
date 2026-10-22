"""A routed fake ParaBank at https://fixture.test: no server, no network.

The open-account markup is copied from local ParaBank's page source: the result and
error panels are in the page, hidden, from the start; the funding list arrives by
XHR; the submit is an `<input type="button">` that POSTs createAccount; the
account-type options are valued 0/1 and labelled CHECKING/SAVINGS. The session
behaves as recon found it (`apps/parabank/profile.yaml`): Tomcat-style `;jsessionid=` URL rewriting
before the cookie is established, the login form re-rendered in place on an
expired session, and a bad account id answered with a business error.

Redirect hops are never routed by Playwright, so this site never redirects: login
answers with the overview page directly.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import BrowserContext, Request, Route

ORIGIN = "https://fixture.test"
BASE_URL = f"{ORIGIN}/parabank"
USERNAME = "john"
PASSWORD = "fixture-pw-7781"
SSN = "123-45-6789"
CUSTOMER_ID = "12212"

_PAGES = Path(__file__).parent / "pages"
_PATH_PARAMS = re.compile(r";jsessionid=([0-9A-Fa-f]+)", re.IGNORECASE)


def page(name: str, **values: str) -> str:
    text = (_PAGES / name).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


@dataclass
class FixtureSite:
    accounts: list[str] = field(default_factory=lambda: ["12345", "12456", "13344", "54321"])
    next_id: int = 13566
    accounts_delay_ms: int = 0
    """Page-side delay before the funding list is filled, for readiness tests."""
    result_delay_ms: int = 0
    """Page-side delay before a created account is shown: the app committed, and the
    page says so only later (after replay has stopped waiting, in the verify tests)."""
    accounts_status: int = 200
    """Status the accounts list answers with; >= 400 makes the page show its
    internal-error panel, as ParaBank does when that XHR fails."""
    session_id: str = field(default_factory=lambda: secrets.token_hex(16).upper())
    authenticated: bool = False
    requests: list[tuple[str, str]] = field(default_factory=list)
    """(method, path) for every request that reached the site."""

    def install(self, context: BrowserContext) -> None:
        context.route(f"{ORIGIN}/**", self._handle)

    def reached(self, path: str) -> int:
        return sum(1 for _, p in self.requests if p == path)

    # --- routing -------------------------------------------------------------------

    def _handle(self, route: Route, request: Request) -> None:
        parts = urlsplit(request.url)
        url_sid = _PATH_PARAMS.search(parts.path)
        path = _PATH_PARAMS.sub("", parts.path)
        self.requests.append((request.method, path))
        cookie = request.all_headers().get("cookie", "")
        has_cookie = f"JSESSIONID={self.session_id}" in cookie
        known = has_cookie or (url_sid is not None and url_sid.group(1) == self.session_id)
        if not known:
            # As Tomcat does: a request that names no live session gets a new one,
            # signed out. So clearing cookies really does expire the session.
            self.session_id = secrets.token_hex(16).upper()
            self.authenticated = False
        signed_in = self.authenticated and known
        query = {k: v[0] for k, v in parse_qs(parts.query).items()}
        sid = "" if has_cookie else f";jsessionid={self.session_id}"
        headers = (
            {}
            if has_cookie
            else {"Set-Cookie": f"JSESSIONID={self.session_id}; Path=/parabank; HttpOnly"}
        )

        name = path.removeprefix("/parabank/")
        if name in ("", "index.htm"):
            return self._html(route, "Welcome", self._login(sid), "<h2>Welcome</h2>", sid, headers)
        if name == "login.htm" and request.method == "POST":
            form = {k: v[0] for k, v in parse_qs(request.post_data or "").items()}
            if form.get("username") == USERNAME and form.get("password") == PASSWORD:
                self.authenticated = True
                return self._overview(route)
            error = '<p class="error">The username and password could not be verified.</p>'
            return self._html(route, "Error", self._login(sid, error), "", sid, headers)
        if name == "logout.htm":
            self.authenticated = False
            return self._html(route, "Welcome", self._login(sid), "", sid, headers)
        if name.startswith("legacy/"):
            return self._legacy(route, name)
        protected = (
            "overview.htm",
            "openaccount.htm",
            "activity.htm",
            "admin.htm",
        )
        if name in protected and not signed_in:
            # Same URL, login form rendered in place. Not a redirect.
            return self._html(route, "Customer Login", self._login(sid), "", sid, headers)
        if name == "overview.htm":
            return self._overview(route)
        if name == "openaccount.htm":
            right = page(
                "openaccount.html",
                customer=CUSTOMER_ID,
                delay=str(self.accounts_delay_ms),
                result_delay=str(self.result_delay_ms),
            )
            return self._html(route, "Open Account", page("services_panel.html"), right, "", {})
        if name == "activity.htm":
            return self._activity(route, query.get("id", ""))
        if name == "admin.htm":
            return self._html(route, "Administration", "", "<h1>Administration</h1>", "", {})
        if name == f"services_proxy/bank/customers/{CUSTOMER_ID}/accounts":
            if not signed_in:
                return route.fulfill(status=401, body="")
            if self.accounts_status >= 400:
                return route.fulfill(status=self.accounts_status, body="")
            return self._json(
                route, [{"id": int(a), "customerId": int(CUSTOMER_ID)} for a in self.accounts]
            )
        if name == "services_proxy/bank/createAccount" and request.method == "POST":
            if not signed_in or query.get("fromAccountId") not in self.accounts:
                return route.fulfill(status=500, body="")
            new_id = str(self.next_id)
            self.next_id += 111
            self.accounts.append(new_id)
            return self._json(route, {"id": int(new_id), "customerId": int(CUSTOMER_ID)})
        return route.fulfill(status=404, content_type="text/html", body="<h1>Not Found</h1>")

    # --- pages ---------------------------------------------------------------------

    def _login(self, sid: str, error: str = "") -> str:
        return page("login_panel.html", sid=sid, error=error)

    def _overview(self, route: Route) -> None:
        rows = "\n".join(
            f'      <tr><td><a href="activity.htm?id={a}">{a}</a></td>'
            f"<td>$100.00</td><td>$100.00</td></tr>"
            for a in self.accounts
        )
        total = f"${100 * len(self.accounts)}.00"
        right = page("overview.html", rows=rows, total=total, ssn=SSN)
        self._html(route, "Accounts Overview", page("services_panel.html"), right, "", {})

    def _activity(self, route: Route, account: str) -> None:
        if account in self.accounts:
            right = f'<h1 class="title">Account Details</h1><p id="accountId">{account}</p>'
        else:
            # A business error, not the generic internal-error panel.
            right = (
                '<h1 class="title">Error!</h1>'
                f'<p class="error">Could not find account #{account}</p>'
            )
        self._html(route, "Account Activity", page("services_panel.html"), right, "", {})

    def _legacy(self, route: Route, name: str) -> None:
        files = {
            "legacy/frameset.htm": "legacy_frameset.html",
            "legacy/nav.htm": "legacy_nav.html",
            "legacy/main.htm": "legacy_main.html",
        }
        if name not in files:
            route.fulfill(status=404, body="")
            return
        route.fulfill(status=200, content_type="text/html", body=page(files[name]))

    @staticmethod
    def _html(
        route: Route, title: str, left: str, right: str, sid: str, headers: dict[str, str]
    ) -> None:
        body = page("layout.html", title=title, left=left, right=right, sid=sid)
        route.fulfill(status=200, content_type="text/html", headers=headers, body=body)

    @staticmethod
    def _json(route: Route, payload: object) -> None:
        route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))
