"""The browser-layer guard: every request, every navigation, and the redirect gap."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from playwright.sync_api import BrowserContext, Page, Playwright

from tests.conftest import PARABANK_POLICY, Loopback, sign_in
from tests.fixtures.site import BASE_URL, FixtureSite
from understudy.artifact.signatures import VisibleCondition
from understudy.policy.gate import PolicyGate
from understudy.surface.base import ActionTimeout, PolicyBlocked
from understudy.surface.web import CONTEXT_OPTIONS, WebSurface
from understudy.tenant import TenantBinding

ADMIN = "/parabank/admin.htm"


def loopback_gate(server: Loopback) -> PolicyGate:
    tenant = TenantBinding(id="loop", vendor_app="parabank", base_url=server.base_url, sandbox=True)
    return PolicyGate.load(PARABANK_POLICY, tenant)


# --- the routed fixture ------------------------------------------------------------------


def test_navigating_to_a_denied_route_sends_nothing(surface: WebSurface, site: FixtureSite) -> None:
    with pytest.raises(PolicyBlocked) as caught:
        surface.navigate(f"{BASE_URL}/admin.htm")
    assert caught.value.rule == "routes.deny"
    assert site.reached(ADMIN) == 0


def test_a_clicked_link_to_a_denied_route_is_stopped_at_the_network(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    sign_in(page)
    surface.navigate(f"{BASE_URL}/overview.htm")
    # A nav link is a read, so the click itself is allowed; the request is not.
    surface.act("click", page.get_by_role("link", name="Admin Page"))
    page.wait_for_load_state("load")
    assert site.reached(ADMIN) == 0
    assert [(v.rule, v.navigation) for v in surface.violations] == [("routes.deny", True)]
    with pytest.raises(PolicyBlocked, match="taken off policy"):
        surface.probe(VisibleCondition(text="Administration"))

    surface.navigate(f"{BASE_URL}/overview.htm")
    assert surface.probe(VisibleCondition(role="heading", name="Accounts Overview"))


def test_an_off_host_link_is_stopped_at_the_network(
    page: Page, surface: WebSurface, site: FixtureSite
) -> None:
    sign_in(page)
    surface.navigate(f"{BASE_URL}/overview.htm")
    page.evaluate(
        "() => { const a = document.createElement('a'); a.href = 'https://www.parasoft.com/x';"
        " a.textContent = 'Products'; document.body.prepend(a); }"
    )
    surface.act("click", page.get_by_role("link", name="Products"))
    page.wait_for_load_state("load")
    assert [v.rule for v in surface.violations] == ["host"]


def test_off_policy_subresources_are_blocked_without_tainting_the_page(
    page: Page, surface: WebSurface
) -> None:
    surface.navigate(f"{BASE_URL}/index.htm")
    status = page.evaluate(
        "() => fetch('https://tracker.example/pixel').then(r => r.status, () => 'blocked')"
    )
    assert status == "blocked"
    assert [(v.rule, v.navigation) for v in surface.violations] == [("host", False)]
    assert surface.probe(VisibleCondition(role="heading", name="Customer Login"))


# --- redirects: a real server, because Playwright never routes redirect hops -------------


@pytest.fixture
def browser_with_rules(playwright: Playwright, loopback: Loopback) -> Iterator[BrowserContext]:
    browser = playwright.chromium.launch(args=loopback_gate(loopback).chromium_args())
    context = browser.new_context(**CONTEXT_OPTIONS)
    yield context
    context.close()
    browser.close()


def test_a_same_host_redirect_to_a_denied_route_is_detected(
    context: BrowserContext, loopback: Loopback
) -> None:
    """Detected, not prevented: the hop reaches the server before anything can stop it.
    The surface refuses to read or drive the page it landed on."""
    surface = WebSurface(context.new_page(), loopback_gate(loopback), mode="replay")
    with pytest.raises(PolicyBlocked) as caught:
        surface.navigate(f"{loopback.base_url}/hop-to-admin")
    assert caught.value.rule == "routes.deny"
    assert [v.via for v in surface.violations] == ["redirect"]
    assert f"127.0.0.1:{loopback.port}/app/admin.htm" in loopback.hits
    with pytest.raises(PolicyBlocked):
        surface.observe()


def test_without_resolver_rules_an_off_host_redirect_lands(
    context: BrowserContext, loopback: Loopback
) -> None:
    surface = WebSurface(context.new_page(), loopback_gate(loopback), mode="replay")
    with pytest.raises(PolicyBlocked):
        surface.navigate(f"{loopback.base_url}/hop-off-host")
    assert f"localhost:{loopback.port}/app/landed" in loopback.hits


def test_resolver_rules_stop_an_off_host_redirect_before_it_is_sent(
    browser_with_rules: BrowserContext, loopback: Loopback
) -> None:
    surface = WebSurface(browser_with_rules.new_page(), loopback_gate(loopback), mode="replay")
    assert surface.navigate(f"{loopback.base_url}/landed") == 200
    with pytest.raises(PolicyBlocked) as caught:
        surface.navigate(f"{loopback.base_url}/hop-off-host")
    assert caught.value.rule == "host"
    assert not any(hit.startswith("localhost") for hit in loopback.hits)


# --- regressions from a code review ----------------------------------------------------------


def test_a_page_route_registered_first_cannot_bypass_the_guard(
    page: Page, site: FixtureSite, gate: PolicyGate
) -> None:
    page.route("**/admin.htm", lambda route: route.fulfill(body="<h1>Administration</h1>"))
    surface = WebSurface(page, gate, mode="replay")
    sign_in(page)
    surface.navigate(f"{BASE_URL}/overview.htm")
    surface.act("click", page.get_by_role("link", name="Admin Page"))
    page.wait_for_load_state("load")
    assert [v.rule for v in surface.violations] == ["routes.deny"]
    with pytest.raises(PolicyBlocked):
        surface.probe(VisibleCondition(text="Administration"))


def test_a_slow_page_with_a_blocked_tracker_times_out_not_policy_blocked(
    context: BrowserContext, site: FixtureSite, gate: PolicyGate
) -> None:
    body = (
        "<h1>Slow</h1><img src='https://tracker.example/p.gif'>"
        "<img src='https://fixture.test/parabank/never.gif'>"
    )
    context.route("**/slow.htm", lambda route: route.fulfill(content_type="text/html", body=body))
    context.route("**/never.gif", lambda route: None)  # never answered
    surface = WebSurface(context.new_page(), gate, mode="replay")
    with pytest.raises(ActionTimeout):
        surface.navigate(f"{BASE_URL}/slow.htm", timeout_ms=1_000)
    assert [(v.rule, v.navigation) for v in surface.violations] == [("host", False)]


def test_a_redirect_hop_to_a_commit_route_is_detected(
    context: BrowserContext, loopback: Loopback
) -> None:
    surface = WebSurface(context.new_page(), loopback_gate(loopback), mode="replay")
    with pytest.raises(PolicyBlocked) as caught:
        surface.navigate(f"{loopback.base_url}/hop-to-commit")
    assert caught.value.rule == "commit_window"
    assert [(v.rule, v.via) for v in surface.violations] == [("commit_window", "redirect")]
