import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from tests.fixtures.site import BASE_URL, PASSWORD, USERNAME, FixtureSite
from understudy.policy.gate import PolicyGate
from understudy.surface.web import CONTEXT_OPTIONS, WebSurface
from understudy.tenant import TenantBinding

REPO = Path(__file__).resolve().parent.parent
PARABANK_POLICY = REPO / "apps" / "parabank" / "policy.yaml"
FIXTURE_TENANT = TenantBinding(id="fixture", vendor_app="parabank", base_url=BASE_URL, sandbox=True)


@pytest.fixture(scope="session")
def playwright() -> Iterator[Playwright]:
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser(playwright: Playwright) -> Iterator[Browser]:
    browser = playwright.chromium.launch()
    yield browser
    browser.close()


@pytest.fixture
def context(browser: Browser) -> Iterator[BrowserContext]:
    context = browser.new_context(**CONTEXT_OPTIONS)
    yield context
    context.close()


@pytest.fixture
def page(context: BrowserContext) -> Page:
    return context.new_page()


@pytest.fixture
def site(context: BrowserContext) -> FixtureSite:
    site = FixtureSite()
    site.install(context)
    return site


@pytest.fixture
def gate() -> PolicyGate:
    """The real ParaBank policy, bound to the fixture origin."""
    return PolicyGate.load(PARABANK_POLICY, FIXTURE_TENANT)


@pytest.fixture
def surface(page: Page, site: FixtureSite, gate: PolicyGate) -> WebSurface:
    # `site` is requested first so its router is installed before the guard.
    return WebSurface(page, gate, mode="replay", run_id="test-run")


def sign_in(page: Page) -> None:
    """Stands in for the session provider: credentials never go through the surface."""
    page.goto(f"{BASE_URL}/index.htm")
    page.fill("input[name='username']", USERNAME)
    page.fill("input[name='password']", PASSWORD)
    page.click("input[value='Log In']")
    page.wait_for_load_state("load")


@dataclass
class Loopback:
    """A real HTTP server on 127.0.0.1, for the one thing the routed fixture cannot
    show: redirect hops, which Playwright never passes to route handlers."""

    port: int
    hits: list[str] = field(default_factory=list)
    """"host/path" for every request that arrived."""

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/app"


@pytest.fixture
def loopback() -> Iterator[Loopback]:
    state: dict[str, Loopback] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            server = state["server"]
            server.hits.append(f"{self.headers.get('Host')}{self.path}")
            redirects = {
                "/app/hop-off-host": f"http://localhost:{server.port}/app/landed",
                "/app/hop-to-admin": "/app/admin.htm",
                "/app/hop-to-commit": "/app/services_proxy/bank/createAccount?x=1",
            }
            if self.path in redirects:
                self.send_response(302)
                self.send_header("Location", redirects[self.path])
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Loopback page</h1>")

        def log_message(self, format: str, *args: object) -> None:
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state["server"] = Loopback(port=httpd.server_address[1])
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield state["server"]
    httpd.shutdown()
    httpd.server_close()
