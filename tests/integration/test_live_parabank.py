"""Live checks against the local ParaBank container (`make parabank-up`).

Skipped by default; run with `make test-live`, which reads PARABANK_PORT from .env.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Playwright

from tests.conftest import PARABANK_POLICY, REPO
from understudy.artifact.signatures import UrlCondition, VisibleCondition
from understudy.policy.gate import PolicyGate
from understudy.surface.web import CONTEXT_OPTIONS, WebSurface
from understudy.tenant import load_tenant

pytestmark = pytest.mark.live


def test_the_local_tenant_reaches_parabank_through_the_gated_surface(
    playwright: Playwright,
) -> None:
    tenant = load_tenant(REPO / "tenants" / "parabank-local.yaml")
    gate = PolicyGate.load(PARABANK_POLICY, tenant)
    browser = playwright.chromium.launch(args=gate.chromium_args())
    try:
        surface = WebSurface(browser.new_context(**CONTEXT_OPTIONS).new_page(), gate, mode="replay")
        assert surface.navigate(f"{tenant.base_url}/index.htm") == 200
        assert surface.probe(VisibleCondition(role="heading", name="Customer Login"))
        assert surface.probe(UrlCondition(path_regex=r"^/index\.htm$"))
    finally:
        browser.close()
