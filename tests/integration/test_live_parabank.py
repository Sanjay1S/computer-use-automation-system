"""Live checks against the local ParaBank container (`make parabank-up`).

Skipped by default; run with `make test-live`, which reads PARABANK_PORT and the
ParaBank credentials from .env (or from the make command line).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import Playwright

from tests.conftest import PARABANK_POLICY, REPO
from understudy.artifact import hashing
from understudy.artifact.signatures import UrlCondition, VisibleCondition
from understudy.policy.gate import PolicyGate
from understudy.profile import load_profile
from understudy.replay.engine import Replayer
from understudy.replay.preflight import grant
from understudy.session import Credentials, MissingCredentials
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


def test_the_hand_written_artifact_replays_on_local_parabank(
    playwright: Playwright, tmp_path: Path
) -> None:
    """Replay acceptance: the real app, the real policy, no fixture. Opens one account."""
    tenant = load_tenant(REPO / "tenants" / "parabank-local.yaml")
    gate = PolicyGate.load(PARABANK_POLICY, tenant)
    profile = load_profile(REPO / "apps" / "parabank" / "profile.yaml")
    try:
        credentials = Credentials.from_env(profile.login)
    except MissingCredentials as error:
        pytest.fail(f"this live check needs ParaBank credentials: {error}")
    capability = hashing.load(REPO / "tests" / "fixtures" / "example_capability.json")
    inputs = {"account_type": "SAVINGS", "funding_account_id": "12345"}
    browser = playwright.chromium.launch(args=gate.chromium_args())
    try:
        result = Replayer(
            capability, gate=gate, profile=profile, credentials=credentials, runs_dir=tmp_path
        ).run(
            inputs,
            open_page=lambda: browser.new_context(**CONTEXT_OPTIONS).new_page(),
            approval=grant(capability, inputs, tenant.id, "live test"),
            allow_draft=True,
        )
    finally:
        browser.close()
    assert result.status == "succeeded", result.model_dump_json(indent=2)
    assert result.outputs is not None
    assert re.fullmatch(r"\d+", result.outputs["new_account_id"])
    assert result.side_effects.commit_state == "committed"
    assert result.side_effects.commit_requests == 1
