"""The policy gate: hosts, routes, members, keys, commit classification."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.conftest import PARABANK_POLICY, REPO
from tests.fixtures.example_capability import example_capability
from understudy.artifact.lint import lint
from understudy.artifact.schema import Disposition, Expectation, Step
from understudy.artifact.signatures import VisibleCondition
from understudy.policy.gate import PolicyGate
from understudy.policy.models import Policy, PolicyOverlay, load_policy
from understudy.surface.base import ElementFacts
from understudy.tenant import TenantBinding, load_tenant

LOCAL = TenantBinding(
    id="local", vendor_app="parabank", base_url="http://localhost:8089/parabank/", sandbox=True
)


@pytest.fixture
def local() -> PolicyGate:
    return PolicyGate.load(PARABANK_POLICY, LOCAL)


def button(name: str) -> ElementFacts:
    return ElementFacts(tag="input", role="button", name=name, input_type="button")


LINK = ElementFacts(tag="a", role="link", name="Open New Account", href="http://x/openaccount.htm")


# --- tenant binding ----------------------------------------------------------------


def test_tenant_binding_parts() -> None:
    assert LOCAL.base_url == "http://localhost:8089/parabank"
    assert (LOCAL.scheme, LOCAL.host, LOCAL.port, LOCAL.base_path) == (
        "http",
        "localhost",
        8089,
        "/parabank",
    )


@pytest.mark.parametrize(
    "base_url",
    ["localhost:8080/parabank", "ftp://h/x", "http://h/x?a=1", "http://h/x;jsessionid=AB"],
)
def test_tenant_rejects_malformed_base_urls(base_url: str) -> None:
    with pytest.raises(ValidationError):
        TenantBinding(id="t", vendor_app="parabank", base_url=base_url)


def test_the_committed_tenants_and_policy_load() -> None:
    policy = load_policy(PARABANK_POLICY)
    for name in ("parabank-local", "parabank-public"):
        tenant = load_tenant(REPO / "tenants" / f"{name}.yaml")
        assert tenant.id == name and tenant.sandbox
        PolicyGate(policy, tenant)


def test_a_policy_for_another_app_is_refused() -> None:
    policy = load_policy(PARABANK_POLICY).model_copy(update={"vendor_app": "other"})
    with pytest.raises(ValueError, match="policy is for 'other'"):
        PolicyGate(policy, LOCAL)


# --- URLs --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8089/parabank/openaccount.htm",
        "http://localhost:8089/parabank/openaccount.htm;jsessionid=C66C519E?x=1",
        "http://localhost:8089/parabank/",
        "http://localhost:8089/parabank",
        "http://LOCALHOST:8089/parabank/overview.htm",
        "about:blank",
    ],
)
def test_allowed_urls(local: PolicyGate, url: str) -> None:
    assert local.check_url(url).allowed, local.check_url(url)


@pytest.mark.parametrize(
    ("url", "rule"),
    [
        ("http://localhost:8089/parabank/admin.htm", "routes.deny"),
        ("http://localhost:8089/parabank/ADMIN.htm", "routes.deny"),
        ("http://localhost:8089/parabank/admin.htm;jsessionid=ABC", "routes.deny"),
        ("http://localhost:8089/parabank/%61dmin.htm", "routes.deny"),
        ("http://localhost:8089/parabank/x/../admin.htm", "routes.deny"),
        ("http://localhost:8089/parabank/x/%2e%2e/initializeDB.htm", "routes.deny"),
        ("http://localhost:8089/parabank/logout.htm", "routes.deny"),
        ("http://localhost:8089/parabank/services/bank/cleanDB", "routes.deny"),
        ("http://localhost:8089/parabank/services_proxy/bank/initializeDB", "routes.deny"),
        ("http://localhost:8089/other/index.htm", "routes.base_path"),
        ("http://localhost:8089/parabankx/index.htm", "routes.base_path"),
        ("http://localhost:8080/parabank/index.htm", "host"),
        ("http://127.0.0.1:8089/parabank/index.htm", "host"),
        ("http://www.parasoft.com/jsp/products.jsp", "host"),
        ("https://localhost:8089/parabank/index.htm", "scheme"),
        ("javascript:alert(1)", "scheme"),
        ("file:///etc/passwd", "scheme"),
        ("data:text/html,hi", "scheme"),
    ],
)
def test_denied_urls(local: PolicyGate, url: str, rule: str) -> None:
    decision = local.check_url(url)
    assert decision.verdict == "deny"
    assert decision.rule == rule


def test_routes_are_relative_to_the_tenant_base_url(local: PolicyGate) -> None:
    assert local.route_of("http://localhost:8089/parabank/openaccount.htm;jsessionid=A") == (
        "/openaccount.htm"
    )
    assert local.route_of("http://localhost:8089/elsewhere/x") is None
    assert local.route_of("http://other:8089/parabank/x") is None


def test_commit_routes(local: PolicyGate) -> None:
    assert local.is_commit_request(
        "http://localhost:8089/parabank/services_proxy/bank/createAccount?customerId=1"
    )
    assert not local.is_commit_request(
        "http://localhost:8089/parabank/services_proxy/bank/customers/1/accounts"
    )


def test_chromium_args_resolve_only_the_tenant_host(local: PolicyGate) -> None:
    assert local.chromium_args() == ["--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost"]


# --- tenant overlay ------------------------------------------------------------------


def test_the_public_tenant_overlay_tightens_the_vendor_policy() -> None:
    public = PolicyGate.load(PARABANK_POLICY, load_tenant(REPO / "tenants/parabank-public.yaml"))
    assert public.check_url("https://parabank.parasoft.com/parabank/transfer.htm").rule == (
        "routes.deny"
    )
    assert public.check_url("https://parabank.parasoft.com/parabank/openaccount.htm").allowed
    assert public.check_url("https://parabank.parasoft.com/parabank/admin.htm").rule == (
        "routes.deny"
    )


def test_an_overlay_can_remove_members_and_keys() -> None:
    tenant = LOCAL.model_copy(
        update={"policy": PolicyOverlay(remove_actions=["key"], remove_keys=["Enter"])}
    )
    gate = PolicyGate.load(PARABANK_POLICY, tenant)
    assert gate.check_member("key").rule == "actions"
    assert "key" not in gate.allowed_members


def test_an_overlay_has_no_field_that_could_widen_policy() -> None:
    for widening in ({"allow_routes": ["/admin.htm"]}, {"actions": ["javascript_exec"]}):
        with pytest.raises(ValidationError):
            PolicyOverlay.model_validate(widening)


def test_a_policy_cannot_enable_members_outside_the_supported_set() -> None:
    with pytest.raises(ValidationError):
        Policy.model_validate({"vendor_app": "parabank", "actions": ["javascript_exec"]})


# --- members, keys, commit classification ------------------------------------------


def test_members_not_in_the_policy_are_denied(local: PolicyGate) -> None:
    assert local.check_action("left_click", button("Go"), mode="replay").allowed
    decision = local.check_action("find", None, mode="discovery")
    assert decision.verdict == "deny" and decision.rule == "actions"


@pytest.mark.parametrize(
    ("key", "allowed"),
    [("Enter", True), ("Return", True), ("Esc", True), ("Tab", True), ("ctrl+a", False)],
)
def test_keys(local: PolicyGate, key: str, allowed: bool) -> None:
    assert local.check_key(key).allowed is allowed


def test_a_commit_button_needs_approval_but_the_same_named_link_does_not(
    local: PolicyGate,
) -> None:
    submit = local.check_action("left_click", button("Open New Account"), mode="replay")
    assert submit.verdict == "require_approval"
    assert submit.effect == "commit"

    nav = local.check_action("left_click", LINK, mode="replay")
    assert nav.allowed and nav.effect == "read"


def test_commit_names_match_whole_names_only(local: PolicyGate) -> None:
    assert local.classify("left_click", button("Open New Account Help")) == "input"
    assert local.classify("left_click", button("  Open   New Account ")) == "commit"


def test_a_submit_posting_to_a_commit_route_commits_whatever_its_label(local: PolicyGate) -> None:
    action = "http://localhost:8089/parabank/services_proxy/bank/createAccount"
    renamed = ElementFacts(
        tag="input", role="button", name="Continue", input_type="submit", form_action=action
    )
    assert local.classify("left_click", renamed) == "commit"
    plain = ElementFacts(
        tag="input",
        role="button",
        name="Continue",
        input_type="button",
        form_action=action,
    )
    assert local.classify("left_click", plain) == "input"


def test_a_link_to_a_commit_route_commits(local: PolicyGate) -> None:
    link = ElementFacts(
        tag="a",
        role="link",
        name="Go",
        href="http://localhost:8089/parabank/services_proxy/bank/createAccount?x=1",
    )
    assert local.classify("left_click", link) == "commit"


def test_enter_inside_a_commit_form_is_a_commit(local: PolicyGate) -> None:
    field = ElementFacts(tag="select", role="combobox", form_buttons=("Open New Account",))
    assert local.check_action("key", field, key="Enter", mode="replay").verdict == (
        "require_approval"
    )
    assert local.check_action("key", field, key="Tab", mode="replay").allowed
    search = ElementFacts(tag="input", input_type="text", form_buttons=("Find Transactions",))
    assert local.check_action("key", search, key="Enter", mode="replay").allowed


def test_a_click_with_nothing_to_judge_is_treated_as_a_commit(local: PolicyGate) -> None:
    assert local.classify("left_click", None) == "commit"


def test_read_members_are_reads(local: PolicyGate) -> None:
    for member in ("navigate", "read_page", "screenshot", "scroll", "hover"):
        assert local.classify(member, None) == "read"


def test_typing_a_password_is_denied_in_discovery(local: PolicyGate) -> None:
    password = ElementFacts(tag="input", input_type="password", name_attr="password")
    denied = local.check_action("type", password, mode="discovery")
    assert denied.verdict == "deny" and denied.rule == "password"
    assert local.check_action("form_input", password, mode="replay").allowed


def test_discovery_needs_a_sandbox_tenant() -> None:
    production = LOCAL.model_copy(update={"sandbox": False})
    assert PolicyGate.load(PARABANK_POLICY, production).check_discovery_allowed().rule == (
        "discovery"
    )
    assert PolicyGate.load(PARABANK_POLICY, LOCAL).check_discovery_allowed().allowed


# --- lint L007 against the real gate -------------------------------------------------


def test_the_example_artifact_is_within_the_real_policy(local: PolicyGate) -> None:
    assert "L007" not in lint(example_capability(), local).codes()


def test_l007_flags_a_step_on_a_denied_route(local: PolicyGate) -> None:
    capability = example_capability()
    capability.steps[0] = capability.steps[0].model_copy(update={"url": "{base_url}/admin.htm"})
    assert "L007" in lint(capability, local).codes()


def test_l007_flags_a_locator_that_names_a_denied_page(local: PolicyGate) -> None:
    capability = example_capability()
    step = capability.steps[3]
    assert step.target is not None
    locator = step.target.locators[1].model_copy(update={"value": "a[href*='admin.htm']"})
    capability.steps[3] = Step.model_validate(
        {
            **step.model_dump(),
            "target": {**step.target.model_dump(), "locators": [locator.model_dump()]},
            "expect": [
                Expectation(
                    description="x",
                    when=VisibleCondition(text="x"),
                    then=Disposition(do="continue"),
                ).model_dump()
            ],
        }
    )
    assert "L007" in lint(capability, local).codes()


def test_enter_on_a_commit_button_outside_any_form_is_a_commit(local: PolicyGate) -> None:
    assert local.classify("key", button("Open New Account"), key="Enter") == "commit"


def test_a_hidden_commit_button_is_judged_by_its_label(local: PolicyGate) -> None:
    """A hidden element has no accessible name; its value still says what it does."""
    hidden = ElementFacts(tag="input", input_type="button", value_attr="Open New Account")
    assert local.classify("left_click", hidden) == "commit"
