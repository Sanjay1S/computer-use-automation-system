from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.conftest import REPO
from tests.fixtures.example_capability import example_capability
from understudy.artifact import hashing
from understudy.artifact.schema import Fingerprint, Locator, Target
from understudy.artifact.signatures import DialogCondition, VisibleCondition
from understudy.chaos import Fault, parse_fault
from understudy.profile import Interstitial, load_profile

CAP = example_capability()


@pytest.mark.parametrize(
    ("spec", "fault"),
    [
        ("drop_commit_response", Fault("drop_commit_response")),
        ("app_error_at_commit", Fault("app_error_at_commit")),
        ("session_expired_at=s3", Fault("session_expired_at", step="s3_funding_account")),
        ("rename_target:s4", Fault("rename_target", step="s4_submit")),
        (
            "unknown_modal_at:s3_funding_account",
            Fault("unknown_modal_at", step="s3_funding_account"),
        ),
        ("http_503_once:/openaccount.htm", Fault("http_503_once", route="/openaccount.htm")),
        ("latency:/services*/accounts:800", Fault("latency", route="/services*/accounts", ms=800)),
    ],
)
def test_fault_specs_parse(spec: str, fault: Fault) -> None:
    assert parse_fault(spec, CAP) == fault


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("teleport", "unknown fault"),
        ("rename_target:s9", "matches no step"),
        ("rename_target", "needs: <step>"),
        ("drop_commit_response:s4", "takes no arguments"),
        ("http_503_once:openaccount.htm", "must start with /"),
        ("latency:/a:0", "1-60000 ms"),
        ("latency:/a", "1-60000 ms"),
    ],
)
def test_bad_fault_specs_say_what_is_wrong(spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_fault(spec, CAP)


def test_a_fault_prints_back_as_a_spec() -> None:
    assert str(parse_fault("latency:/a/*:50", CAP)) == "latency:/a/*:50"


def test_the_parabank_profile_loads_and_names_the_capabilitys_session() -> None:
    profile = load_profile(REPO / "apps" / "parabank" / "profile.yaml")
    assert profile.login.session == CAP.requires.session
    assert {s.id: s.handling for s in profile.global_states} == {
        "session_expired": "reauthenticate",
        "server_error": "transient",
        "app_internal_error": "transient",
    }
    assert profile.interstitials == []  # none verified on ParaBank


def test_an_interstitial_needs_exactly_one_way_out() -> None:
    button = Target(
        description="OK",
        fingerprint=Fingerprint(role="button", name="OK"),
        locators=[Locator(by="css", value="#ok", stability="high", note="n")],
    )
    when = VisibleCondition(css="#notice")
    with pytest.raises(ValidationError):
        Interstitial(id="x", description="d", when=when)
    with pytest.raises(ValidationError):
        Interstitial(id="x", description="d", when=when, dismiss=button, dialog="accept")
    with pytest.raises(ValidationError, match="is a dialog"):
        Interstitial(id="x", description="d", when=when, dialog="accept")
    known = Interstitial(
        id="x", description="d", when=DialogCondition(message_regex="^Hi"), dialog="accept"
    )
    profile = load_profile(REPO / "apps" / "parabank" / "profile.yaml")
    assert profile.model_copy(update={"interstitials": [known]}).dialog_rules() == [
        ("^Hi", "accept")
    ]


def test_the_committed_draft_is_what_the_generator_builds() -> None:
    """Two copies of one artifact: the JSON reviewers read and the code tests mutate."""
    committed = REPO / "tests" / "fixtures" / "example_capability.json"
    assert committed.read_text(encoding="utf-8") == hashing.canonical_json(CAP)


class _FakeRoute:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fallback(self) -> None:
        self.calls.append("fallback")

    def fulfill(self, **kwargs: object) -> None:
        self.calls.append(f"fulfill {kwargs['status']}")


class _FakeRequest:
    url = "https://fixture.test/parabank/services_proxy/bank/customers/1/accounts"


def test_network_faults_fire_only_while_armed_and_not_during_a_login() -> None:
    from tests.conftest import FIXTURE_TENANT, PARABANK_POLICY
    from understudy.chaos import Chaos
    from understudy.policy.gate import PolicyGate

    chaos = Chaos(
        [parse_fault("http_503_once:/services_proxy/bank/customers/*/accounts", CAP)],
        PolicyGate.load(PARABANK_POLICY, FIXTURE_TENANT),
    )
    route = _FakeRoute()
    chaos._route(route, _FakeRequest())  # type: ignore[arg-type]
    chaos._armed = True
    with chaos.paused():
        chaos._route(route, _FakeRequest())  # type: ignore[arg-type]
    chaos._route(route, _FakeRequest())  # type: ignore[arg-type]
    assert route.calls == ["fallback", "fallback", "fulfill 503"]
