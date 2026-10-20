"""Outcome-race priority, against a fake page whose true signatures are listed."""

from __future__ import annotations

from collections.abc import Mapping

from understudy.artifact.schema import Disposition, Expectation, Fingerprint, Locator, Target
from understudy.artifact.signatures import (
    HttpStatusCondition,
    Signature,
    VisibleCondition,
)
from understudy.profile import GlobalState, Interstitial
from understudy.replay.race import first_hit

LOGIN = VisibleCondition(role="heading", name="Customer Login")
ERROR = HttpStatusCondition(min=500, max=599)
SUCCESS = VisibleCondition(text="Account Opened!")
APP_ERROR = VisibleCondition(text="An internal error has occurred and has been logged.")
NOTICE = VisibleCondition(css="#notice")


class FakePage:
    def __init__(self, *true: Signature, dialog: tuple[str, str] | None = None) -> None:
        self.true = list(true)
        self.dialog = dialog
        self.probed: list[Signature] = []

    def probe(self, signature: Signature, params: Mapping[str, str] | None = None) -> bool:
        self.probed.append(signature)
        return signature in self.true

    @property
    def pending_dialog(self) -> tuple[str, str] | None:
        return self.dialog


GLOBALS = [
    GlobalState(id="session_expired", description="expired", when=LOGIN, handling="reauthenticate"),
    GlobalState(id="server_error", description="5xx", when=ERROR, handling="transient"),
]
INTERSTITIALS = [
    Interstitial(
        id="notice",
        description="notice",
        when=NOTICE,
        dismiss=Target(
            description="Close",
            fingerprint=Fingerprint(role="button", name="Close"),
            locators=[Locator(by="css", value="#notice button", stability="high", note="n")],
        ),
    ),
]
EXPECT = [
    Expectation(description="opened", when=SUCCESS, then=Disposition(do="continue")),
    Expectation(
        description="app error",
        when=APP_ERROR,
        then=Disposition(do="escalate", code="app_error_after_commit"),
    ),
]


def hit(page: FakePage) -> tuple[str, int] | None:
    found = first_hit(
        page,
        global_states=GLOBALS,
        interstitials=INTERSTITIALS,
        expectations=EXPECT,
        params={},
    )
    return (found.source, found.index) if found else None


def test_a_global_state_wins_over_the_steps_own_expectation() -> None:
    """An expired session must not be read as success because a stale success panel
    is still on screen."""
    assert hit(FakePage(SUCCESS, LOGIN)) == ("global", 0)


def test_global_states_are_checked_in_profile_order() -> None:
    assert hit(FakePage(ERROR, LOGIN)) == ("global", 0)
    assert hit(FakePage(ERROR)) == ("global", 1)


def test_interstitials_come_after_global_states_and_before_expectations() -> None:
    assert hit(FakePage(NOTICE, SUCCESS)) == ("interstitial", 0)
    assert hit(FakePage(NOTICE, ERROR)) == ("global", 1)


def test_expectations_are_tried_in_declared_order() -> None:
    assert hit(FakePage(APP_ERROR, SUCCESS)) == ("expect", 0)
    assert hit(FakePage(APP_ERROR)) == ("expect", 1)


def test_an_unknown_dialog_is_reported_last_and_only_when_nothing_else_holds() -> None:
    dismissed = FakePage(dialog=("alert", "Are you sure?"))
    found = first_hit(
        dismissed, global_states=GLOBALS, interstitials=[], expectations=EXPECT, params={}
    )
    assert found is not None and (found.source, found.index) == ("dialog", -1)
    assert "alert" in found.description
    assert hit(FakePage(ERROR, dialog=("alert", "x"))) == ("global", 1)
    assert hit(FakePage(SUCCESS, dialog=("alert", "x"))) == ("expect", 0)


def test_nothing_matching_is_no_hit() -> None:
    assert hit(FakePage()) is None
