"""The session provider against the routed fixture: the login form from the profile."""

from __future__ import annotations

import pytest

from tests.conftest import FIXTURE_TENANT, REPO
from tests.fixtures.site import PASSWORD, USERNAME, FixtureSite
from understudy.profile import load_profile
from understudy.session import Credentials, FormLogin, LoginFailed, MissingCredentials
from understudy.surface.web import WebSurface

FORM = load_profile(REPO / "apps" / "parabank" / "profile.yaml").login


def test_logging_in_reaches_a_signed_in_page(surface: WebSurface, site: FixtureSite) -> None:
    FormLogin(FORM, FIXTURE_TENANT, Credentials(USERNAME, PASSWORD)).login(surface)
    assert site.authenticated
    assert surface.probe(FORM.signed_in)


def test_refused_credentials_raise_login_failed(surface: WebSurface, site: FixtureSite) -> None:
    with pytest.raises(LoginFailed, match="refused"):
        FormLogin(FORM, FIXTURE_TENANT, Credentials(USERNAME, "wrong")).login(surface)
    assert not site.authenticated


def test_credentials_never_print_themselves() -> None:
    assert PASSWORD not in repr(Credentials(USERNAME, PASSWORD))
    assert USERNAME not in repr(Credentials(USERNAME, PASSWORD))


def test_missing_credentials_name_the_variables_not_the_values() -> None:
    with pytest.raises(MissingCredentials, match="PARABANK_PASSWORD") as raised:
        Credentials.from_env(FORM, {"PARABANK_USERNAME": "someone"})
    assert "someone" not in str(raised.value)
