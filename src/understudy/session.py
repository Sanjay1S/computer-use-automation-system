"""The session provider: signs in before a capability runs and again when replay
recovers an expired session. Credentials pass through here only. They are never
part of a capability, never sent to a model, and registered with the redactor so
they are masked wherever they might reappear.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass

from playwright.sync_api import Locator as PwLocator

from understudy.artifact.schema import Sensitivity, Target
from understudy.control.plane import Lease
from understudy.profile import LoginForm
from understudy.surface.base import SurfaceError
from understudy.surface.web import WebSurface
from understudy.tenant import TenantBinding


class LoginFailed(SurfaceError):
    code = "login_failed"


class MissingCredentials(Exception):
    pass


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str

    def __repr__(self) -> str:
        return "Credentials(username=[PII], password=[SECRET])"

    @classmethod
    def from_env(cls, form: LoginForm, env: Mapping[str, str] | None = None) -> Credentials:
        source = os.environ if env is None else env
        missing = [n for n in (form.username_env, form.password_env) if not source.get(n)]
        if missing:
            raise MissingCredentials(f"set {' and '.join(missing)} (in the environment or .env)")
        return cls(source[form.username_env], source[form.password_env])

    @property
    def sensitivities(self) -> dict[str, Sensitivity]:
        return {self.username: "pii", self.password: "secret"}


class FormLogin:
    def __init__(self, form: LoginForm, tenant: TenantBinding, credentials: Credentials) -> None:
        self.form = form
        self.tenant = tenant
        self._credentials = credentials

    def login(self, surface: WebSurface, *, poll_ms: int = 100, lease: Lease | None = None) -> None:
        """Raises LoginFailed when the app refuses the credentials or never shows a
        signed-in page; other surface errors (policy, infra) pass through. `lease` is
        the lease of the automation it signs in for."""
        form = self.form
        deadline = time.monotonic() + form.timeout_ms / 1000
        surface.navigate(
            f"{self.tenant.base_url}{form.route}", timeout_ms=form.timeout_ms, lease=lease
        )
        with surface.session_provider():
            for target, value in (
                (form.username_field, self._credentials.username),
                (form.password_field, self._credentials.password),
            ):
                surface.act(
                    "fill",
                    self._find(surface, target, deadline, poll_ms),
                    value,
                    fingerprint=target.fingerprint,
                    lease=lease,
                )
            surface.act(
                "click",
                self._find(surface, form.submit, deadline, poll_ms),
                fingerprint=form.submit.fingerprint,
                lease=lease,
            )
        while True:
            # Rejected first: a refused login re-renders the form, which is not signed in
            # but also not "still loading".
            if surface.probe(form.rejected):
                raise LoginFailed("the app refused the credentials")
            if surface.probe(form.signed_in):
                return
            if time.monotonic() >= deadline:
                raise LoginFailed("no signed-in page appeared after logging in")
            surface.pause(poll_ms)

    @staticmethod
    def _find(surface: WebSurface, target: Target, deadline: float, poll_ms: int) -> PwLocator:
        while True:
            resolution = surface.resolve(target, {})
            if resolution.element is not None:
                return resolution.element
            if time.monotonic() >= deadline:
                raise LoginFailed(f"the login form has no {target.description.lower()}")
            surface.pause(poll_ms)
