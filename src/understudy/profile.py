"""The app profile: what is known about a vendor app, shared by every capability and
every tenant of it. Replay reads its global states and interstitials, the session
provider reads its login form, and evidence capture reads its screenshot masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from understudy.artifact.schema import Target
from understudy.artifact.signatures import DialogCondition, Signature


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GlobalState(_Model):
    """A page state any step can land in, checked ahead of the step's own expectations.

    `reauthenticate`: the session is gone. Before a commit, log in again and restart
    from the re-entry point; after it, a human decides.
    `transient`: the app or server failed in a way a retry may fix. Before a commit,
    retry the step if its policy allows; after it, a human decides.
    """

    id: str
    description: str
    when: Signature
    handling: Literal["reauthenticate", "transient"]


class Interstitial(_Model):
    """A known page or dialog that gets in the way and has a safe way out."""

    id: str
    description: str
    when: Signature
    dismiss: Target | None = None
    """The control that closes it. Clicked through the policy gate like any action."""
    dialog: Literal["accept", "dismiss"] | None = None
    """For a native dialog: how to answer it. The surface answers it the moment it
    opens (a dialog left open freezes the page), so `when` must be a dialog
    condition, whose message regex is the rule."""

    @model_validator(mode="after")
    def _one_way_out(self) -> Interstitial:
        if (self.dismiss is None) == (self.dialog is None):
            raise ValueError(f"interstitial {self.id!r} needs exactly one of: dismiss, dialog")
        if self.dialog is not None and not isinstance(self.when, DialogCondition):
            raise ValueError(f"interstitial {self.id!r} answers a dialog, so `when` is a dialog")
        return self


class LoginForm(_Model):
    session: str
    """The session id capabilities name in `requires.session`."""
    route: str
    username_field: Target
    password_field: Target
    submit: Target
    signed_in: Signature
    rejected: Signature
    username_env: str
    password_env: str
    timeout_ms: int = Field(default=15_000, ge=1_000, le=120_000)


class AppProfile(_Model):
    schema_version: Literal["understudy.app_profile/v1"] = "understudy.app_profile/v1"
    vendor_app: str
    surface: Literal["web"] = "web"
    description: str
    login: LoginForm
    global_states: list[GlobalState] = Field(default_factory=list)
    interstitials: list[Interstitial] = Field(default_factory=list)
    screenshot_masks: list[str] = Field(default_factory=list)
    """CSS selectors painted over in every screenshot: panels that print names or
    account numbers, which field-level masking does not reach."""
    recon: dict[str, Any] = Field(default_factory=dict)
    """Recorded facts about the app, for people. No code reads this."""

    @model_validator(mode="after")
    def _unique_ids(self) -> AppProfile:
        ids = [s.id for s in self.global_states] + [i.id for i in self.interstitials]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate state ids: {', '.join(duplicates)}")
        return self

    def dialog_rules(self) -> list[tuple[str | None, Literal["accept", "dismiss"]]]:
        return [
            (i.when.message_regex, i.dialog)
            for i in self.interstitials
            if i.dialog is not None and isinstance(i.when, DialogCondition)
        ]


def load_profile(path: Path) -> AppProfile:
    return AppProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
