"""Policy documents: the vendor default (one per app profile) and the tenant overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Member = Literal[
    "navigate",
    "read_page",
    "get_page_text",
    "screenshot",
    "zoom",
    "left_click",
    "form_input",
    "type",
    "key",
    "scroll",
    "scroll_to",
    "wait",
    "hover",
]
"""The browser toolset members a policy may enable. Members outside this list
(find, tabs, drag, raw mouse, javascript_exec, file_upload) cannot be enabled at all."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouteRules(_Model):
    """Globs over the path relative to the tenant's base URL, matched case-insensitively
    after path parameters (`;jsessionid=…`) are stripped. Deny wins over allow."""

    allow: list[str] = Field(default_factory=lambda: ["/*"])
    deny: list[str] = Field(default_factory=list)


class CommitRules(_Model):
    button_names: list[str] = Field(default_factory=list)
    """Full-match regexes on the accessible name of a button. Links never match:
    a nav link may share a commit button's name (ParaBank's does)."""
    routes: list[str] = Field(default_factory=list)
    """Route globs that move money or change state. A request to one is only let
    through while an approved commit action is in progress."""


class ApprovalModes(_Model):
    discovery: Literal["interactive"] = "interactive"
    replay: Literal["token_or_interactive", "interactive"] = "token_or_interactive"


class Policy(_Model):
    schema_version: Literal["understudy.policy/v1"] = "understudy.policy/v1"
    vendor_app: str
    routes: RouteRules = Field(default_factory=RouteRules)
    actions: list[Member]
    keys: list[str] = Field(default_factory=list)
    commit: CommitRules = Field(default_factory=CommitRules)
    approval: ApprovalModes = Field(default_factory=ApprovalModes)
    discovery_sandbox_only: bool = True


class PolicyOverlay(_Model):
    """A tenant's tightening of the vendor policy.

    Every field adds a restriction or removes a permission. No field could widen
    one, so an overlay cannot loosen the vendor policy by construction rather than
    by a check that might be forgotten.
    """

    deny_routes: list[str] = Field(default_factory=list)
    remove_actions: list[Member] = Field(default_factory=list)
    remove_keys: list[str] = Field(default_factory=list)
    commit_button_names: list[str] = Field(default_factory=list)
    commit_routes: list[str] = Field(default_factory=list)

    def apply(self, policy: Policy) -> Policy:
        return policy.model_copy(
            update={
                "routes": RouteRules(
                    allow=list(policy.routes.allow),
                    deny=[*policy.routes.deny, *self.deny_routes],
                ),
                "actions": [a for a in policy.actions if a not in self.remove_actions],
                "keys": [k for k in policy.keys if k not in self.remove_keys],
                "commit": CommitRules(
                    button_names=[*policy.commit.button_names, *self.commit_button_names],
                    routes=[*policy.commit.routes, *self.commit_routes],
                ),
            }
        )


def load_policy(path: Path) -> Policy:
    return Policy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
