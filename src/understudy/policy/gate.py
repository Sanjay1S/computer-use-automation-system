"""The single policy gate. Discovery's SDK hooks, the browser's request guard,
replay's per-step check and lint's drift check all ask this one object, so the
model, the engine and the network layer can never disagree about what is allowed.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from understudy.artifact.lint import PolicyView
from understudy.artifact.schema import Action, Effect
from understudy.policy.models import Member, Policy, load_policy
from understudy.surface.base import ElementFacts
from understudy.tenant import TenantBinding

Verdict = Literal["allow", "deny", "require_approval"]
Mode = Literal["discovery", "replay"]

READ_MEMBERS = frozenset(
    {"navigate", "read_page", "get_page_text", "screenshot", "zoom", "scroll", "scroll_to"}
    | {"wait", "hover"}
)

REPLAY_MEMBER: dict[Action, Member] = {
    "navigate": "navigate",
    "click": "left_click",
    "fill": "form_input",
    "select": "form_input",
    "check": "form_input",
    "press": "key",
    "extract": "get_page_text",
    "wait_for": "wait",
}
"""Which toolset member a replay step's action counts as, so one action list in the
policy governs both modes."""

KEY_ALIASES = {"return": "Enter", "esc": "Escape", "tab": "Tab", "backspace": "Backspace"}

_PATH_PARAMS = re.compile(r";[^/]*")


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    rule: str
    reason: str
    effect: Effect = "read"

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"


def _allow(effect: Effect = "read") -> Decision:
    return Decision("allow", "ok", "allowed", effect)


class PolicyGate(PolicyView):
    def __init__(self, policy: Policy, tenant: TenantBinding) -> None:
        if policy.vendor_app != tenant.vendor_app:
            raise ValueError(
                f"policy is for {policy.vendor_app!r}, tenant {tenant.id!r} runs "
                f"{tenant.vendor_app!r}"
            )
        self.tenant = tenant
        self.policy = tenant.policy.apply(policy)
        self._commit_names = [re.compile(p) for p in self.policy.commit.button_names]

    @classmethod
    def load(cls, policy_path: Path, tenant: TenantBinding) -> PolicyGate:
        return cls(load_policy(policy_path), tenant)

    # --- URLs ------------------------------------------------------------------

    def check_url(self, url: str) -> Decision:
        if url == "about:blank":
            return _allow()
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return Decision("deny", "scheme", f"scheme {parts.scheme or '(none)'!r} is not allowed")
        if parts.scheme != self.tenant.scheme:
            return Decision("deny", "scheme", f"{parts.scheme} is not the tenant's scheme")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            return Decision("deny", "host", "malformed port")
        if (parts.hostname or "").lower() != self.tenant.host or port != self.tenant.port:
            return Decision("deny", "host", f"host {parts.netloc!r} is not the tenant's host")
        route = self._route(parts.path)
        if route is None:
            return Decision("deny", "routes.base_path", "outside the app's base path")
        return self._check_route(route)

    def route_of(self, url: str) -> str | None:
        """The path relative to the tenant's base URL, normalised, or None if outside it."""
        parts = urlsplit(url)
        if (parts.hostname or "").lower() != self.tenant.host:
            return None
        return self._route(parts.path)

    def is_commit_request(self, url: str) -> bool:
        route = self.route_of(url)
        return route is not None and _matches(route, self.policy.commit.routes)

    def allows_route(self, route: str) -> bool:
        normalised = self._normalise(route)
        return self._check_route(normalised).allowed

    def allows_locator_value(self, value: str) -> bool:
        """A locator naming a denied page (`a[href*='admin.htm']`) is a standing
        instruction to go there. Only literal deny entries can be recognised this way."""
        lowered = value.lower()
        for pattern in self.policy.routes.deny:
            leaf = pattern.rsplit("/", 1)[-1].lower()
            if leaf and not any(c in leaf for c in "*?[") and leaf in lowered:
                return False
        return True

    def chromium_args(self) -> list[str]:
        """Resolve only the tenant's host. Request interception never sees redirect
        hops (Playwright calls route handlers for the first URL only), so without
        this an open redirect on an allowed host would reach any other host.

        Host names only: a redirect to the same host on another port or scheme still
        goes out (with the host's cookies) and is only detected afterwards. Closing
        that needs an egress proxy, which is design-only here (plan §8 limits)."""
        return [f"--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE {self.tenant.host}"]

    # --- actions ---------------------------------------------------------------

    @property
    def allowed_members(self) -> tuple[Member, ...]:
        return tuple(self.policy.actions)

    def check_member(self, member: str) -> Decision:
        if member not in self.policy.actions:
            return Decision("deny", "actions", f"member {member!r} is not enabled")
        return _allow()

    def check_key(self, key: str) -> Decision:
        if normalise_key(key) not in self.policy.keys:
            return Decision("deny", "keys", f"key {key!r} is not allowed")
        return _allow("input")

    def classify(
        self, member: str, element: ElementFacts | None, *, key: str | None = None
    ) -> Effect:
        """Effect is judged on the resolved element, never on the member name: a
        purchase is just a left_click, and Enter in a payment form submits it."""
        if member in READ_MEMBERS:
            return "read"
        if member in ("type", "form_input"):
            return "input"
        if member == "key":
            enter = normalise_key(key or "") == "Enter"
            if (
                element is not None
                and enter
                and (self._element_commits(element) or self._form_commits(element))
            ):
                return "commit"
            return "input"
        if member == "left_click":
            if element is None:
                return "commit"  # nothing to judge it by, so it is judged the riskiest way
            if self._element_commits(element):
                return "commit"
            if element.is_form_control or element.is_button:
                return "input"
            return "read"
        return "commit"

    def check_action(
        self,
        member: str,
        element: ElementFacts | None = None,
        *,
        key: str | None = None,
        mode: Mode,
        provider: bool = False,
    ) -> Decision:
        """`provider`: the session provider is acting, which alone may fill credential
        fields. Every other rule applies to it as to anyone."""
        decision = self.check_member(member)
        if not decision.allowed:
            return decision
        if member == "key":
            decision = self.check_key(key or "")
            if not decision.allowed:
                return decision
        if (
            mode == "discovery"
            and not provider
            and member in ("type", "form_input")
            and element
            and element.is_password
        ):
            return Decision(
                "deny",
                "password",
                "credentials only ever pass through the session provider",
                "input",
            )
        effect = self.classify(member, element, key=key)
        if effect == "commit":
            return Decision("require_approval", "commit", "this action commits", "commit")
        return _allow(effect)

    def check_discovery_allowed(self) -> Decision:
        if self.policy.discovery_sandbox_only and not self.tenant.sandbox:
            return Decision(
                "deny",
                "discovery",
                f"tenant {self.tenant.id!r} is not a sandbox; discovery sends page content "
                "to a model",
            )
        return _allow()

    # --- internals -------------------------------------------------------------

    def _element_commits(self, element: ElementFacts) -> bool:
        # A hidden or unnamed button has no accessible name; its label still counts.
        label = element.name or element.value_attr or element.text
        if element.is_button and label and self._commit_name(label):
            return True
        submits = element.is_button and element.input_type != "button"
        if submits and element.form_action and self.is_commit_request(element.form_action):
            return True
        return bool(element.href and self.is_commit_request(element.href))

    def _form_commits(self, element: ElementFacts) -> bool:
        if any(self._commit_name(name) for name in element.form_buttons):
            return True
        return bool(element.form_action and self.is_commit_request(element.form_action))

    def _commit_name(self, name: str) -> bool:
        normalised = " ".join(name.split())
        return any(pattern.fullmatch(normalised) for pattern in self._commit_names)

    def _check_route(self, route: str) -> Decision:
        if _matches(route, self.policy.routes.deny):
            return Decision("deny", "routes.deny", f"route {route!r} is denied")
        if not _matches(route, self.policy.routes.allow):
            return Decision("deny", "routes.allow", f"route {route!r} is not allowed")
        return _allow()

    def _route(self, path: str) -> str | None:
        normalised = self._normalise(path)
        base = self.tenant.base_path.lower()
        lowered = normalised.lower()
        if not base:
            return normalised
        if lowered == base:
            return "/"
        if lowered.startswith(base + "/"):
            return normalised[len(base) :]
        return None

    @staticmethod
    def _normalise(path: str) -> str:
        """Decode, drop path parameters and resolve dot segments, so that
        `/a/%2e%2e/admin.htm;jsessionid=X` is judged as `/admin.htm`."""
        decoded = _PATH_PARAMS.sub("", unquote(path))
        normalised = posixpath.normpath("/" + decoded.lstrip("/"))
        return "/" if normalised in (".", "//") else normalised


def normalise_key(key: str) -> str:
    return KEY_ALIASES.get(key.strip().lower(), key.strip())


def _matches(route: str, patterns: list[str]) -> bool:
    lowered = route.lower()
    return any(fnmatch.fnmatchcase(lowered, pattern.lower()) for pattern in patterns)
