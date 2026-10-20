"""The Playwright web surface. Discovery and replay reach the browser only through
this class, so policy, redaction and ref handling cannot be bypassed by a caller.

Policy is enforced at three layers here:
- `navigate` checks the URL before any request is sent;
- `act` judges the resolved element and refuses a commit unless the caller has
  opened `commit_window()`, which it does only after approval;
- a context-wide request guard checks every request the page makes, and blocks
  requests to commit routes outside a commit window, so a commit reached by some
  other element or by page script is stopped at the network.

Request interception never sees redirect hops. Hops to another host name are
stopped by launching Chromium with `PolicyGate.chromium_args()`. Any other hop the
policy forbids (a denied route, a commit route, the same host on another port or
scheme) is detected from the request event after it has gone out, and the page is
treated as off policy until the next navigation.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from playwright.sync_api import (
    ConsoleMessage,
    Dialog,
    ElementHandle,
    Page,
    Request,
    Response,
    Route,
)
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator as PwLocator
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from understudy.artifact import signatures as sig
from understudy.artifact.schema import Extraction, Fingerprint, Target
from understudy.evidence.run import Event
from understudy.policy.gate import REPLAY_MEMBER, Decision, Mode, PolicyGate
from understudy.redaction.redactor import Redactor
from understudy.surface import capture, locators
from understudy.surface.base import (
    ActAction,
    ActionBlocked,
    ActionFailed,
    ActionTimeout,
    DialogBlocking,
    ElementFacts,
    InfraError,
    Observation,
    OptionAbsent,
    PolicyBlocked,
    ProbeError,
    Resolution,
    StaleRef,
    SurfaceError,
    TargetChanged,
)
from understudy.surface.snapshot import RefTable, scrub_for_model

CONTEXT_OPTIONS: dict[str, Any] = {
    "viewport": {"width": 1280, "height": 800},
    "locale": "en-US",
    "timezone_id": "UTC",
    "reduced_motion": "reduce",
    "service_workers": "block",
}
"""Fixed so replay does not vary with the machine. Service workers are blocked
because their requests bypass the request guard."""

_PARAM = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

_OPTIONS_JS = """
(s) => s.tagName !== "SELECT" ? null : Array.from(s.options).map((o) => ({
  value: o.value, label: (o.label || o.textContent || "").replace(/\\s+/g, " ").trim(),
  disabled: o.disabled,
}))
"""

_PASSWORD_VALUES_JS = """
() => Array.from(document.querySelectorAll("input[type=password]")).map((i) => i.value)
"""

_SELECTED_JS = """
(els) => els.flatMap((el) => {
  const norm = (s) => (s || "").replace(/\\s+/g, " ").trim();
  if (el.tagName === "SELECT") {
    return Array.from(el.selectedOptions).map((o) => [o.value, norm(o.label || o.textContent)]);
  }
  if ((el.type === "radio" || el.type === "checkbox") && el.checked) {
    const label = el.labels && el.labels.length ? norm(el.labels[0].innerText) : "";
    return [[el.value, label]];
  }
  return [];
})
"""


@dataclass(frozen=True)
class PolicyViolation:
    url: str
    """Raw. Redact before writing."""
    rule: str
    reason: str
    navigation: bool
    via: Literal["request", "redirect"]


class WebSurface:
    def __init__(self, page: Page, gate: PolicyGate, *, mode: Mode, run_id: str = "adhoc") -> None:
        """Install any other route handlers (fixtures, fault injection) before
        constructing this. Playwright runs page routes before context routes, and
        each set last-registered first, so the guard goes on both: on the page it
        runs ahead of every earlier page route, on the context it covers pages the
        app opens later."""
        self.page = page
        self.gate = gate
        self.mode: Mode = mode
        self.run_id = run_id
        self.refs = RefTable()
        self._dialogs: list[Dialog] = []
        self._console: list[str] = []
        self._violations: list[PolicyViolation] = []
        self._tainted: PolicyViolation | None = None
        self._commit_budget: dict[str, int] | None = None
        self._last_status: int | None = None
        page.on("dialog", self._on_dialog)
        page.on("console", self._on_console)
        page.on("pageerror", lambda error: self._console.append(f"pageerror: {error}"))
        page.on("response", self._on_response)
        page.context.on("request", self._on_request)
        page.context.route("**/*", self._context_guard)
        page.route("**/*", self._guard)

    # --- observation -------------------------------------------------------------

    def observe(self) -> Observation:
        self._check_tainted()
        if self._dialogs:
            dialog = self._dialogs[0]
            return Observation(
                url=self.page.url,
                title="",
                snapshot="",
                status=self._last_status,
                dialog=f"{dialog.type}: {dialog.message}",
            )
        snapshot = self.refs.rewrite(
            scrub_for_model(self.page.aria_snapshot(mode="ai"), self._password_values())
        )
        return Observation(
            url=self.page.url,
            title=self.page.title(),
            snapshot=snapshot,
            status=self._last_status,
        )

    def page_text(self) -> str:
        self._ready()
        return self.page.locator("body").inner_text()

    def element_for_ref(self, model_ref: str) -> PwLocator:
        node = self.refs.resolve(model_ref)
        self._ready()
        element = self.page.get_by_ref(node.pw_ref)
        if element.count() != 1:
            raise StaleRef(f"{model_ref} no longer resolves; call read_page for fresh refs")
        return element

    def facts(self, element: PwLocator) -> ElementFacts:
        try:
            return locators.read_facts(element)
        finally:
            self._restore_refs()

    def resolve(
        self, target: Target, params: Mapping[str, str], *, require_visible: bool = True
    ) -> Resolution[PwLocator]:
        self._ready()
        try:
            return locators.resolve(self.page, target, params, require_visible=require_visible)
        finally:
            self._restore_refs()

    def synthesize(
        self,
        element: PwLocator,
        *,
        params: Mapping[str, str] | None = None,
        description: str | None = None,
    ) -> locators.Synthesis:
        try:
            return locators.synthesize(
                self.page, element, run_id=self.run_id, params=params, description=description
            )
        finally:
            self._restore_refs()

    # --- actions -----------------------------------------------------------------

    def navigate(self, url: str, *, timeout_ms: int = 30_000) -> int | None:
        decision = self.gate.check_url(url)
        if not decision.allowed:
            raise PolicyBlocked(decision.reason, decision.rule)
        self._tainted = None
        seen = len(self._violations)
        try:
            response = self.page.goto(url, wait_until="load", timeout=timeout_ms)
        except PlaywrightError as error:
            # Only a blocked navigation explains a failed goto; a blocked tracker
            # pixel on a page that then timed out does not.
            caused = [v for v in self._violations[seen:] if v.navigation]
            if caused:
                raise PolicyBlocked(caused[-1].reason, caused[-1].rule) from None
            raise _surface_error(error) from None
        self._check_tainted()
        return response.status if response else None

    def act(
        self,
        action: ActAction,
        element: PwLocator,
        value: str | None = None,
        *,
        fingerprint: Fingerprint | None = None,
        params: Mapping[str, str] | None = None,
        timeout_ms: int = 5_000,
    ) -> None:
        """Acts on the exact element whose facts were checked: the locator is pinned
        to a handle first, so a re-render between resolution and action fails
        instead of landing on whatever now matches. Replay passes the target's
        fingerprint, which is re-checked here against that same element."""
        self._ready()
        try:
            handle = element.element_handle(timeout=timeout_ms)
            facts = self.facts(element)
            same = bool(element.evaluate("(e, h) => e === h", handle))
        except PlaywrightError as error:
            raise _surface_error(error) from None
        if not same:
            raise TargetChanged("the element was replaced while it was being checked")
        if fingerprint is not None:
            problems = locators.fingerprint_mismatches(fingerprint, facts, params or {})
            if problems:
                raise TargetChanged("; ".join(problems))
        decision = self.gate.check_action(
            REPLAY_MEMBER[action],
            facts,
            key=value if action == "press" else None,
            mode=self.mode,
        )
        self._enforce(decision)
        try:
            if action == "click":
                handle.click(timeout=timeout_ms)
            elif action == "fill":
                handle.fill(_required(value, action), timeout=timeout_ms)
            elif action == "select":
                self._select(handle, _required(value, action), timeout_ms)
            elif action == "check":
                handle.set_checked(_truthy(_required(value, action)), timeout=timeout_ms)
            else:
                handle.press(_required(value, action), timeout=timeout_ms)
        except SurfaceError:
            raise
        except PlaywrightError as error:
            raise _surface_error(error) from None

    def read(self, element: PwLocator, extraction: Extraction) -> str:
        """The raw string. Parsing and pattern checks belong to the caller."""
        self._ready()
        try:
            if extraction.read == "value":
                return element.input_value(timeout=2_000)
            if extraction.read == "attr":
                return element.get_attribute(extraction.attr or "", timeout=2_000) or ""
            return element.inner_text(timeout=2_000)
        except PlaywrightError as error:
            raise _surface_error(error) from None

    @contextmanager
    def commit_window(self) -> Iterator[None]:
        """One approved commit: one commit action through `act()` and one request to
        a commit route. A second of either inside the window is refused, so a page
        script cannot ride an approval to commit again. Open it only once the commit
        is approved, and keep it open through the outcome race: the request may
        leave after the click returns. (Binding the approval to the capability,
        inputs and tenant is the control plane's job, M6.)"""
        if self._commit_budget is not None:
            raise RuntimeError("a commit window is already open")
        self._commit_budget = {"actions": 1, "requests": 1}
        try:
            yield
        finally:
            self._commit_budget = None

    # --- probing -----------------------------------------------------------------

    def probe(self, signature: sig.Signature, params: Mapping[str, str] | None = None) -> bool:
        """Evaluate a signature against the page as it is now. Never waits."""
        self._check_tainted()
        try:
            return self._eval(signature, params or {})
        except (PlaywrightError, ValueError, re.error) as error:
            raise ProbeError(str(error).strip().split("\n", 1)[0][:300]) from None

    def _eval(self, node: sig.Signature, params: Mapping[str, str]) -> bool:
        if isinstance(node, sig.AllCondition):
            return all(self._eval(child, params) for child in node.all)
        if isinstance(node, sig.AnyCondition):
            return any(self._eval(child, params) for child in node.any)
        if isinstance(node, sig.NotCondition):
            return not self._eval(node.not_, params)
        if isinstance(node, sig.UrlCondition):
            route = self.gate.route_of(self.page.url)
            return route is not None and re.search(node.path_regex, route) is not None
        if isinstance(node, sig.HttpStatusCondition):
            return self._last_status is not None and node.min <= self._last_status <= node.max
        if isinstance(node, sig.DialogCondition):
            return any(
                node.message_regex is None or re.search(node.message_regex, d.message)
                for d in self._dialogs
            )
        if self._dialogs:
            raise DialogBlocking("a dialog is open; handle it before reading the page")
        if isinstance(node, sig.VisibleCondition):
            return self._matching(node, params).filter(visible=True).count() > 0
        if isinstance(node, sig.AbsentCondition):
            return self._matching(node, params).filter(visible=True).count() == 0
        if isinstance(node, sig.OptionsLoadedCondition):
            target = self.page.locator(f"css={node.target}")
            if target.count() != 1:
                return False
            options = cast(list[dict[str, Any]] | None, target.evaluate(_OPTIONS_JS))
            real = [o for o in options or [] if o["value"] != "" and not o["disabled"]]
            return len(real) >= node.min
        expected = node.value if node.value is not None else params.get(node.param or "")
        if expected is None:
            raise ValueError(f"selected: param {node.param!r} is not bound")
        chosen = cast(
            list[list[str]], self.page.locator(f"css={node.target}").evaluate_all(_SELECTED_JS)
        )
        return any(expected in pair for pair in chosen)

    def _matching(
        self, node: sig.VisibleCondition | sig.AbsentCondition, params: Mapping[str, str]
    ) -> PwLocator:
        root: Page | PwLocator = self.page
        if node.within:
            root = self.page.locator(f"css={_fill(node.within, params, selector=True)}")
        parts: list[PwLocator] = []
        if node.role:
            name: str | re.Pattern[str] | None = None
            if node.name_regex:
                name = re.compile(_fill(node.name_regex, params, regex=True))
            elif node.name:
                name = _fill(node.name, params)
            parts.append(
                root.get_by_role(
                    cast(Any, node.role),
                    name=name,
                    exact=node.exact if isinstance(name, str) else None,
                )
            )
        if node.css:
            parts.append(root.locator(f"css={_fill(node.css, params, selector=True)}"))
        if node.text or node.text_regex:
            text: str | re.Pattern[str] = (
                re.compile(_fill(node.text_regex, params, regex=True))
                if node.text_regex
                else _fill(node.text or "", params)
            )
            parts.append(
                root.get_by_text(text, exact=node.exact if isinstance(text, str) else None)
            )
        matched = parts[0]
        for part in parts[1:]:
            matched = matched.and_(part)
        return matched

    # --- dialogs -------------------------------------------------------------------

    @property
    def pending_dialog(self) -> tuple[str, str] | None:
        return (self._dialogs[0].type, self._dialogs[0].message) if self._dialogs else None

    def accept_dialog(self, prompt_text: str | None = None) -> None:
        self._dialogs.pop(0).accept(prompt_text)

    def dismiss_dialog(self) -> None:
        self._dialogs.pop(0).dismiss()

    # --- evidence ------------------------------------------------------------------

    @property
    def console_errors(self) -> list[str]:
        """Raw page output; redact before writing."""
        return list(self._console)

    @property
    def violations(self) -> list[PolicyViolation]:
        return list(self._violations)

    def screenshot(self, path: Path, extra_masks: Sequence[str] = ()) -> Path:
        return capture.screenshot(self.page, path, extra_masks)

    def failure_bundle(
        self,
        directory: Path,
        redactor: Redactor,
        *,
        reason: str,
        events: Sequence[Event] = (),
        extra_masks: Sequence[str] = (),
    ) -> Path:
        try:
            snapshot = "" if self._dialogs else self.page.aria_snapshot()
        except PlaywrightError as error:
            snapshot = f"(no snapshot: {error})"
        finally:
            self._restore_refs()
        return capture.write_failure_bundle(
            self.page,
            directory,
            redactor,
            reason=reason,
            snapshot=snapshot,
            console=self._console,
            events=events,
            extra_masks=extra_masks,
        )

    # --- internals -----------------------------------------------------------------

    def _enforce(self, decision: Decision) -> None:
        if decision.verdict == "deny":
            raise PolicyBlocked(decision.reason, decision.rule)
        if decision.verdict == "require_approval" and not self._spend_commit("actions"):
            raise PolicyBlocked(
                "this action commits and no approved commit is left in this window", "commit"
            )

    def _password_values(self) -> set[str]:
        values: set[str] = set()
        for frame in self.page.frames:
            try:
                values.update(cast(list[str], frame.evaluate(_PASSWORD_VALUES_JS)))
            except PlaywrightError:
                continue
        return values

    def _spend_commit(self, kind: Literal["actions", "requests"]) -> bool:
        if self._commit_budget is None or self._commit_budget[kind] < 1:
            return False
        self._commit_budget[kind] -= 1
        return True

    def _select(self, element: ElementHandle, value: str, timeout_ms: int) -> None:
        """By visible label first, then by value. Labels are what the artifact's
        enum speaks (ParaBank's account type is labelled SAVINGS, valued 1)."""
        options = cast(list[dict[str, Any]] | None, element.evaluate(_OPTIONS_JS))
        if options is None:
            raise ActionFailed("select needs a <select> element")
        real = [o for o in options if o["value"] != "" and not o["disabled"]]
        chosen = [o for o in options if o["label"] == value] or [
            o for o in options if o["value"] == value
        ]
        if not chosen:
            raise OptionAbsent(
                f"none of the {len(real)} loaded options matches the requested value", len(real)
            )
        if len({o["value"] for o in chosen}) > 1:
            raise ActionFailed("more than one option matches the requested value")
        element.select_option(value=chosen[0]["value"], timeout=timeout_ms)

    def _restore_refs(self) -> None:
        """Element-level snapshots replace Playwright's page-level one, stranding every
        ref. Retaking the page snapshot restores them (Playwright keeps refs per
        element), so a ref the model holds, or a ref locator mid-action, keeps working.
        Only needed once refs have been handed out."""
        if self.refs.generation and not self._dialogs:
            self.refs.rewrite(self.page.aria_snapshot(mode="ai"))

    def _ready(self) -> None:
        self._check_tainted()
        if self._dialogs:
            raise DialogBlocking("a dialog is open; handle it before driving the page")

    def _check_tainted(self) -> None:
        if self._tainted is not None:
            raise PolicyBlocked(
                f"the page was taken off policy ({self._tainted.reason})", self._tainted.rule
            )

    def _guard(self, route: Route, request: Request) -> None:
        decision = self._judge(request)
        if decision.allowed:
            route.fallback()
            return
        self._record(request, decision, "request")
        route.abort("blockedbyclient")

    def _context_guard(self, route: Route, request: Request) -> None:
        """Requests from this surface's own page were already judged by the page guard
        (and may have spent the commit budget); judge only other pages'."""
        try:
            own = request.frame.page == self.page
        except PlaywrightError:
            own = False
        if own:
            route.fallback()
            return
        self._guard(route, request)

    def _judge(self, request: Request) -> Decision:
        decision = self.gate.check_url(request.url)
        if (
            decision.allowed
            and self.gate.is_commit_request(request.url)
            and not self._spend_commit("requests")
        ):
            return Decision(
                "deny", "commit_window", "a commit request outside an approved commit action"
            )
        return decision

    def _on_request(self, request: Request) -> None:
        """Redirect hops never reach the guards. They can only be detected here, after
        the hop has gone out, which is why the page is then treated as off policy."""
        if request.redirected_from is None:
            return
        decision = self._judge(request)
        if not decision.allowed:
            self._record(request, decision, "redirect")

    def _record(
        self, request: Request, decision: Decision, via: Literal["request", "redirect"]
    ) -> None:
        violation = PolicyViolation(
            request.url, decision.rule, decision.reason, self._is_main_navigation(request), via
        )
        self._violations.append(violation)
        if violation.navigation:
            self._tainted = violation

    def _is_main_navigation(self, request: Request) -> bool:
        try:
            return request.is_navigation_request() and request.frame == self.page.main_frame
        except PlaywrightError:
            return False

    def _on_response(self, response: Response) -> None:
        try:
            if response.request.is_navigation_request() and response.frame == self.page.main_frame:
                self._last_status = response.status
        except PlaywrightError:
            return

    def _on_dialog(self, dialog: Dialog) -> None:
        self._dialogs.append(dialog)

    def _on_console(self, message: ConsoleMessage) -> None:
        if message.type == "error":
            self._console.append(message.text)


def _surface_error(error: PlaywrightError) -> SurfaceError:
    text = str(error)
    first = text.strip().split("\n", 1)[0][:300]
    if "intercepts pointer events" in text:
        return ActionBlocked(first)
    if "not attached to the DOM" in text:
        return TargetChanged(first)
    if any(s in text for s in ("has been closed", "Target crashed", "net::ERR_")):
        return InfraError(first)
    if isinstance(error, PlaywrightTimeoutError):
        return ActionTimeout(first)
    return ActionFailed(first)


def _fill(
    text: str, params: Mapping[str, str], *, regex: bool = False, selector: bool = False
) -> str:
    """Substitute declared params only: `\\d{3}` in a regex is not a param. A value
    that could break out of a selector is refused, as the resolver refuses it."""

    def one(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in params:
            return match.group(0)
        if selector and locators.UNSAFE_IN_SELECTOR.search(params[name]):
            raise ValueError(f"param {name!r} has characters that cannot go in a selector")
        return re.escape(params[name]) if regex else params[name]

    return _PARAM.sub(one, text)


def _required(value: str | None, action: str) -> str:
    if value is None:
        raise ActionFailed(f"{action} needs a value")
    return value


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes", "on", "checked")
