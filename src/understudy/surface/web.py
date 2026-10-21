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

import contextlib
import re
import time
from collections import deque
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
from understudy.control.plane import ControlPlane, Lease
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

# The element that would receive a pointer event at the target's centre. Playwright
# checks this only for clicks; select_option and fill act straight through a modal.
# Asked of the target's own root, so an element inside a shadow tree is not "covered"
# by its host; a label for the control (or a child of one) passes its clicks on.
_COVERED_JS = """
(e) => {
  const r = e.getBoundingClientRect();
  if (!r.width || !r.height) return null;
  const root = e.getRootNode();
  const at = typeof root.elementFromPoint === "function" ? root : e.ownerDocument;
  const top = at.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  if (!top || top === e || e.contains(top)) return null;
  const label = top.closest("label");
  if (label && label.control === e) return null;
  return top.tagName.toLowerCase() + (top.id ? "#" + top.id : "");
}
"""

# A selector for the element at a point (or the focused one) that names exactly that
# element: a child-combinator path of nth-of-type steps from <html>. The nearest
# clickable ancestor stands in for a span inside a button. Frames are not entered.
_PATH_TO_JS = """
(point) => {
  let el = point ? document.elementFromPoint(point[0], point[1]) : document.activeElement;
  if (!el || el === document.body || el === document.documentElement) return null;
  if (point) {
    el = el.closest("a, button, input, select, textarea, label, summary, [role=button], "
      + "[role=link], [role=checkbox], [role=radio], [onclick]") || el;
  }
  if (el.tagName === "IFRAME" || el.tagName === "FRAME") return "frame";
  const parts = [];
  for (let n = el; n && n !== document.documentElement; n = n.parentElement) {
    if (!n.parentElement) return null;
    const same = Array.from(n.parentElement.children).filter((c) => c.tagName === n.tagName);
    parts.unshift(`${n.tagName.toLowerCase()}:nth-of-type(${same.indexOf(n) + 1})`);
  }
  return "html > " + parts.join(" > ");
}
"""

_MARKUP_SIZE_JS = "() => document.documentElement ? document.documentElement.outerHTML.length : 0"

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


DialogAnswer = Literal["accept", "dismiss"]


@dataclass(frozen=True)
class DialogSeen:
    type: str
    message: str
    """Raw page text. Redact before writing."""
    answer: DialogAnswer
    known: bool
    """Answered by a rule the caller gave, rather than by the default."""

    def __str__(self) -> str:
        return f"{self.type}: {self.message}"


@dataclass(frozen=True)
class PolicyViolation:
    url: str
    """Raw. Redact before writing."""
    rule: str
    reason: str
    navigation: bool
    via: Literal["request", "redirect"]


class WebSurface:
    def __init__(
        self,
        page: Page,
        gate: PolicyGate,
        *,
        mode: Mode,
        run_id: str = "adhoc",
        dialog_rules: Sequence[tuple[str | None, DialogAnswer]] = (),
        control: ControlPlane | None = None,
    ) -> None:
        """Install any other route handlers (fixtures, fault injection) before
        constructing this. Playwright runs page routes before context routes, and
        each set last-registered first, so the guard goes on both: on the page it
        runs ahead of every earlier page route, on the context it covers pages the
        app opens later.

        Native dialogs are answered the moment they open: while one is open every
        page read blocks, whatever its timeout, so one left open for later would
        freeze whatever call was in flight. A dialog matching a `dialog_rules` entry
        (message regex, or None for any) gets that answer; any other is dismissed,
        the non-committal answer, and then blocks driving the page until someone
        acknowledges it.

        With a `control` plane, every actuation (navigate, act, hover, scroll) must
        carry the current lease, or it raises ControlLost before touching the page."""
        self.page = page
        self.control = control
        self._acting: deque[tuple[float, float]] = deque(maxlen=500)
        """Wall-clock windows in which automation was delivering input, so the human
        recorder can tell automation's own events from a person's."""
        self.gate = gate
        self.mode: Mode = mode
        self._provider = 0
        """Depth of `session_provider()` scopes. A count, so a nested one cannot end the outer."""
        self.run_id = run_id
        self.refs = RefTable()
        self._dialog_rules = [(re.compile(r) if r else None, a) for r, a in dialog_rules]
        self._dialogs: list[DialogSeen] = []
        """Answered, not yet acknowledged."""
        self._dialog_history: list[DialogSeen] = []
        self._console: list[str] = []
        self._violations: list[PolicyViolation] = []
        self._tainted: PolicyViolation | None = None
        self._commit_budget: dict[str, int] | None = None
        self._commit_requests = 0
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
        snapshot = self.refs.rewrite(
            scrub_for_model(self.page.aria_snapshot(mode="ai"), self._password_values())
        )
        dialog = self._dialogs[0] if self._dialogs else None
        return Observation(
            url=self.page.url,
            title=self.page.title(),
            snapshot=snapshot,
            status=self._last_status,
            dialog=f"{dialog} ({dialog.answer}ed)" if dialog else None,
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

    def navigate(
        self, url: str, *, timeout_ms: int = 30_000, lease: Lease | None = None
    ) -> int | None:
        self._fence(lease)
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
        lease: Lease | None = None,
    ) -> None:
        """Acts on the exact element whose facts were checked: the locator is pinned
        to a handle first, so a re-render between resolution and action fails
        instead of landing on whatever now matches. Replay passes the target's
        fingerprint, which is re-checked here against that same element.

        An element covered by something else (a modal, an overlay) is refused for
        every action, not only clicks. An error raised by the browser call itself has
        `dispatched` set: the input may have reached the page."""
        self._fence(lease)
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
            provider=self._provider > 0,
        )
        self._enforce(decision, spend=False)
        try:
            handle.scroll_into_view_if_needed(timeout=timeout_ms)
            cover = cast(str | None, handle.evaluate(_COVERED_JS))
        except PlaywrightError as error:
            raise _surface_error(error) from None
        if cover:
            raise ActionBlocked(f"{cover} covers the target and would receive the input")
        self._enforce(decision)
        started = time.time()
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
            mapped = _surface_error(error)
            mapped.dispatched = True
            raise mapped from None
        finally:
            self._acting.append((started, time.time()))

    def automation_at(self, at: float, *, slack_s: float = 0.5) -> bool:
        """Whether automation was delivering input at this wall-clock time. The slack
        covers events the page reports just after the call returns."""
        return any(start - 0.05 <= at <= end + slack_s for start, end in self._acting)

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

    def current_value(self, element: PwLocator) -> str:
        self._ready()
        try:
            return element.input_value(timeout=2_000)
        except PlaywrightError as error:
            raise _surface_error(error) from None

    def selected(self, element: PwLocator) -> tuple[str, str] | None:
        """A select's (or checked radio's) chosen option as (value, label)."""
        self._ready()
        try:
            # One element, by evaluate: evaluate_all finds nothing through a ref locator.
            chosen = cast(list[list[str]], element.evaluate(f"(el) => ({_SELECTED_JS})([el])"))
        except PlaywrightError as error:
            raise _surface_error(error) from None
        return (chosen[0][0], chosen[0][1]) if chosen else None

    def settle(self, *, quiet_ms: int = 300, max_ms: int = 2_000, poll_ms: int = 100) -> None:
        """Wait until the page's markup has not changed for `quiet_ms`, or `max_ms` has
        passed. Discovery uses it to record a page as the model will next see it;
        replay never does (it waits on signatures, not on quiet)."""
        deadline = time.monotonic() + max_ms / 1000
        last: object = None
        quiet_since = time.monotonic()
        while time.monotonic() < deadline:
            try:
                now: object = (self.page.url, self.page.evaluate(_MARKUP_SIZE_JS))
            except PlaywrightError:
                now = None  # mid-navigation: not settled
            if now is None or now != last:
                last, quiet_since = now, time.monotonic()
            elif time.monotonic() - quiet_since >= quiet_ms / 1000:
                return
            self.pause(poll_ms)

    def element_at(self, x: float, y: float) -> PwLocator:
        """The element a click at this viewport point would land on (its nearest
        clickable ancestor), as a locator naming exactly it. Lets a coordinate action
        be judged and recorded by element, like a ref action."""
        return self._path_locator([x, y], "at that point")

    def focused_element(self) -> PwLocator | None:
        """The focused element, where keys and typed text go; None when it is the page."""
        try:
            return self._path_locator(None, "focused")
        except ActionFailed:
            return None

    def hover(
        self, element: PwLocator, *, timeout_ms: int = 5_000, lease: Lease | None = None
    ) -> None:
        self._fence(lease)
        self._ready()
        try:
            element.hover(timeout=timeout_ms)
        except PlaywrightError as error:
            raise _surface_error(error) from None

    def scroll_into_view(
        self, element: PwLocator, *, timeout_ms: int = 5_000, lease: Lease | None = None
    ) -> None:
        self._fence(lease)
        self._ready()
        try:
            element.scroll_into_view_if_needed(timeout=timeout_ms)
        except PlaywrightError as error:
            raise _surface_error(error) from None

    def wheel(
        self, x: float, y: float, dx: float, dy: float, *, lease: Lease | None = None
    ) -> None:
        self._fence(lease)
        self._ready()
        try:
            self.page.mouse.move(x, y)
            self.page.mouse.wheel(dx, dy)
        except PlaywrightError as error:
            raise _surface_error(error) from None

    def screenshot_png(
        self, *, clip: Mapping[str, float] | None = None, extra_masks: Sequence[str] = ()
    ) -> bytes:
        """A screenshot in memory, for the model rather than for filing (see
        `capture.screenshot_png` for what is masked)."""
        self._ready()
        try:
            return capture.screenshot_png(self.page, clip=clip, extra_masks=extra_masks)
        except PlaywrightError as error:
            raise _surface_error(error) from None

    def pause(self, ms: int) -> None:
        """Wait while the page's events (routes, dialogs, responses) keep being
        dispatched. A Python sleep would hold them until the next Playwright call."""
        try:
            self.page.wait_for_timeout(ms)
        except PlaywrightError as error:
            raise _surface_error(error) from None

    @property
    def commit_requests(self) -> int:
        """Requests to commit routes let through so far. Evidence for whoever has to
        decide what a commit with an unknown outcome did."""
        return self._commit_requests

    @contextmanager
    def session_provider(self) -> Iterator[None]:
        """Actions taken by the session provider, which alone may fill credential fields.
        In discovery the model may never type into a password field; signing in before
        the model starts is what keeps credentials away from it."""
        self._provider += 1
        try:
            yield
        finally:
            self._provider -= 1

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
        except PlaywrightError as error:
            mapped = _surface_error(error)
            if isinstance(mapped, InfraError):
                raise mapped from None
            raise ProbeError(str(mapped)) from None
        except (ValueError, re.error) as error:
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
        """The first unacknowledged dialog no rule knew, as (type, message)."""
        unknown = next((d for d in self._dialogs if not d.known), None)
        return (unknown.type, unknown.message) if unknown else None

    @property
    def dialogs(self) -> list[DialogSeen]:
        """Answered and not yet acknowledged."""
        return list(self._dialogs)

    @property
    def dialog_history(self) -> list[DialogSeen]:
        """Every dialog the page showed, acknowledged or not."""
        return list(self._dialog_history)

    def acknowledge_dialogs(
        self, *, matching: Sequence[str | None] | None = None, known_only: bool = False
    ) -> list[DialogSeen]:
        """Mark dialogs as dealt with, so they stop matching probes and blocking.
        `matching`: only those whose message matches one of these regexes (None: any)."""

        def selected(d: DialogSeen) -> bool:
            if known_only and not d.known:
                return False
            return matching is None or any(r is None or re.search(r, d.message) for r in matching)

        done = [d for d in self._dialogs if selected(d)]
        self._dialogs = [d for d in self._dialogs if d not in done]
        return done

    # --- evidence ------------------------------------------------------------------

    @property
    def console_errors(self) -> list[str]:
        """Raw page output; redact before writing."""
        return list(self._console)

    @property
    def violations(self) -> list[PolicyViolation]:
        return list(self._violations)

    @property
    def off_policy(self) -> PolicyViolation | None:
        """The violation that took the page off policy, until the next navigation."""
        return self._tainted

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
            snapshot = self.page.aria_snapshot()
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
            dialog="\n".join(f"{d} ({d.answer}ed)" for d in self._dialog_history)
            if self._dialog_history
            else None,
        )

    # --- internals -----------------------------------------------------------------

    def _fence(self, lease: Lease | None) -> None:
        if self.control is not None:
            self.control.check(lease)

    def _enforce(self, decision: Decision, *, spend: bool = True) -> None:
        """`spend=False` only checks, so a refusal comes before anything touches the page
        and the commit is spent only once the action is about to be delivered."""
        if decision.verdict == "deny":
            raise PolicyBlocked(decision.reason, decision.rule)
        if decision.verdict != "require_approval":
            return
        budget = self._commit_budget
        available = budget is not None and budget["actions"] > 0
        if not available or (spend and not self._spend_commit("actions")):
            raise PolicyBlocked(
                "this action commits and no approved commit is left in this window", "commit"
            )

    def _path_locator(self, point: list[float] | None, what: str) -> PwLocator:
        self._ready()
        try:
            path = cast(str | None, self.page.evaluate(_PATH_TO_JS, point))
        except PlaywrightError as error:
            raise _surface_error(error) from None
        if path is None:
            raise ActionFailed(f"no element is {what}")
        if path == "frame":
            raise ActionFailed(f"the element {what} is inside a frame; use a ref from read_page")
        element = self.page.locator(f"css={path}")
        if element.count() != 1:
            raise TargetChanged(f"the element {what} changed while it was being found")
        return element

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
        if self.refs.generation:
            self.refs.rewrite(self.page.aria_snapshot(mode="ai"))

    def _ready(self) -> None:
        self._check_tainted()
        if self.pending_dialog is not None:
            raise DialogBlocking(
                "an unexpected dialog appeared (and was dismissed); acknowledge it before "
                "driving the page"
            )

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
        if decision.allowed and self.gate.is_commit_request(request.url):
            if not self._spend_commit("requests"):
                return Decision(
                    "deny", "commit_window", "a commit request outside an approved commit action"
                )
            self._commit_requests += 1
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
        # While a person holds the session no rule answers for them: only the
        # non-committal default, which keeps the page readable.
        held = self.control is not None and self.control.state == "HUMAN"
        rule = (
            None
            if held
            else next(
                (a for r, a in self._dialog_rules if r is None or r.search(dialog.message)), None
            )
        )
        answer: DialogAnswer = rule or "dismiss"
        with contextlib.suppress(PlaywrightError):  # already answered, or its page is gone
            if answer == "accept":
                dialog.accept()
            else:
                dialog.dismiss()
        seen = DialogSeen(dialog.type, dialog.message, answer, rule is not None)
        self._dialogs.append(seen)
        self._dialog_history.append(seen)

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
