"""The discovery driver: Anthropic's browser use toolset, executed on our WebSurface.

The SDK base class supplies the seams; each is wired to the one policy gate:
- `configs` comes from the policy's enabled members, so the model is never offered
  a member the policy would refuse (and the SDK refuses it if called anyway);
- `url_policy` is the gate's URL check;
- `confirm` resolves the element a call would act on and asks the gate about that
  element, not the member name: a purchase is just a left_click. A commit goes to a
  person, and only an approved one gets a commit window;
- the `execute` override records every call: the page before and after, the
  element with verified locators, and the value bound to a goal param.

Every action still goes through `WebSurface.act`, so the gate is asked again there,
and the browser's request guard sits under both.
"""

from __future__ import annotations

import base64
import contextlib
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast, get_args
from urllib.parse import urljoin

from anthropic.tools import ToolError
from anthropic.tools.browser import (
    BetaAbstractBrowserToolset20260801,
    BetaBrowserMemberResult,
    BetaBrowserNavigateResult,
    BetaBrowserState,
    BetaConfirmContext,
    BetaDialogDismissed,
    BetaNavigationRefused,
    BetaScreenshotResult,
    BetaToolsetCallContext,
    BetaURLContext,
)
from anthropic.types.beta import (
    BetaBrowserCoordinateTarget,
    BetaBrowserFormInputInput,
    BetaBrowserGetPageTextInput,
    BetaBrowserHoverInput,
    BetaBrowserKeyInput,
    BetaBrowserLeftClickInput,
    BetaBrowserMemberInput,
    BetaBrowserMemberName,
    BetaBrowserNavigateInput,
    BetaBrowserReadPageInput,
    BetaBrowserRefTarget,
    BetaBrowserScreenshotInput,
    BetaBrowserScrollInput,
    BetaBrowserScrollToInput,
    BetaBrowserToolsetConfigsParam,
    BetaBrowserTypeInput,
    BetaBrowserWaitInput,
    BetaBrowserZoomInput,
)
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator as PwLocator

from understudy.control.plane import Lease
from understudy.discovery.trajectory import (
    ACTION_MEMBERS,
    ElementRecord,
    PageState,
    RecordedCall,
    Recorder,
    StepValue,
    elapsed_ms,
)
from understudy.policy.gate import Decision, normalise_key
from understudy.redaction.detectors import JSESSIONID
from understudy.surface.base import ElementFacts, Observation, PolicyBlocked, SurfaceError
from understudy.surface.snapshot import parse_line
from understudy.surface.web import WebSurface

IMPLEMENTED = frozenset(
    {"navigate", "read_page", "get_page_text", "screenshot", "zoom", "left_click"}
    | {"form_input", "type", "key", "scroll", "scroll_to", "wait", "hover"}
)
"""The members this driver serves. The policy can only narrow it."""

TAB_ID = "tab_1"
"""One tab per run: the surface drives a single page."""

MAX_WAIT_S = 10.0
MAX_REPEAT = 10
SCROLL_STEP_PX = 100

INTERACTIVE_ROLES = frozenset(
    {"button", "link", "textbox", "searchbox", "combobox", "listbox", "option", "checkbox"}
    | {"radio", "switch", "slider", "spinbutton", "tab", "menuitem", "heading"}
)


@dataclass(frozen=True)
class ApprovalRequest:
    """What a person sees before a commit runs. No values: the target and the page."""

    member: str
    target: str
    url: str
    reason: str
    tool_use_id: str | None


Approver = Callable[[ApprovalRequest], bool]


def refuse_all(request: ApprovalRequest) -> bool:
    return False


def member_configs(allowed: frozenset[str]) -> BetaBrowserToolsetConfigsParam:
    """Every member outside `allowed` is switched off on the wire. The driver
    overrides `execute`, so the SDK counts every member as served; this is what
    keeps the rest off."""
    return cast(
        BetaBrowserToolsetConfigsParam,
        {
            name: {"enabled": False}
            for name in get_args(BetaBrowserMemberName)
            if name not in allowed
        },
    )


@dataclass
class _Pending:
    """What `confirm` established about a call, handed to `execute` for that same call."""

    tool_use_id: str | None
    element: PwLocator | None
    facts: ElementFacts | None
    decision: Decision
    approved: bool


@dataclass
class _Done:
    """What a member did, for the recorder."""

    value: str | None = None
    candidates: tuple[str, ...] = ()
    key: str | None = None
    url: str | None = None
    selected: tuple[str, str] | None = None
    observation: Observation | None = None


class DiscoveryBrowser(BetaAbstractBrowserToolset20260801):
    def __init__(
        self,
        surface: WebSurface,
        recorder: Recorder,
        *,
        approve: Approver = refuse_all,
        lease: Callable[[], Lease | None] = lambda: None,
        settle_max_ms: int = 2_000,
        commit_wait_ms: int = 5_000,
    ) -> None:
        """`lease` gives the agent's current lease: with a control plane, every
        actuation is fenced by it, and a hand-back issues a new one."""
        self.surface = surface
        self._lease = lease
        self.gate = surface.gate
        self.recorder = recorder
        self._approve = approve
        self._settle_max_ms = settle_max_ms
        self._commit_wait_ms = commit_wait_ms
        self._pending: _Pending | None = None
        self._current: _Pending | None = None
        self._done = _Done()
        self._dialogs_reported = 0
        self._violations_reported = 0
        self.approval_seconds = 0.0
        """Time spent waiting for a person, which the run's time budget does not count."""
        self.state: PageState = recorder.trajectory.start or recorder.capture()
        self.served = frozenset(self.gate.allowed_members) & IMPLEMENTED
        super().__init__(
            configs=member_configs(self.served), confirm=self._confirm, url_policy=self._url_policy
        )

    @property
    def last_state_key(self) -> str:
        return self.state.key

    # --- the SDK hooks ------------------------------------------------------------------

    def _url_policy(self, context: BetaURLContext, url: str) -> None:
        decision = self.gate.check_url(urljoin(self.surface.page.url, url))
        if not decision.allowed:
            self.recorder.refused("navigate", f"{decision.rule}: {decision.reason}")
            raise ToolError(f"Refused by policy ({decision.rule}): {decision.reason}.")

    def _confirm(self, context: BetaConfirmContext) -> bool:
        self._pending = None
        member = context.member
        tool_use_id = context.tool_use.id if context.tool_use else None
        try:
            _check_tab(context.input)
            element = self._element_for(member, context.input)
            facts = self.surface.facts(element) if element is not None else None
        except SurfaceError as error:
            raise ToolError(str(error)) from None
        key = context.input.text if isinstance(context.input, BetaBrowserKeyInput) else None
        decision = self.gate.check_action(member, facts, key=key, mode="discovery")
        if (
            decision.verdict != "deny"
            and facts is not None
            and facts.href
            and member == "left_click"
        ):
            # A link to a denied page is refused before the click, not stopped after it.
            target = self.gate.check_url(facts.href)
            if not target.allowed:
                decision = target
        if decision.verdict == "deny":
            self.recorder.refused(member, f"{decision.rule}: {decision.reason}")
            raise ToolError(f"Refused by policy ({decision.rule}): {decision.reason}.")
        approved = False
        if decision.verdict == "require_approval":
            request = ApprovalRequest(
                member=member,
                target=_describe(facts),
                url=JSESSIONID.sub("", self.surface.page.url),
                reason=decision.reason,
                tool_use_id=tool_use_id,
            )
            started = time.monotonic()
            try:
                approved = self._approve(request) is True
            finally:
                self.approval_seconds += time.monotonic() - started
            self.recorder.approval(member, request.target, decision.reason, approved)
            if not approved:
                self.recorder.refused(member, "a person did not approve this commit")
                return False
        self._pending = _Pending(tool_use_id, element, facts, decision, approved)
        return True

    def execute(
        self,
        context: BetaToolsetCallContext,
        name: BetaBrowserMemberName,
        input: BetaBrowserMemberInput,
    ) -> BetaBrowserMemberResult:
        tool_use_id = context.tool_use.id if context.tool_use else None
        pending = (
            self._pending if self._pending and self._pending.tool_use_id == tool_use_id else None
        )
        self._pending = None
        self._done = _Done()
        if name not in ACTION_MEMBERS:
            return self._look(context, name, input, pending)
        return self._act(context, name, input, pending)

    def _browser_state(self, context: BetaToolsetCallContext) -> BetaBrowserState:
        try:
            url, title = JSESSIONID.sub("", self.surface.page.url), self.surface.page.title()
        except PlaywrightError:
            url, title = self.state.url, self.state.title
        changes: list[Any] = []
        history = self.surface.dialog_history
        for dialog in history[self._dialogs_reported :]:
            changes.append(BetaDialogDismissed(kind=dialog.type, message=dialog.message))
        self._dialogs_reported = len(history)
        # Answered (dismissed) on arrival; the model is told above, so they stop blocking.
        self.surface.acknowledge_dialogs()
        violations = self.surface.violations
        if any(v.navigation for v in violations[self._violations_reported :]):
            changes.append(BetaNavigationRefused())
        self._violations_reported = len(violations)
        return BetaBrowserState(
            tabs=[{"tab_id": TAB_ID, "title": title, "url": url, "active": True}],
            state_changes=changes,
        )

    # --- recording ---------------------------------------------------------------------------

    def _look(
        self,
        context: BetaToolsetCallContext,
        name: BetaBrowserMemberName,
        input: BetaBrowserMemberInput,
        pending: _Pending | None,
    ) -> BetaBrowserMemberResult:
        self._current = pending
        try:
            with _as_tool_error():
                _check_tab(input)
                result = super().execute(context, name, input)
        finally:
            self._current = None
        with _as_tool_error():
            self.state = self.recorder.capture(self._done.observation)
        self.recorder.observed(self.state)
        return result

    def _act(
        self,
        context: BetaToolsetCallContext,
        name: BetaBrowserMemberName,
        input: BetaBrowserMemberInput,
        pending: _Pending | None,
    ) -> BetaBrowserMemberResult:
        """Runs one action and records it, whatever happens after it ran: an action
        that reached the page is never missing from the trajectory or the log."""
        started = time.monotonic()
        if name == "navigate":
            # A page taken off policy refuses every read until the next navigation, so
            # the navigate that recovers it starts from the last page it knew.
            before = self._capture() or self.state
        else:
            with _as_tool_error():
                before = self.recorder.capture()
        with _as_tool_error():
            element = (
                self.recorder.element(pending.element)
                if pending is not None and pending.element is not None
                else None
            )
        approved = pending is not None and pending.approved
        error: str | None = None
        dispatched = False
        unexpected: Exception | None = None
        result: BetaBrowserMemberResult = None
        self._current = pending
        try:
            with self.surface.commit_window() if approved else contextlib.nullcontext():
                commits_before = self.surface.commit_requests
                result = super().execute(context, name, input)
                if approved:
                    self._await_commit(commits_before)
        except (SurfaceError, ToolError, PlaywrightError) as failure:
            error = _message(failure)
            dispatched = isinstance(failure, SurfaceError) and failure.dispatched
        except Exception as failure:  # a driver bug: still recorded, then raised as itself
            error, unexpected = f"{type(failure).__name__}: {failure}", failure
        finally:
            self._current = None
        if error is None:
            with contextlib.suppress(SurfaceError, PlaywrightError):
                self.surface.settle(max_ms=self._settle_max_ms)
        after = self._capture()
        if after is not None:
            self.state = after
        if element is not None and self._done.selected is not None:
            element = element.model_copy(update={"selected": self._done.selected})
        self.recorder.call(
            RecordedCall(
                seq=self.recorder.next_seq(),
                turn=self.recorder.turn,
                member=name,
                ok=error is None,
                error=error,
                dispatched=dispatched,
                effect=pending.decision.effect if pending else "read",
                approved=approved,
                element=element,
                value=self._value(element),
                key=self._done.key,
                url=self._done.url,
                before=before,
                after=after,
                rationale=self.recorder.rationale,
                duration_ms=elapsed_ms(started),
            )
        )
        if unexpected is not None:
            raise unexpected
        if error is not None:
            raise ToolError(error)
        return result

    def _capture(self) -> PageState | None:
        try:
            return self.recorder.capture()
        except (SurfaceError, PlaywrightError):
            return None

    def _value(self, element: ElementRecord | None) -> StepValue | None:
        if self._done.value is None:
            return None
        candidates = (self._done.value, *self._done.candidates)
        return self.recorder.bind(candidates, element.sensitivity if element else "pii")

    def _await_commit(self, before: int) -> None:
        """Keep the window open until the commit request has left (it may go after the
        click returns), or the wait runs out: a commit-classified control need not send
        a commit-route request at all."""
        deadline = time.monotonic() + self._commit_wait_ms / 1000
        while self.surface.commit_requests <= before and time.monotonic() < deadline:
            self.surface.pause(100)

    # --- elements ------------------------------------------------------------------------------

    def _element_for(self, member: str, input: BetaBrowserMemberInput) -> PwLocator | None:
        if isinstance(input, BetaBrowserLeftClickInput | BetaBrowserHoverInput):
            return self._target(input.target)
        if isinstance(input, BetaBrowserFormInputInput | BetaBrowserScrollToInput):
            return self.surface.element_for_ref(input.target.ref)
        if isinstance(input, BetaBrowserTypeInput):
            focused = self.surface.focused_element()
            if focused is None:
                raise _NoFocus("no field has focus: click the field first, or use form_input")
            return focused
        if isinstance(input, BetaBrowserKeyInput):
            return self.surface.focused_element() or self.surface.page.locator("body")
        return None

    def _target(self, target: BetaBrowserRefTarget | BetaBrowserCoordinateTarget) -> PwLocator:
        if isinstance(target, BetaBrowserRefTarget):
            return self.surface.element_for_ref(target.ref)
        return self.surface.element_at(target.x, target.y)

    def _resolved(self) -> tuple[PwLocator, ElementFacts | None]:
        current = self._current
        if current is None or current.element is None:
            raise ToolError("the target could not be resolved; call read_page and try again")
        return current.element, current.facts

    # --- members ---------------------------------------------------------------------------------

    def navigate(
        self, context: BetaToolsetCallContext, input: BetaBrowserNavigateInput
    ) -> BetaBrowserNavigateResult:
        page = self.surface.page
        word = input.url.strip().lower()
        if word in ("back", "forward"):
            raise ToolError("going back or forward is not available; navigate to an address")
        url = page.url if word == "reload" else urljoin(page.url, input.url)
        status = self.surface.navigate(url, lease=self._lease())
        self._done.url = JSESSIONID.sub("", url)
        return BetaBrowserNavigateResult(
            url=JSESSIONID.sub("", page.url), status=status, title=page.title()
        )

    def read_page(self, context: BetaToolsetCallContext, input: BetaBrowserReadPageInput) -> str:
        observation = self.surface.observe()
        self._done.observation = observation
        text = observation.snapshot
        if input.ref:
            text = _subtree(text, input.ref)
            if not text:
                raise ToolError(f"{input.ref} is not on the page now; call read_page without ref")
        if input.filter == "interactive":
            text = _interactive(text)
        if input.depth is not None:
            text = _limit_depth(text, input.depth)
        return text or "(the page has no accessible content)"

    def get_page_text(
        self, context: BetaToolsetCallContext, input: BetaBrowserGetPageTextInput
    ) -> str:
        return JSESSIONID.sub("", self.surface.page_text())

    def screenshot(
        self, context: BetaToolsetCallContext, input: BetaBrowserScreenshotInput
    ) -> BetaScreenshotResult:
        return _image(self.surface.screenshot_png())

    def zoom(
        self, context: BetaToolsetCallContext, input: BetaBrowserZoomInput
    ) -> BetaScreenshotResult:
        if len(input.region) != 4:
            raise ToolError("region takes four numbers: x0, y0, x1, y1")
        x0, y0, x1, y1 = input.region
        if x1 <= x0 or y1 <= y0:
            raise ToolError("region must have x1 > x0 and y1 > y0")
        clip = {"x": float(x0), "y": float(y0), "width": float(x1 - x0), "height": float(y1 - y0)}
        return _image(self.surface.screenshot_png(clip=clip))

    def left_click(
        self, context: BetaToolsetCallContext, input: BetaBrowserLeftClickInput
    ) -> str | None:
        if input.modifiers:
            raise ToolError("clicks with modifier keys are not available")
        element, _ = self._resolved()
        self.surface.act("click", element, lease=self._lease())
        return None

    def hover(self, context: BetaToolsetCallContext, input: BetaBrowserHoverInput) -> str | None:
        element = self._current.element if self._current else None
        self.surface.hover(
            element if element is not None else self._target(input.target), lease=self._lease()
        )
        return None

    def form_input(
        self, context: BetaToolsetCallContext, input: BetaBrowserFormInputInput
    ) -> str | None:
        element, facts = self._resolved()
        value = _text(input.value)
        if facts is not None and facts.tag == "select":
            self.surface.act("select", element, value, lease=self._lease())
            self._done.selected = self.surface.selected(element)
            self._done.candidates = self._done.selected or ()
        elif facts is not None and facts.input_type in ("checkbox", "radio"):
            value = "true" if value.lower() in ("true", "1", "yes", "on", "checked") else "false"
            self.surface.act("check", element, value, lease=self._lease())
        else:
            self.surface.act("fill", element, value, lease=self._lease())
        self._done.value = value
        return None

    def type(self, context: BetaToolsetCallContext, input: BetaBrowserTypeInput) -> str | None:
        element, facts = self._resolved()
        if facts is None or not facts.is_form_control:
            raise ToolError("the focused element is not a text field; click a field first")
        value = self.surface.current_value(element) + input.text
        self.surface.act("fill", element, value, lease=self._lease())
        self._done.value = value
        return None

    def key(self, context: BetaToolsetCallContext, input: BetaBrowserKeyInput) -> str | None:
        element, _ = self._resolved()
        key = normalise_key(input.text)
        repeat = max(1, min(input.repeat or 1, MAX_REPEAT))
        if repeat > 1 and self._current is not None and self._current.approved:
            raise ToolError("a key press that commits is pressed once; repeat must be 1")
        for _ in range(repeat):
            self.surface.act("press", element, key, lease=self._lease())
        self._done.key = key
        return None

    def scroll(self, context: BetaToolsetCallContext, input: BetaBrowserScrollInput) -> str | None:
        step = SCROLL_STEP_PX * max(1, min(input.scroll_amount or 3, 20))
        dx, dy = {"up": (0, -step), "down": (0, step), "left": (-step, 0), "right": (step, 0)}[
            input.scroll_direction
        ]
        self.surface.wheel(input.target.x, input.target.y, dx, dy, lease=self._lease())
        return None

    def scroll_to(
        self, context: BetaToolsetCallContext, input: BetaBrowserScrollToInput
    ) -> str | None:
        self.surface.scroll_into_view(
            self.surface.element_for_ref(input.target.ref), lease=self._lease()
        )
        return None

    def wait(self, context: BetaToolsetCallContext, input: BetaBrowserWaitInput) -> str | None:
        self.surface.pause(int(max(0.0, min(input.duration, MAX_WAIT_S)) * 1000))
        return None


class _NoFocus(SurfaceError):
    code = "no_focus"


@contextlib.contextmanager
def _as_tool_error() -> Iterator[None]:
    """Surface errors are the model's to read; anything else is a driver bug and
    propagates as itself (the SDK turns it into an error result and logs it)."""
    try:
        yield
    except SurfaceError as error:
        raise ToolError(_message(error)) from None


def _message(error: Exception) -> str:
    text = str(error).strip().split("\n", 1)[0][:300] or type(error).__name__
    if isinstance(error, PolicyBlocked) and "taken off policy" in text:
        text += "; navigate to an allowed address to continue"
    return text


def _check_tab(input: BetaBrowserMemberInput) -> None:
    tab = getattr(input, "tab_id", None)
    if tab is not None and tab != TAB_ID:
        raise ToolError(f"there is one tab, {TAB_ID}; {tab!r} does not exist")


def _describe(facts: ElementFacts | None) -> str:
    if facts is None:
        return "the page"
    what = facts.role or facts.tag
    named = facts.name or facts.value_attr or facts.label
    return f"{what} {named!r}" if named else what


def _text(value: str | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _image(png: bytes) -> BetaScreenshotResult:
    return BetaScreenshotResult(data=base64.b64encode(png).decode("ascii"), media_type="image/png")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _subtree(snapshot: str, ref: str) -> str:
    lines = snapshot.splitlines()
    marker = f"[ref={ref}]"
    for index, line in enumerate(lines):
        if marker in line:
            depth = _indent(line)
            block = [line]
            for rest in lines[index + 1 :]:
                if rest.strip() and _indent(rest) <= depth:
                    break
                block.append(rest)
            return "\n".join(block)
    return ""


def _interactive(snapshot: str) -> str:
    kept = []
    for line in snapshot.splitlines():
        role, _ = parse_line(line)
        if role in INTERACTIVE_ROLES:
            kept.append(line)
    return "\n".join(kept)


def _limit_depth(snapshot: str, depth: int) -> str:
    return "\n".join(line for line in snapshot.splitlines() if _indent(line) // 2 <= depth)
