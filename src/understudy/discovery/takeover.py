"""A person in a discovery run: commit approvals in the console, takeovers of a stuck
run, and what the person did, recorded as trajectory calls.

A person's inputs are compiled like the model's (they may be the step the model
could not find) and marked as theirs, so the review summary flags each one: it may
be part of the flow, or only the clearing of something in the way.
"""

from __future__ import annotations

import contextlib

from playwright.sync_api import Error as PlaywrightError

from understudy.control.handoff import Handoffs
from understudy.control.human_recorder import HumanInput
from understudy.control.interventions import HumanAction, Intervention
from understudy.discovery.agent import Handback
from understudy.discovery.driver import ApprovalRequest, DiscoveryBrowser
from understudy.discovery.trajectory import (
    ElementRecord,
    PageState,
    RecordedCall,
    Recorder,
    StepValue,
)
from understudy.redaction.redactor import Redactor
from understudy.surface.base import SurfaceError


class PersonInRun:
    def __init__(self, handoffs: Handoffs, recorder: Recorder) -> None:
        self.handoffs = handoffs
        self.recorder = recorder
        self.browser: DiscoveryBrowser | None = None
        """Set once built: a hand-back refreshes the page state it judges progress by."""
        self._calls: list[RecordedCall] = []
        self._told: list[str] = []
        self._before: PageState | None = None
        handoffs.on_human = self.record

    def on_turn(self, turn: int, rationale: str | None) -> None:
        """Each model turn: input nobody should have given is logged as it happens."""
        self.recorder.begin_turn(turn, rationale)
        self.handoffs.poll()

    # --- approvals --------------------------------------------------------------------

    def approve(self, request: ApprovalRequest) -> bool:
        """The driver's approver: a commit the model attempts waits for a person."""
        handoffs = self.handoffs
        detail = (
            f"The model wants to run {request.member} on {request.target} ({request.reason}). "
            f"Page: {request.url}."
        )
        with handoffs.opened("approval", "commit_approval", detail) as intervention:
            decision = handoffs.wait(intervention)
            approved = decision is not None and decision.action == "approve"
            resolution = "approved" if approved else ("denied" if decision else "timed_out")
            handoffs.close(intervention, resolution, to="AGENT")
        return approved

    # --- takeovers ----------------------------------------------------------------------

    def on_stuck(self, code: str, reason: str) -> Handback:
        handoffs = self.handoffs
        if not handoffs.can_hand_off:
            return Handback(False, resolution="no further handoffs are allowed in this run")
        handoffs.poll()  # anything still queued belongs to the time before this one
        self._before = self._capture()
        self._calls, self._told = [], []
        with handoffs.opened("takeover", code, reason) as intervention:
            decision = handoffs.wait(intervention)
            if decision is None:
                handoffs.close(intervention, "timed_out", to="AGENT")
                return Handback(False, resolution="nobody took over in time")
            if decision.action == "abort":
                handoffs.close(intervention, "aborted", to="ABORTED")
                return Handback(False, resolution=f"aborted by {decision.operator}")
            done = decision.action == "complete"
            handoffs.close(intervention, "completed_by_person" if done else "resumed", to="AGENT")
        after = self._capture()
        if self._calls and after is not None:
            self._calls[-1].after = after
        if self.browser is not None and after is not None:
            self.browser.state = after
        return Handback(True, text=self._account(decision.note, done))

    def record(self, item: HumanInput, intervention: Intervention) -> HumanAction:
        """A person's input, as a trajectory call (logged as actor human) when it is an
        action of its own; a click into a field is part of the change that follows."""
        handoffs = self.handoffs
        member = item.member
        effect = handoffs.effect(item)
        late = handoffs.plane.get(intervention.id).status == "closed"
        if member is None or late:
            # A report read after its handoff closed would land among the model's later
            # calls, out of order: it is logged, never compiled.
            return handoffs.log_human(item, intervention)
        element = self._element(item)
        value: StepValue | None = None
        if item.kind == "change":
            if item.checked is not None:
                candidates = ["true" if item.checked else "false"]
            else:
                candidates = [v for v in (item.value, item.option_label) if v]
            value = self.recorder.bind(candidates, element.sensitivity)
            if item.facts.tag == "select":
                element = element.model_copy(
                    update={"selected": (item.value or "", item.option_label or "")}
                )
        # A commit outside an approved window is stopped at the network: it did not
        # happen, so it is recorded as not run and never compiled.
        blocked = effect == "commit"
        call = RecordedCall(
            seq=self.recorder.next_seq(),
            turn=self.recorder.turn,
            member=member,
            actor="human",
            ok=not blocked,
            error="a person's commit is not sent; only an approved commit is" if blocked else None,
            effect=effect,
            element=element,
            value=value,
            before=self._before if not self._calls else None,
        )
        self.recorder.call(call)
        if not blocked:
            self._calls.append(call)
        told = _told(item, element, value)
        self._told.append(f"{told} (a commit, which was not sent)" if blocked else told)
        return handoffs.human_action(
            item, element.description, effect, verified=element.target is not None
        )

    # --- internals ------------------------------------------------------------------------

    def _element(self, item: HumanInput) -> ElementRecord:
        if item.locator is not None:
            with contextlib.suppress(SurfaceError, PlaywrightError):
                return self.recorder.element(item.locator)
        facts = item.facts
        return ElementRecord(
            description=item.describe(),
            tag=facts.tag,
            role=facts.role,
            name=facts.name,
            input_type=facts.input_type,
            sensitivity=Redactor.classify_field(
                input_type=facts.input_type,
                label=facts.label,
                name_attr=facts.name_attr,
                autocomplete=facts.autocomplete,
            ),
            rejected=[("target", "the element was gone before its locators could be verified")],
        )

    def _capture(self) -> PageState | None:
        try:
            return self.recorder.capture()
        except (SurfaceError, PlaywrightError):
            return None

    def _account(self, note: str | None, done: bool) -> str:
        parts = ["A person took over the browser and has handed it back."]
        if self._told:
            parts.append("What they did: " + "; ".join(self._told) + ".")
        else:
            parts.append("They gave the page no input.")
        if note:
            parts.append(f"Their note: {note}")
        if done:
            parts.append(
                "They say the task is complete: check the page, then record the outputs and "
                "declare success if it shows the result."
            )
        parts.append("Read the page again before you continue; earlier refs are stale.")
        return " ".join(parts)


def _told(item: HumanInput, element: ElementRecord, value: StepValue | None) -> str:
    """One input as the model is told it. Values are named by their param when they
    are one, and never shown when they were withheld."""
    what = element.description
    if item.kind != "change":
        return f"clicked {what}"
    if item.checked is not None:
        return f"{'checked' if item.checked else 'unchecked'} {what}"
    if value is not None and value.param:
        return f"set {what} to the value of {value.param}"
    if value is not None and value.literal is not None:
        return f"set {what} to {value.literal!r}"
    return f"entered a value in {what}"
