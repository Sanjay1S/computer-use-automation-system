"""Handoffs: the automation thread's side of the control plane.

Opening an intervention pauses automation. While a person decides, this loop keeps
the browser's events flowing (route handlers, dialog handlers and the human
recorder run only while this thread is inside a browser call), refreshes a masked
live screenshot for an operator who cannot see the window, and logs every input the
person gives. Closing the intervention writes the handoff to handoff.json (who, when,
what they did, how it ended) and gives automation its new lease.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page

from understudy.artifact.schema import Effect, Sensitivity
from understudy.control.human_recorder import HumanInput, HumanRecorder
from understudy.control.interventions import (
    Decision,
    Handoff,
    HandoffSummary,
    HumanAction,
    Intervention,
    InterventionKind,
    Subject,
)
from understudy.control.plane import Change, Closing, ControlPlane, Holder, Lease, Transition
from understudy.evidence.run import EventTarget, RunContext
from understudy.redaction.redactor import Redactor
from understudy.surface.base import SurfaceError
from understudy.surface.web import WebSurface

InputRelay = Callable[[Page, Intervention], None]
"""Delivers a remote operator's input to the page. The browser is driven from the
automation thread only, so input that comes from elsewhere (a screencast relay,
which is design only; a scripted person in tests) is delivered here, on every poll
while automation waits. A relay delivers input only from whoever may give it (in
production, the operator holding the session); anyone else's is reported by the
human recorder as unexpected, as input at a headed window would be."""

HumanHook = Callable[[HumanInput, Intervention], HumanAction]


@dataclass
class Control:
    """How a run involves people. Built by the caller, used by one run."""

    plane: ControlPlane
    escalate: bool = False
    """Hand a stuck or unknown state to a person instead of stopping (--on-stuck escalate)."""
    interactive_approval: bool = False
    """Replay: a person approves the commit in the console when the run reaches it."""
    relay: InputRelay | None = None
    poll_ms: int = 200
    max_wait_s: float = 1_800.0
    max_handoffs: int = 3
    """Takeovers per run. One that keeps coming back is reported, not looped."""
    live_every_s: float = 2.0


@dataclass
class _Open:
    intervention: Intervention
    started: float
    actions: list[HumanAction] = field(default_factory=list)
    unexpected: int = 0


class Handoffs:
    def __init__(
        self,
        control: Control,
        run: RunContext,
        surface: WebSurface,
        *,
        automation: Holder,
        subject: Subject,
        artifact_hash: str | None = None,
        masks: Sequence[str] = (),
        params: Mapping[str, str] | None = None,
        on_human: HumanHook | None = None,
    ) -> None:
        """The surface must have been built with `control=control.plane`, and this
        must be built before the first navigation, so the human recorder runs in
        every document of the session."""
        self.control = control
        self.plane = control.plane
        self.run = run
        self.surface = surface
        self.automation = automation
        self.subject = subject
        self.artifact_hash = artifact_hash
        self.masks = list(masks)
        self.params = dict(params or {})
        self.on_human = on_human or self.log_human
        self.records: list[Handoff] = []
        self.takeovers = 0
        self.foreign_inputs = 0
        """Inputs the page reported from anyone but automation, a person's included:
        the page may no longer hold what automation entered."""
        self.waited_s = 0.0
        """Time spent waiting for people, which run budgets do not count."""
        self._open: _Open | None = None
        self._last: _Open | None = None
        self._seq = 0
        self.lease: Lease = self.plane.start(
            run.run_id, automation=automation, on_change=self._on_change
        )
        self.human = HumanRecorder(surface, self.plane)

    @property
    def can_hand_off(self) -> bool:
        return self.control.escalate and self.takeovers < self.control.max_handoffs

    # --- the lifecycle of one intervention ---------------------------------------------

    @contextlib.contextmanager
    def opened(
        self,
        kind: InterventionKind,
        reason: str,
        detail: str,
        *,
        step_id: str | None = None,
        intent: str | None = None,
        resume_points: Sequence[str] = (),
    ) -> Iterator[Intervention]:
        """An intervention that is closed however the block ends: an error while a
        person holds the session (the browser died, the page closed) closes it as
        interrupted, so the console stops offering decisions nobody will act on."""
        intervention = self.open(
            kind, reason, detail, step_id=step_id, intent=intent, resume_points=resume_points
        )
        try:
            yield intervention
        except BaseException as error:
            if self.plane.get(intervention.id).status != "closed":
                with contextlib.suppress(Exception):
                    self.close(intervention, f"interrupted ({type(error).__name__})", to="ABORTED")
            raise
        if self.plane.get(intervention.id).status != "closed":
            self.close(intervention, "interrupted (left open)", to="ABORTED")
            raise RuntimeError(f"intervention {intervention.id} was left open")

    def open(
        self,
        kind: InterventionKind,
        reason: str,
        detail: str,
        *,
        step_id: str | None = None,
        intent: str | None = None,
        resume_points: Sequence[str] = (),
    ) -> Intervention:
        self.poll()
        redactor = self.run.redactor
        try:
            url: str | None = redactor.url(self.surface.page.url)
        except PlaywrightError:
            url = None
        intervention = self.plane.open(
            self.lease,
            kind=kind,
            subject=self.subject,
            artifact_hash=self.artifact_hash,
            reason=reason,
            detail=redactor.text(detail),
            step_id=step_id,
            intent=redactor.text(intent) if intent else None,
            url=url,
            screenshot=self._screenshot(f"intervention_{reason}"),
            recent_events=[
                e.model_dump(mode="json", exclude_none=True) for e in self.run.recent(10)
            ],
            resume_points=resume_points,
        )
        if kind != "approval":
            self.takeovers += 1
        self._open = _Open(intervention, time.monotonic())
        return intervention

    def wait(self, intervention: Intervention) -> Decision | None:
        """Until a person decides, or the wait runs out (None)."""
        started = time.monotonic()
        deadline = started + self.control.max_wait_s
        next_live = started
        dialogs = len(self.surface.dialog_history)
        try:
            while True:
                decision = self.plane.decision(intervention.id)
                if decision is not None:
                    return decision
                now = time.monotonic()
                if now >= deadline:
                    return None
                if now >= next_live:
                    self._live(intervention)
                    next_live = now + self.control.live_every_s
                self.surface.pause(self.control.poll_ms)
                self._take(self.human.drain())
                dialogs = self._dialogs_since(dialogs)
                if self.control.relay is not None:
                    self.control.relay(self.surface.page, self.plane.get(intervention.id))
        finally:
            self.waited_s += time.monotonic() - started

    def close(self, intervention: Intervention, resolution: str, *, to: Closing) -> Lease:
        """Every input the page reported up to now still belongs to the person, then
        the session changes hands."""
        self._take(self.human.flush())
        self.lease = self.plane.close(intervention.id, resolution, to=to)
        closed = self.plane.get(intervention.id)
        current = self._open
        self._last, self._open = current, None
        if closed.kind != "approval" and current is not None:
            decision = closed.decision
            self.records.append(
                Handoff(
                    intervention_id=closed.id,
                    kind=closed.kind,
                    reason=closed.reason,
                    step_id=closed.step_id,
                    operator=closed.claimed_by or (decision.operator if decision else None),
                    opened_at=closed.opened_at,
                    claimed_at=closed.claimed_at,
                    closed_at=closed.closed_at or datetime.now(UTC),
                    duration_ms=int((time.monotonic() - current.started) * 1000),
                    actions=current.actions,
                    unexpected_inputs=current.unexpected,
                    resolution=resolution,
                    note=decision.note if decision else None,
                    resume_from=decision.resume_from if decision else None,
                )
            )
            current.actions = self.records[-1].actions  # a late report lands in the record
            self._write_handoffs()
        return self.lease

    def poll(self) -> None:
        """Outside an intervention: log input nobody should have given."""
        self._take(self.human.drain())

    def summaries(self) -> list[HandoffSummary]:
        actions = {h.intervention_id: len(h.actions) for h in self.records}
        return [
            HandoffSummary(
                intervention_id=i.id,
                kind=i.kind,
                reason=i.reason,
                step_id=i.step_id,
                operator=i.claimed_by or (i.decision.operator if i.decision else None),
                resolution=i.resolution or "open",
                human_actions=actions.get(i.id, 0),
            )
            for i in self.plane.interventions()
        ]

    # --- what a person did ---------------------------------------------------------------

    def log_human(self, item: HumanInput, intervention: Intervention) -> HumanAction:
        """The default record of a person's input: an `act` event with actor human,
        and its line in the handoff."""
        target = None
        if item.locator is not None:
            with contextlib.suppress(SurfaceError, PlaywrightError):
                target = self.surface.synthesize(item.locator, params=self.params).target
        description = target.description if target is not None else item.describe()
        effect = self.effect(item)
        value, sensitivity = self.value(item)
        action = self.human_action(
            item, description, effect, verified=target is not None, value=(value, sensitivity)
        )
        self.run.event(
            "act",
            actor="human",
            member=item.member,
            action=item.kind,
            target=target
            or EventTarget(
                description=description,
                fingerprint=item.facts.fingerprint().model_dump(exclude_none=True),
            ),
            value=value,
            value_sensitivity=sensitivity,
            url=item.url,
            ok=True,
            intervention=intervention.id,
            effect=effect,
            verified=target is not None,
            in_frame=item.in_frame,
        )
        return action

    def human_action(
        self,
        item: HumanInput,
        description: str,
        effect: Effect,
        *,
        verified: bool,
        value: tuple[str | None, Sensitivity] | None = None,
    ) -> HumanAction:
        """`value`: the input's value as `value()` gives it, when already worked out."""
        redactor = self.run.redactor
        raw, sensitivity = value or self.value(item)
        self._seq += 1
        return HumanAction(
            seq=self._seq,
            at=datetime.fromtimestamp(item.at, UTC),
            kind=item.kind,
            description=redactor.text(description),
            effect=effect,
            value=redactor.value(raw, sensitivity) if raw is not None else None,
            url=redactor.url(item.url) if item.url else None,
            verified=verified,
            in_frame=item.in_frame,
        )

    def effect(self, item: HumanInput) -> Effect:
        if item.member is None:
            return "input" if item.facts.is_form_control else "read"
        return self.surface.gate.classify(item.member, item.facts)

    def value(self, item: HumanInput) -> tuple[str | None, Sensitivity]:
        """The value as it should be logged: a public field's value still goes through
        the run's known values and detectors, so an input typed by hand is masked
        just as when replay types it."""
        if item.checked is not None:
            return ("true" if item.checked else "false"), "public"
        if item.value is None:
            return None, "public"
        facts = item.facts
        sensitivity = Redactor.classify_field(
            input_type=facts.input_type,
            label=facts.label,
            name_attr=facts.name_attr,
            autocomplete=facts.autocomplete,
        )
        if sensitivity == "public":
            return self.run.redactor.text(item.value), "public"
        return item.value, sensitivity

    # --- internals ---------------------------------------------------------------------

    def _take(self, inputs: list[HumanInput]) -> None:
        for item in inputs:
            if item.attribution == "automation":
                continue
            self.foreign_inputs += 1
            current = self._open or self._last
            if item.attribution == "unexpected":
                if self._open is not None:
                    self._open.unexpected += 1
                self.run.event(
                    "control",
                    actor="human",
                    action=item.kind,
                    target=EventTarget(
                        description=item.describe(),
                        fingerprint=item.facts.fingerprint().model_dump(exclude_none=True),
                    ),
                    url=item.url,
                    ok=False,
                    warning="unexpected_input",
                    state=self.plane.state_at(item.at),
                )
                continue
            if current is None:
                continue
            action = self.on_human(item, current.intervention)
            current.actions.append(action)
            if current is self._last:
                self._write_handoffs()  # reported after its handoff was closed

    def _on_change(
        self, change: Change, intervention: Intervention, transition: Transition | None
    ) -> None:
        """Runs on whichever thread changed the plane, under the plane's lock."""
        self.run.write_json(
            f"interventions/{intervention.id}.json",
            intervention,
            verbatim=Intervention.IDENTIFIERS,
        )
        if change == "updated":
            return
        decision = intervention.decision
        fields: dict[str, Any] = {
            "change": change,
            "intervention": intervention.id,
            "intervention_kind": intervention.kind,
            "reason": intervention.reason,
            "state": transition.state if transition else None,
            "epoch": transition.lease.epoch if transition else None,
            "operator": intervention.claimed_by or (decision.operator if decision else None),
            "decision": decision.action if decision and change == "decided" else None,
            "note": decision.note if decision and change == "decided" else None,
            "resume_from": decision.resume_from if decision and change == "decided" else None,
            "resolution": intervention.resolution if change == "closed" else None,
        }
        self.run.event(
            "control",
            actor=self.automation if change in ("opened", "closed") else "human",
            step_id=intervention.step_id,
            ok=True,
            **{k: v for k, v in fields.items() if v is not None},
        )

    def _dialogs_since(self, seen: int) -> int:
        """Native dialogs still get the surface's answer while a person holds the
        session (an open one would freeze every read this loop makes), and it is the
        non-committal one: logged, so the person can see what they were spared."""
        history = self.surface.dialog_history
        for dialog in history[seen:]:
            self.run.event(
                "control",
                actor="system",
                ok=False,
                warning="dialog_answered_during_handoff",
                dialog=self.run.redactor.text(str(dialog)),
                answer=dialog.answer,
            )
        return len(history)

    def _write_handoffs(self) -> None:
        self.run.write_json(
            "handoff.json", {"handoffs": [h.model_dump(mode="json") for h in self.records]}
        )

    def _screenshot(self, label: str) -> str | None:
        try:
            path = self.surface.screenshot(self.run.screenshot_path(label), self.masks)
        except (PlaywrightError, SurfaceError):
            return None  # evidence never stops a handoff
        return str(path.relative_to(self.run.dir))

    def _live(self, intervention: Intervention) -> None:
        path = self.run.subdir("interventions") / f"{intervention.id}.live.png"
        partial = path.with_suffix(".part.png")
        try:
            self.surface.screenshot(partial, self.masks)
        except (PlaywrightError, SurfaceError):
            return
        os.replace(partial, path)  # the console never reads a half-written image
        self.plane.update_live(intervention.id, str(path.relative_to(self.run.dir)))
