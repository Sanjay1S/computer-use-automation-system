"""The control plane: who may drive the live session, and the interventions that move
control between automation and a person.

Control is a lease with a fencing token. The plane issues a lease (holder and
epoch) and every actuation is checked against it (`WebSurface.act(..., lease=)`).
Each change of hands raises the epoch, so a lease taken before a handover can
never act after it.

    AGENT ──open──► AWAITING_HUMAN ──claim──► HUMAN
      ▲                 │  approve/deny,          │  resume ──► AGENT
      └──── close ──────┘  abort, timeout         ├  complete ─► COMPLETED_BY_HUMAN
                                                  └  abort ────► ABORTED

Operators only record decisions. Automation acts on a decision and then closes the
intervention, which is the actual change of state, so the human's inputs up to that
moment are still theirs. Two threads touch the plane: automation (open, wait,
close, check) and the console (claim, decide). All state is behind one lock, and
nothing here touches the browser.
"""

from __future__ import annotations

import secrets
import string
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from understudy.control.interventions import (
    ACTIONS,
    Decision,
    Intervention,
    InterventionKind,
    InterventionQueue,
    OperatorAction,
    Subject,
)
from understudy.surface.base import SurfaceError

ControlState = Literal["AGENT", "AWAITING_HUMAN", "HUMAN", "COMPLETED_BY_HUMAN", "ABORTED"]
Holder = Literal["agent", "engine", "human"]
Closing = Literal["AGENT", "COMPLETED_BY_HUMAN", "ABORTED"]
Change = Literal["opened", "claimed", "decided", "closed", "updated"]

MAX_OPERATOR = 80
MAX_NOTE = 2_000


@dataclass(frozen=True)
class Lease:
    holder: Holder | None
    """None while automation waits for a person who has not claimed the session."""
    epoch: int
    operator: str | None = None


@dataclass(frozen=True)
class Transition:
    at: float
    """Wall clock, seconds: comparable with the page's Date.now() / 1000."""
    state: ControlState
    lease: Lease
    intervention: str | None
    why: str


class ControlLost(SurfaceError):
    """An actuation under a lease that is no longer current. Nothing was done."""

    code = "control_lost"


class Refused(Exception):
    """An operator action that is not possible now. The console answers 409."""


class ControlPlane:
    def __init__(
        self,
        *,
        queue: InterventionQueue | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.queue = queue
        self._clock = clock
        self._lock = threading.RLock()
        self.run_id: str | None = None
        self._automation: Holder = "engine"
        self._state: ControlState = "AGENT"
        self._lease = Lease("engine", 0)
        self._interventions: dict[str, Intervention] = {}
        self._transitions: list[Transition] = []
        self._on_change: Callable[[Change, Intervention, Transition | None], None] | None = None

    def start(
        self,
        run_id: str,
        *,
        automation: Holder,
        on_change: Callable[[Change, Intervention, Transition | None], None] | None = None,
    ) -> Lease:
        """Binds the plane to one run and returns automation's first lease."""
        with self._lock:
            if self.run_id is not None:
                raise RuntimeError("a control plane serves one run")
            self.run_id = run_id
            self._automation = automation
            self._on_change = on_change
            self._lease = Lease(automation, 1)
            self._record("AGENT", None, "run started")
            return self._lease

    # --- reading ------------------------------------------------------------------

    @property
    def state(self) -> ControlState:
        with self._lock:
            return self._state

    @property
    def lease(self) -> Lease:
        with self._lock:
            return self._lease

    def check(self, lease: Lease | None) -> None:
        """Fencing: the actor's lease must be the current one."""
        with self._lock:
            current = self._lease
        if lease is None or lease != current:
            held = f"held by {current.holder}" if current.holder else "held by nobody"
            raise ControlLost(
                f"this actor's lease (epoch {lease.epoch if lease else '-'}) is not the current "
                f"one (epoch {current.epoch}, {held}); nothing was done"
            )

    def state_at(self, at: float) -> ControlState:
        """The state at a wall-clock time, so input reported late is still judged by
        who held the session when it happened."""
        with self._lock:
            state: ControlState = "AGENT"
            for transition in self._transitions:
                if transition.at > at:
                    break
                state = transition.state
            return state

    @property
    def transitions(self) -> list[Transition]:
        with self._lock:
            return list(self._transitions)

    def get(self, intervention_id: str) -> Intervention:
        with self._lock:
            return self._interventions[intervention_id].model_copy(deep=True)

    def interventions(self) -> list[Intervention]:
        with self._lock:
            return [i.model_copy(deep=True) for i in self._interventions.values()]

    def decision(self, intervention_id: str) -> Decision | None:
        with self._lock:
            return self._interventions[intervention_id].decision

    def snapshot(self) -> dict[str, Any]:
        """What the console shows."""
        with self._lock:
            return {
                "run_id": self.run_id,
                "state": self._state,
                "lease": {
                    "holder": self._lease.holder,
                    "epoch": self._lease.epoch,
                    "operator": self._lease.operator,
                },
                "interventions": [
                    i.model_dump(mode="json") for i in reversed(self._interventions.values())
                ],
            }

    # --- automation's side ----------------------------------------------------------

    def open(
        self,
        lease: Lease,
        *,
        kind: InterventionKind,
        subject: Subject,
        reason: str,
        detail: str,
        artifact_hash: str | None = None,
        step_id: str | None = None,
        intent: str | None = None,
        url: str | None = None,
        screenshot: str | None = None,
        recent_events: Sequence[dict[str, Any]] = (),
        resume_points: Sequence[str] = (),
    ) -> Intervention:
        """Automation pauses and asks for a person. Only the lease holder can ask,
        and from then on its lease no longer acts."""
        with self._lock:
            self.check(lease)
            if self._state != "AGENT" or self.run_id is None:
                raise RuntimeError(f"an intervention cannot open in state {self._state}")
            intervention = Intervention(
                id=_new_id(),
                run_id=self.run_id,
                kind=kind,
                subject=subject,
                artifact_hash=artifact_hash,
                step_id=step_id,
                intent=intent,
                reason=reason,
                detail=detail,
                url=url,
                screenshot=screenshot,
                recent_events=list(recent_events),
                allowed=list(ACTIONS[kind]),
                resume_points=list(resume_points) if kind == "takeover" else [],
                opened_at=_now(),
            )
            self._interventions[intervention.id] = intervention
            self._lease = Lease(None, self._lease.epoch + 1)
            transition = self._record("AWAITING_HUMAN", intervention.id, f"{kind}: {reason}")
            self._changed("opened", intervention, transition)
            opened = intervention.model_copy(deep=True)
        if self.queue is not None:
            self.queue.publish(opened)
        return opened

    def close(self, intervention_id: str, resolution: str, *, to: Closing) -> Lease:
        """Automation has acted on the decision (or stopped waiting). Returns its
        new lease; after a terminal state no lease acts."""
        with self._lock:
            intervention = self._interventions[intervention_id]
            if intervention.status == "closed":
                raise RuntimeError(f"intervention {intervention_id} is already closed")
            intervention.status = "closed"
            intervention.resolution = resolution
            intervention.closed_at = _now()
            holder = self._automation if to == "AGENT" else None
            self._lease = Lease(holder, self._lease.epoch + 1)
            transition = self._record(to, intervention_id, resolution)
            self._changed("closed", intervention, transition)
            return self._lease

    def update_live(self, intervention_id: str, path: str) -> None:
        with self._lock:
            intervention = self._interventions[intervention_id]
            if intervention.live_screenshot != path:
                intervention.live_screenshot = path
                self._changed("updated", intervention, None)

    # --- the operator's side ----------------------------------------------------------

    def act(
        self,
        intervention_id: str,
        action: OperatorAction,
        *,
        operator: str,
        note: str | None = None,
        resume_from: str | None = None,
    ) -> Intervention:
        operator = " ".join(operator.split())
        if not operator or len(operator) > MAX_OPERATOR:
            raise ValueError(f"an operator name of 1-{MAX_OPERATOR} characters is required")
        if note is not None and len(note) > MAX_NOTE:
            raise ValueError(f"a note is at most {MAX_NOTE} characters")
        with self._lock:
            if intervention_id not in self._interventions:
                raise KeyError(intervention_id)
            intervention = self._interventions[intervention_id]
            if intervention.status in ("decided", "closed"):
                raise Refused(f"intervention {intervention_id} is already {intervention.status}")
            if action not in intervention.allowed:
                raise Refused(f"{action} is not possible for a {intervention.kind} intervention")
            claimed = intervention.status == "claimed"
            if claimed and operator != intervention.claimed_by:
                raise Refused(f"{intervention.claimed_by} holds this session; only they can decide")
            if action == "claim":
                if claimed:
                    raise Refused("already claimed")
                intervention.status = "claimed"
                intervention.claimed_by = operator
                intervention.claimed_at = _now()
                self._lease = Lease("human", self._lease.epoch + 1, operator)
                transition = self._record("HUMAN", intervention_id, f"claimed by {operator}")
                self._changed("claimed", intervention, transition)
                return intervention.model_copy(deep=True)
            if action in ("resume", "complete") and not claimed:
                raise Refused(f"claim the session before you {action}")
            if action == "resume" and resume_from not in (None, *intervention.resume_points):
                raise Refused(
                    f"resume from one of: {', '.join(intervention.resume_points) or 'none'}"
                )
            if action != "resume":
                resume_from = None
            intervention.decision = Decision(
                action=action,
                operator=operator,
                note=note.strip() if note and note.strip() else None,
                resume_from=resume_from,
                at=_now(),
            )
            intervention.status = "decided"
            self._changed("decided", intervention, None)
            return intervention.model_copy(deep=True)

    # --- internals ----------------------------------------------------------------------

    def _record(self, state: ControlState, intervention: str | None, why: str) -> Transition:
        self._state = state
        transition = Transition(self._clock(), state, self._lease, intervention, why)
        self._transitions.append(transition)
        return transition

    def _changed(
        self, change: Change, intervention: Intervention, transition: Transition | None
    ) -> None:
        # Under the lock, so the two threads never write one intervention's file at once.
        if self._on_change is not None:
            self._on_change(change, intervention.model_copy(deep=True), transition)


def _new_id() -> str:
    # Letters only, like run ids: the redactor's digit detectors leave it alone.
    return "iv-" + "".join(secrets.choice(string.ascii_lowercase) for _ in range(8))


def _now() -> datetime:
    return datetime.now(UTC)
