"""Interventions: what automation asks a person for, and what the person decided.

Three kinds:
- `approval`: a commit needs a yes or a no before it runs. The approver does not
  take the session.
- `takeover`: automation reached a state it cannot handle safely (an unknown
  overlay, a stuck model). A person claims the live session, acts in it, then
  hands it back (resume), says the task is done (complete) or stops it (abort).
- `verify_commit`: a commit may have gone through and nothing definite came back.
  A person checks the app; automation never retries the commit.

Everything here is shown to an operator and written to the run folder, so the
fields are redacted before an intervention is opened.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from understudy.artifact.schema import Effect

InterventionKind = Literal["approval", "takeover", "verify_commit"]
OperatorAction = Literal["claim", "approve", "deny", "resume", "complete", "abort"]
DecidedAction = Literal["approve", "deny", "resume", "complete", "abort"]
Status = Literal["open", "claimed", "decided", "closed"]

ACTIONS: dict[InterventionKind, tuple[OperatorAction, ...]] = {
    "approval": ("approve", "deny"),
    "takeover": ("claim", "resume", "complete", "abort"),
    "verify_commit": ("claim", "complete", "abort"),
}
"""A commit whose outcome is unknown is never resumed: resuming could send it again."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Subject(_Model):
    """What the run is executing: an approved capability, or a discovery goal."""

    kind: Literal["capability", "goal"]
    id: str
    version: str | None = None


class Decision(_Model):
    action: DecidedAction
    operator: str
    note: str | None = None
    resume_from: str | None = None
    at: datetime


class Intervention(_Model):
    IDENTIFIERS: ClassVar[tuple[str, ...]] = (
        "id",
        "run_id",
        "kind",
        "artifact_hash",
        "status",
        "opened_at",
        "claimed_at",
        "closed_at",
    )
    """Top-level keys persisted verbatim: system identifiers, never data."""

    id: str
    run_id: str
    kind: InterventionKind
    subject: Subject
    artifact_hash: str | None = None
    step_id: str | None = None
    intent: str | None = None
    reason: str
    detail: str
    url: str | None = None
    screenshot: str | None = None
    """Masked, relative to the run folder: the page when the intervention opened."""
    live_screenshot: str | None = None
    """Masked, refreshed while a person decides, for an operator without the window."""
    recent_events: list[dict[str, Any]] = Field(default_factory=list)
    allowed: list[OperatorAction]
    resume_points: list[str] = Field(default_factory=list)
    """Steps a resume may restart from. The first is the paused step itself."""
    opened_at: datetime
    status: Status = "open"
    claimed_by: str | None = None
    claimed_at: datetime | None = None
    decision: Decision | None = None
    resolution: str | None = None
    """How automation closed it: approved, denied, resumed, completed, aborted,
    timed_out, or why it could not act on the decision."""
    closed_at: datetime | None = None


class HumanAction(_Model):
    """One input a person gave the live page while holding the lease. Redacted."""

    seq: int
    at: datetime
    kind: Literal["click", "change", "submit"]
    description: str
    effect: Effect
    value: str | None = None
    url: str | None = None
    verified: bool
    """Locators were synthesized and verified on the live element. False when the
    element was gone before it could be checked (a button that removed itself)."""
    in_frame: bool = False


class Handoff(_Model):
    """One intervention, as handoff.json records it: who, when, what they did, how it ended."""

    intervention_id: str
    kind: InterventionKind
    reason: str
    step_id: str | None
    operator: str | None
    opened_at: datetime
    claimed_at: datetime | None
    closed_at: datetime
    duration_ms: int
    """From opening to closing: how long automation waited."""
    actions: list[HumanAction] = Field(default_factory=list)
    unexpected_inputs: int = 0
    resolution: str
    note: str | None = None
    resume_from: str | None = None


class HandoffSummary(_Model):
    """One intervention, as a run's result lists it. The full record of a takeover
    is in handoff.json."""

    intervention_id: str
    kind: InterventionKind
    reason: str
    step_id: str | None
    operator: str | None
    resolution: str
    human_actions: int = 0


class InterventionQueue(Protocol):
    """Where a new intervention is announced. The built one tells the operator at
    the terminal where the console is; production would page someone or open a
    ticket. The console itself reads the control plane."""

    def publish(self, intervention: Intervention) -> None: ...
