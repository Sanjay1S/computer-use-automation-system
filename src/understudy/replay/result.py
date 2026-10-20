"""The replay result: what a caller gets back from every run, whatever happened.

Exactly one of `outputs` (succeeded), `outcome` (business_outcome), `rejection`
(rejected), `escalation` (needs_human) or `failure` (failed) explains the status;
`needs_human` may also carry the `failure` that led to it. Recoverable conditions
never become the status: they are listed in `recoveries` and the run goes on.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["succeeded", "business_outcome", "rejected", "needs_human", "failed"]
CommitState = Literal["not_attempted", "committed", "refused", "unknown"]
"""`refused`: the commit was sent and the app definitely said no (a business outcome
after the commit). `unknown`: it was sent, or may have been, and nothing definite
came back. Money may have moved, so that goes to a human, never to a retry."""
FailureCategory = Literal["app", "automation", "infra", "policy"]
RejectionCode = Literal[
    "invalid_input", "invalid_artifact", "not_approved", "approval_required", "policy"
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BusinessOutcome(_Model):
    """The app worked and said no. Nothing failed."""

    code: str
    message: str
    """The artifact's description of the outcome, as declared to the caller."""
    step_id: str


class Rejection(_Model):
    """Refused before anything touched the UI."""

    code: RejectionCode
    problems: list[str]


class Attempt(_Model):
    index: int
    by: str
    outcome: str
    count: int = 0
    detail: str = ""


class Evidence(_Model):
    """Paths relative to the run directory."""

    screenshot: str | None = None
    dom: str | None = None
    aria: str | None = None
    console: str | None = None
    dialog: str | None = None
    """The native dialogs the page showed, with the answers they got."""


class Escalation(_Model):
    reason: str
    step_id: str | None
    detail: str
    evidence: Evidence | None = None
    intervention_id: str | None = None
    """Set once an intervention is opened for an operator (M6)."""
    resolution: str | None = None


class Failure(_Model):
    category: FailureCategory
    code: str
    step_id: str | None
    expected: str
    observed: str
    attempts: list[Attempt] = Field(default_factory=list)
    evidence: Evidence | None = None


class SideEffectReport(_Model):
    commit_state: CommitState = "not_attempted"
    commit_step: str | None = None
    commit_requests: int = 0
    """Requests to the policy's commit routes that the page issued and the policy let
    through. Evidence, not proof: not that the server received them, nor what it did."""


class RecoveryEvent(_Model):
    handler: Literal["retry_transient", "reauthenticate", "dismiss_interstitial"]
    step_id: str
    reason: str
    restarted_from: str | None = None


class DriftEvent(_Model):
    """A fallback locator found the target: the step went on, the artifact needs a look."""

    step_id: str
    locator_index: int
    by: str
    attempts: list[Attempt]


class StepTiming(_Model):
    step_id: str
    attempt: int
    duration_ms: int


class Timings(_Model):
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    steps: list[StepTiming] = Field(default_factory=list)


class ReplayResult(_Model):
    status: Status
    capability: str
    version: str
    artifact_hash: str
    tenant: str
    run_id: str
    outputs: dict[str, Any] | None = None
    """Raw for the caller. The copy in result.json is redacted."""
    outcome: BusinessOutcome | None = None
    rejection: Rejection | None = None
    escalation: Escalation | None = None
    failure: Failure | None = None
    side_effects: SideEffectReport = Field(default_factory=SideEffectReport)
    recoveries: list[RecoveryEvent] = Field(default_factory=list)
    drift: list[DriftEvent] = Field(default_factory=list)
    completed_by: Literal["automation", "human"] = "automation"
    timings: Timings

    IDENTIFIERS: ClassVar[tuple[str, ...]] = (
        "run_id",
        "capability",
        "version",
        "artifact_hash",
        "tenant",
        "status",
    )
    """Top-level keys persisted verbatim: system identifiers, never data."""
