"""The capability artifact: the contract between discovery and replay.

A calling agent reads `description`, `inputs`, `outputs`, `side_effects` and
`outcomes`. It never needs `steps`. Everything sensitive is a reference:
values are params or secret refs, the host is a tenant binding, and the
content hash is what approval binds to.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from understudy.artifact.signatures import Signature

Sensitivity = Literal["public", "financial_id", "pii", "secret"]
Effect = Literal["read", "input", "commit"]
ValueType = Literal["string", "integer", "decimal", "boolean", "enum", "date"]
Stability = Literal["high", "medium", "low"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- the agent-facing contract -------------------------------------------------


class InputSpec(_Model):
    name: str
    type: ValueType
    description: str
    required: bool = True
    sensitivity: Sensitivity = "public"
    enum: list[str] | None = None
    pattern: str | None = None
    min: float | None = None
    max: float | None = None
    example: str | None = None
    """A fake value, for the agent-facing tool definition. Never a real one."""

    @model_validator(mode="after")
    def _enum_has_values(self) -> InputSpec:
        if self.type == "enum" and not self.enum:
            raise ValueError(f"input {self.name!r}: type 'enum' needs enum values")
        return self


class OutputSpec(_Model):
    name: str
    type: ValueType
    description: str
    source_step: str
    sensitivity: Sensitivity = "public"
    pattern: str | None = None


class SideEffects(_Model):
    cls: Literal["read", "commit"] = Field(alias="class")
    description: str
    idempotent: bool = False
    approval: Literal["none", "per_invocation"] = "none"

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class OutcomeSpec(_Model):
    """A business outcome: the app worked and said no. Not a failure."""

    code: str
    description: str
    detected_by: str
    """The step id whose `expect` raises this outcome."""
    terminal: bool = True


# --- targets and locators ------------------------------------------------------


class Fingerprint(_Model):
    """Identity used to confirm that a fallback locator found the same element,
    not a look-alike. Checked on every resolution in replay."""

    role: str | None = None
    name: str | None = None
    tag: str | None = None
    input_type: str | None = None
    label: str | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> Fingerprint:
        if not any(v is not None for v in self.__dict__.values()):
            raise ValueError("a fingerprint needs at least one field")
        return self


class Verification(_Model):
    """Recorded at discovery: this locator resolved to exactly one element, and
    to the same element the model actually acted on."""

    unique: bool
    same_element: bool
    run_id: str


class Locator(_Model):
    by: Literal["test_id", "role", "label", "placeholder", "css", "text", "table_cell", "xpath"]
    value: str | dict[str, Any]
    stability: Stability
    note: str
    templated: bool = False
    verified: Verification | None = None


class Target(_Model):
    description: str
    fingerprint: Fingerprint
    locators: list[Locator] = Field(min_length=1)
    frame_path: list[str] = Field(default_factory=list)
    scope: Locator | None = None


# --- steps ---------------------------------------------------------------------


class ParamValue(_Model):
    param: str


class LiteralValue(_Model):
    literal: str


class SecretValue(_Model):
    secret_ref: str


ValueRef = ParamValue | LiteralValue | SecretValue

Action = Literal["navigate", "click", "fill", "select", "check", "press", "extract", "wait_for"]


class OutcomeRef(_Model):
    outcome: str


class Disposition(_Model):
    """What a step does when a signature matches, or when none does."""

    do: Literal["continue", "succeed", "business", "fail", "escalate"]
    code: str | None = None

    @model_validator(mode="after")
    def _code_where_needed(self) -> Disposition:
        needs_code = self.do in ("business", "fail", "escalate")
        if needs_code and not self.code:
            raise ValueError(f"disposition {self.do!r} needs a code")
        if not needs_code and self.code:
            raise ValueError(f"disposition {self.do!r} takes no code")
        return self


class Expectation(_Model):
    when: Signature
    then: Disposition
    description: str


class RetryPolicy(_Model):
    max_attempts: int = Field(default=0, ge=0, le=5)
    backoff_ms: list[int] = Field(default_factory=list)


class Extraction(_Model):
    output: str
    read: Literal["text", "value", "attr"] = "text"
    attr: str | None = None
    parse: Literal["regex", "integer", "decimal"] | None = None
    regex: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Extraction:
        if self.read == "attr" and not self.attr:
            raise ValueError("extract read='attr' needs an attr name")
        if self.parse == "regex" and not self.regex:
            raise ValueError("extract parse='regex' needs a regex")
        return self


class Step(_Model):
    id: str
    intent: str
    action: Action
    effect: Effect
    target: Target | None = None
    url: str | None = None
    value: ValueRef | None = None
    wait_before: Signature | None = None
    expect: list[Expectation] = Field(default_factory=list)
    on_no_match: Disposition | None = None
    """Defaults by effect: read/input fail as unknown_state, commit escalates as
    commit_outcome_unknown. Resolved by `resolved_on_no_match`, never silently."""
    on_target_absent: OutcomeRef | None = None
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_ms: int = Field(default=10_000, ge=100, le=120_000)
    reentry_point: bool = False
    extract: Extraction | None = None

    @model_validator(mode="after")
    def _action_shape(self) -> Step:
        if self.action == "navigate":
            if not self.url:
                raise ValueError(f"step {self.id}: navigate needs a url")
            if self.target:
                raise ValueError(f"step {self.id}: navigate takes no target")
        else:
            if self.url:
                raise ValueError(f"step {self.id}: only navigate takes a url")
            if not self.target:
                raise ValueError(f"step {self.id}: {self.action} needs a target")
        if self.action in ("fill", "select", "check", "press") and not self.value:
            raise ValueError(f"step {self.id}: {self.action} needs a value")
        if self.action == "extract" and not self.extract:
            raise ValueError(f"step {self.id}: extract needs an extract spec")
        if self.extract and self.action != "extract":
            raise ValueError(f"step {self.id}: only extract steps carry an extract spec")
        if self.reentry_point and self.action != "navigate":
            # Replay restarts here after a recovery, which only works from a fresh page.
            raise ValueError(f"step {self.id}: a re-entry point must be a navigate step")
        return self

    @property
    def resolved_on_no_match(self) -> Disposition:
        """The declared disposition, or the effect's default. A commit whose outcome
        is unknown goes to a human; it is never retried and never assumed."""
        if self.on_no_match:
            return self.on_no_match
        if self.effect == "commit":
            return Disposition(do="escalate", code="commit_outcome_unknown")
        return Disposition(do="fail", code="unknown_state")


# --- checkpoints, recovery, provenance, lifecycle ------------------------------


class Checkpoint(_Model):
    signature: Signature
    description: str
    negative_control: Signature | None = None
    """A signature that must hold on the start state. Lint (L001) uses it to prove
    the success checkpoint is not vacuously true before the flow has run."""


class RecoveryPolicy(_Model):
    allowed: list[Literal["retry_transient", "reauthenticate", "dismiss_interstitial"]] = Field(
        default_factory=list
    )
    budget: int = Field(default=3, ge=0, le=10)


class LocatorStats(_Model):
    targets: int = 0
    locators: int = 0
    verified_unique: int = 0


class Provenance(_Model):
    discovery_run_id: str
    recorded_against: str
    """The tenant id only. Never a base URL."""
    goal_template: str | None = None
    models_served: list[str] = Field(default_factory=list)
    human_steps: list[str] = Field(default_factory=list)
    locator_stats: LocatorStats = Field(default_factory=LocatorStats)
    authored: Literal["discovered", "hand_written"] = "discovered"


class Lifecycle(_Model):
    status: Literal["draft", "approved", "deprecated"] = "draft"
    reviewed_by: str | None = None
    approved_at: datetime | None = None
    approved_hash: str | None = None

    @model_validator(mode="after")
    def _approved_is_complete(self) -> Lifecycle:
        if self.status == "approved" and not (
            self.reviewed_by and self.approved_at and self.approved_hash
        ):
            raise ValueError(
                "an approved artifact needs reviewed_by, approved_at and approved_hash"
            )
        return self


class AppBinding(_Model):
    vendor_app: str
    surface: Literal["web"] = "web"
    entry_route: str
    compatible_versions: list[str] = Field(default_factory=list)


class Requirements(_Model):
    session: str
    """A session provider id, e.g. "parabank.customer". Login is never a step."""
    secrets: list[str] = Field(default_factory=list)


class Capability(_Model):
    schema_version: Literal["understudy.capability/v1"] = "understudy.capability/v1"
    id: str = Field(pattern=r"^[a-z0-9_]+\.[a-z0-9_]+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str
    description: str
    lifecycle: Lifecycle
    app: AppBinding
    requires: Requirements
    inputs: list[InputSpec]
    outputs: list[OutputSpec]
    side_effects: SideEffects
    steps: list[Step] = Field(min_length=1)
    success: Checkpoint
    outcomes: list[OutcomeSpec] = Field(default_factory=list)
    recovery: RecoveryPolicy = Field(default_factory=RecoveryPolicy)
    provenance: Provenance

    @model_validator(mode="after")
    def _unique_names(self) -> Capability:
        for label, names in (
            ("step id", [s.id for s in self.steps]),
            ("input", [i.name for i in self.inputs]),
            ("output", [o.name for o in self.outputs]),
            ("outcome", [o.code for o in self.outcomes]),
        ):
            duplicates = {n for n in names if names.count(n) > 1}
            if duplicates:
                raise ValueError(f"duplicate {label}: {', '.join(sorted(duplicates))}")
        return self

    def step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)


CapabilityRef = Annotated[Capability, Field()]
