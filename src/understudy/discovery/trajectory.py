"""The trajectory: what a discovery run did, call by call, in the form the compiler reads.

The recorder builds it while the model works. Element targets are synthesized
and verified on the live page *before* each action, because the element may be
gone afterwards (a submit button hidden by its own result panel). Typed values are
bound to goal params as they are recorded, so a value that is not a param and not
public is withheld here rather than kept for later.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from playwright.sync_api import Locator as PwLocator
from pydantic import BaseModel, ConfigDict, Field

from understudy.artifact.schema import Effect, Fingerprint, Sensitivity, Target
from understudy.discovery.goal import GoalSpec
from understudy.evidence.run import RunContext
from understudy.redaction.detectors import JSESSIONID
from understudy.redaction.redactor import Redactor
from understudy.surface import capture
from understudy.surface.base import Observation
from understudy.surface.web import WebSurface

ACTION_MEMBERS = frozenset({"navigate", "left_click", "form_input", "type", "key"})
"""Members that change the page and become steps. Everything else only looks."""

_MODEL_REF = re.compile(r" ?\[ref=ref_\d+\]")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PageState(_Model):
    url: str
    """Session ids removed."""
    route: str | None
    """Relative to the tenant's base URL; None outside the app."""
    title: str
    snapshot: str
    """The model-facing accessibility snapshot."""
    status: int | None = None

    @property
    def key(self) -> str:
        """Identity of the page's content, refs aside: equal keys mean nothing changed."""
        body = f"{self.url}\n{_MODEL_REF.sub('', self.snapshot)}"
        return hashlib.sha1(body.encode("utf-8")).hexdigest()


class ElementRecord(_Model):
    description: str
    tag: str
    role: str | None = None
    name: str | None = None
    input_type: str | None = None
    sensitivity: Sensitivity = "public"
    """How the field's own label and type classify what is typed into it."""
    target: Target | None = None
    """Verified locators. None when no candidate was unique and the same element."""
    rejected: list[tuple[str, str]] = Field(default_factory=list)
    selected: tuple[str, str] | None = None
    """For a select after the action: the chosen option's (value, label)."""

    @property
    def key(self) -> str | None:
        """Identity across calls in one run: the first verified locator and its frame."""
        if self.target is None:
            return None
        first = self.target.locators[0]
        return json.dumps([self.target.frame_path, first.by, first.value], sort_keys=True)

    @property
    def is_form_control(self) -> bool:
        return self.tag in ("select", "textarea") or (
            self.tag == "input"
            and self.input_type not in ("button", "submit", "image", "reset", "hidden")
        )


class StepValue(_Model):
    """What was entered. A param or a public literal; neither means it was withheld."""

    param: str | None = None
    literal: str | None = None
    sensitivity: Sensitivity = "public"


class RecordedCall(_Model):
    seq: int
    turn: int
    member: str
    actor: Literal["agent", "human"] = "agent"
    ok: bool
    error: str | None = None
    dispatched: bool = False
    """The action failed in the browser call that delivers it: the input may have
    reached the page anyway."""
    effect: Effect | None = None
    approved: bool = False
    element: ElementRecord | None = None
    value: StepValue | None = None
    key: str | None = None
    url: str | None = None
    """navigate: the address it went to, session id removed."""
    before: PageState | None = None
    after: PageState | None = None
    observed_after: PageState | None = None
    """The last page the model looked at before its next action. The page it judged
    the action by, so the compiler prefers it to `after` for the step's checkpoint."""
    rationale: str | None = None
    duration_ms: int = 0

    @property
    def result_state(self) -> PageState | None:
        return self.observed_after or self.after


class RecordedOutput(_Model):
    name: str
    description: str
    element: ElementRecord
    sample: str
    """The value read at discovery. Persisted redacted by the output's sensitivity."""
    after_seq: int
    """The last call before it was recorded; the extract step goes after that one."""


class SuccessClaim(_Model):
    summary: str
    evidence: list[ElementRecord]
    page: PageState


class Trajectory(_Model):
    run_id: str
    tenant: str
    capability: str
    goal: str
    params: dict[str, str]
    start: PageState | None = None
    calls: list[RecordedCall] = Field(default_factory=list)
    outputs: list[RecordedOutput] = Field(default_factory=list)
    success: SuccessClaim | None = None

    def actions(self) -> list[RecordedCall]:
        return [c for c in self.calls if c.member in ACTION_MEMBERS]


class Recorder:
    def __init__(
        self,
        run: RunContext,
        surface: WebSurface,
        goal: GoalSpec,
        values: Mapping[str, str],
        *,
        masks: Sequence[str] = (),
        screenshots: bool = True,
    ) -> None:
        self.run = run
        self.surface = surface
        self.gate = surface.gate
        self.goal = goal
        self.values = dict(values)
        self.masks = list(masks)
        self.screenshots = screenshots
        self.turn = 0
        self.rationale: str | None = None
        self._params = {value: name for name, value in self.values.items()}
        self._sensitivity = {p.name: p.sensitivity for p in goal.params}
        self._masked: set[str] = set()
        self._seq = 0
        self.trajectory = Trajectory(
            run_id=run.run_id,
            tenant=self.gate.tenant.id,
            capability=goal.capability,
            goal=goal.render(values),
            params=self.values,
        )

    # --- turns and page states ------------------------------------------------------

    def begin_turn(self, turn: int, rationale: str | None) -> None:
        self.turn = turn
        self.rationale = rationale

    def start(self) -> PageState:
        state = self.capture()
        self.trajectory.start = state
        return state

    def capture(self, observation: Observation | None = None) -> PageState:
        observation = observation or self.surface.observe()
        # Evidence masking never fails a run; the snapshot is redacted when written anyway.
        with contextlib.suppress(Exception):
            self._masked |= capture.masked_texts(self.surface.page, self.masks)
        return PageState(
            url=JSESSIONID.sub("", observation.url),
            route=self.gate.route_of(observation.url),
            title=observation.title,
            snapshot=observation.snapshot,
            status=observation.status,
        )

    # --- elements and values ----------------------------------------------------------

    def element(
        self,
        locator: PwLocator,
        *,
        description: str | None = None,
        strip: str | None = None,
    ) -> ElementRecord:
        """Synthesize and verify locators for an element. `strip` is a data value the
        element shows (an output's sample): locators and fingerprint fields containing
        it are dropped, because the next run will show a different value."""
        synthesis = self.surface.synthesize(locator, params=self.values, description=description)
        facts = synthesis.facts
        target = synthesis.target
        rejected = list(synthesis.rejected)
        if target is not None and strip:
            target, dropped = _without_value(target, strip)
            rejected += [
                (by, "contains the value it shows, which changes every run") for by in dropped
            ]
        if target is None:
            rejected.append(("target", "no candidate locator was unique and the same element"))
        else:
            target = target.model_copy(update={"description": _short(target.description)})
        name = facts.name if not (strip and facts.name and strip in facts.name) else None
        return ElementRecord(
            description=target.description if target else (description or facts.tag),
            tag=facts.tag,
            role=facts.role,
            name=name,
            input_type=facts.input_type,
            sensitivity=Redactor.classify_field(
                input_type=facts.input_type,
                label=facts.label,
                name_attr=facts.name_attr,
                autocomplete=facts.autocomplete,
            ),
            target=target,
            rejected=rejected,
        )

    def bind(self, candidates: Sequence[str], field: Sensitivity) -> StepValue:
        """The first candidate that is a param's value names the param. Otherwise a
        value is kept only when both the field and the value itself look public."""
        for candidate in candidates:
            name = self._params.get(candidate)
            if name is not None:
                return StepValue(param=name, sensitivity=self._sensitivity.get(name, "public"))
        value = candidates[0] if candidates else ""
        if field == "public" and self.run.redactor.text(value) == value:
            return StepValue(literal=value)
        return StepValue(sensitivity=field if field != "public" else "pii")

    # --- calls ---------------------------------------------------------------------------

    def call(self, call: RecordedCall) -> RecordedCall:
        self.trajectory.calls.append(call)
        value, sensitivity = self._event_value(call.value)
        target = call.element.target if call.element else None
        self.run.event(
            "act",
            actor=call.actor,
            member=call.member,
            target=target,
            value=value,
            value_sensitivity=sensitivity,
            url=call.after.url if call.after else (call.before.url if call.before else None),
            rationale=call.rationale,
            duration_ms=call.duration_ms,
            ok=call.ok,
            error=call.error,
            seq=call.seq,
            turn=call.turn,
            effect=call.effect,
            approved=call.approved,
            dispatched=call.dispatched,
        )
        if self.screenshots and call.ok and call.member in ACTION_MEMBERS:
            with contextlib.suppress(Exception):  # evidence never fails a run
                self.surface.screenshot(
                    self.run.screenshot_path(f"{call.seq}_{call.member}"), self.masks
                )
        return call

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def observed(self, state: PageState) -> None:
        """A look at the page after an action: the state the model judged it by."""
        actions = [c for c in self.trajectory.calls if c.member in ACTION_MEMBERS and c.ok]
        if actions:
            actions[-1].observed_after = state

    def refused(self, member: str, reason: str) -> None:
        self.call(
            RecordedCall(
                seq=self.next_seq(),
                turn=self.turn,
                member=member,
                ok=False,
                error=reason,
                rationale=self.rationale,
            )
        )

    def approval(self, member: str, target: str, reason: str, granted: bool) -> None:
        self.run.event(
            "control",
            actor="human",
            member=member,
            ok=granted,
            approval="granted" if granted else "refused",
            target_description=target,
            reason=reason,
        )

    def output(self, name: str, description: str, element: ElementRecord, sample: str) -> None:
        sensitivity: Sensitivity = next(
            (o.sensitivity for o in self.goal.outputs if o.name == name), "pii"
        )
        self.run.know({sample: sensitivity})
        self.trajectory.outputs = [o for o in self.trajectory.outputs if o.name != name]
        self.trajectory.outputs.append(
            RecordedOutput(
                name=name,
                description=description,
                element=element,
                sample=sample,
                after_seq=self._seq,
            )
        )
        self.run.event("observe", actor="agent", member="record_output", output=name, ok=True)

    def success(self, summary: str, evidence: list[ElementRecord], page: PageState) -> None:
        self.trajectory.success = SuccessClaim(summary=summary, evidence=evidence, page=page)
        self.run.event("decide", actor="agent", member="declare_success", rationale=summary)

    # --- persistence -----------------------------------------------------------------------

    def persist(self) -> None:
        """The trajectory as evidence: snapshots redacted as snapshots (typed values
        dropped, digit runs masked, the profile's masked panels hidden), samples by
        their output's sensitivity, everything else as free text."""
        dumped = self.trajectory.model_dump(mode="json")
        redactor = self.run.redactor
        outputs = {o.name: o.sensitivity for o in self.goal.outputs}

        def scrub(node: Any) -> Any:
            if isinstance(node, dict):
                out = {k: scrub(v) for k, v in node.items()}
                if isinstance(out.get("snapshot"), str):
                    snapshot = capture.mask_snapshot(out["snapshot"], self._masked)
                    out["snapshot"] = redactor.snapshot(snapshot)
                return out
            if isinstance(node, list):
                return [scrub(v) for v in node]
            return node

        scrubbed = scrub(dumped)
        for output in scrubbed["outputs"]:
            output["sample"] = redactor.value(output["sample"], outputs.get(output["name"], "pii"))
        scrubbed["params"] = {
            name: redactor.value(value, self._sensitivity.get(name, "pii"))
            for name, value in self.values.items()
        }
        self.run.write_json(
            "trajectory.json", scrubbed, verbatim=("run_id", "tenant", "capability")
        )

    def _event_value(self, value: StepValue | None) -> tuple[str | None, Sensitivity]:
        if value is None:
            return None, "public"
        if value.param is not None:
            return self.values.get(value.param), value.sensitivity
        if value.literal is not None:
            return value.literal, "public"
        return "[withheld]", "public"


def _short(text: str, limit: int = 60) -> str:
    """Target descriptions are for reviewers; text placed beside a control can be a
    whole paragraph."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:'\"")
    return cut + ("…'" if text.endswith("'") else "…")


def elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _without_value(target: Target, value: str) -> tuple[Target | None, list[str]]:
    def carries(part: object) -> bool:
        if isinstance(part, str):
            return value in part
        if isinstance(part, dict):
            return any(carries(v) for v in part.values())
        return False

    kept = [loc for loc in target.locators if not carries(loc.value)]
    dropped: list[str] = [loc.by for loc in target.locators if carries(loc.value)]
    fields = target.fingerprint.model_dump(exclude_none=True)
    clean = {k: v for k, v in fields.items() if not carries(v)}
    if not kept or not clean:
        return None, dropped
    return (
        target.model_copy(update={"locators": kept, "fingerprint": Fingerprint(**clean)}),
        dropped,
    )
