"""The compiler: a recorded trajectory becomes a draft capability artifact.

The rules, in order:
1. keep only actions that ran (no reads, screenshots, waits, refusals or failures);
2. coalesce: a click into a field followed by entering a value there is one step,
   a value entered twice in a row is the last one, and Tab presses are dropped;
3. canonicalize addresses: no session id, the host becomes {base_url}, param values
   in the query become {param};
4. values the model entered are already bound to params by the recorder;
5. each step's checkpoint comes from what changed on the page after it, never from
   data-dependent text unless that text is a param;
6. the success checkpoint comes from the model's evidence, minus anything that was
   already true when the flow began, with a negative control;
7. recorded outputs become extract steps;
8. a select bound to a free-form param gets a `<param>_not_found` business outcome;
9. effects, retries and the re-entry point follow the policy and the commit rule;
10. lint runs, and its findings go to the review summary with the compiler's notes.

A person's actions during a takeover compile like the model's and are listed in
`provenance.human_steps`, which the review summary flags.

The draft is never approved here: a person reviews it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict
from urllib.parse import parse_qsl, quote, urlsplit

from understudy.artifact import signatures as sig
from understudy.artifact.lint import LintReport, lint
from understudy.artifact.schema import (
    AppBinding,
    Capability,
    Checkpoint,
    Disposition,
    Effect,
    Expectation,
    Extraction,
    Lifecycle,
    LiteralValue,
    LocatorStats,
    OutcomeRef,
    OutcomeSpec,
    OutputSpec,
    ParamValue,
    Provenance,
    RecoveryPolicy,
    Requirements,
    RetryPolicy,
    SecretValue,
    SideEffects,
    Step,
    Target,
    ValueRef,
)
from understudy.discovery.goal import GoalSpec
from understudy.discovery.trajectory import (
    ElementRecord,
    PageState,
    RecordedCall,
    RecordedOutput,
    Trajectory,
)
from understudy.policy.gate import PolicyGate
from understudy.redaction.detectors import JSESSIONID
from understudy.surface.snapshot import parse_line

_LEVEL = re.compile(r"\[level=(\d+)\]")
_TEXT_LINE = re.compile(
    r"^[ \t]*- (?:paragraph|text|status|alert|strong|emphasis)(?: \[[^\]]*\])*: (.+)$"
)
_DATA = re.compile(r"\d{3,}")
_TEXT_MAX = 80
_NOTICE_ROLES = frozenset({"alert", "status", "dialog", "alertdialog", "button", "link"})


class _Fields(TypedDict):
    effect: Effect
    target: Target
    timeout_ms: int
    retry: RetryPolicy


@dataclass(frozen=True)
class _Shape:
    """What every element step shares. A commit is never retried (L003)."""

    effect: Effect
    target: Target
    timeout_ms: int

    def fields(self) -> _Fields:
        return _Fields(
            effect=self.effect,
            target=self.target,
            timeout_ms=self.timeout_ms,
            retry=RetryPolicy(max_attempts=0),
        )


class CompileError(Exception):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Compilation:
    capability: Capability
    notes: list[str]
    report: LintReport


def compile_trajectory(
    trajectory: Trajectory,
    goal: GoalSpec,
    gate: PolicyGate,
    *,
    models_served: Sequence[str] = (),
) -> Compilation:
    return _Compiler(trajectory, goal, gate, models_served).run()


class _Compiler:
    def __init__(
        self,
        trajectory: Trajectory,
        goal: GoalSpec,
        gate: PolicyGate,
        models_served: Sequence[str],
    ) -> None:
        self.t = trajectory
        self.goal = goal
        self.gate = gate
        self.models_served = list(models_served)
        self.values = dict(trajectory.params)
        self.notes: list[str] = []
        self.problems: list[str] = []
        self.steps: list[Step] = []
        self.outcomes: list[OutcomeSpec] = []
        self.outputs: list[OutputSpec] = []
        self.human_steps: list[str] = []
        self.commit_seen = False

    def run(self) -> Compilation:
        if self.t.success is None:
            raise CompileError(["the run never declared success, so there is nothing to compile"])
        for call in self.t.calls:
            if call.approved and call.dispatched and not call.ok:
                self.notes.append(
                    f"call {call.seq}: an approved commit failed after its input may have "
                    "reached the app; check the app's state before replaying this draft"
                )
        actions = self._coalesce([c for c in self.t.actions() if c.ok])
        if not actions:
            raise CompileError(["the run took no actions"])
        if actions[0].member != "navigate":
            self.notes.append(
                "the flow did not begin with an address; a navigate to the page where the "
                "first action happened was added as the re-entry point"
            )
        entry_state = self._entry(actions[0])
        outputs = sorted(self.t.outputs, key=lambda o: o.after_seq)
        for call in actions:
            if call.member == "navigate":
                self._navigate(call)
            else:
                self._action(call)
            while outputs and outputs[0].after_seq < _next_seq(actions, call):
                self._extract(outputs.pop(0))
        for output in outputs:
            self._extract(output)
        if self.problems:
            raise CompileError(self.problems)
        if self.human_steps:
            self.notes.append(
                f"steps {', '.join(self.human_steps)} were recorded from a person's actions: "
                "check that each belongs in the flow and is not only the clearing of something "
                "in the way"
            )
        success = self._success(entry_state)
        capability = self._capability(success)
        report = lint(capability, self.gate)
        return Compilation(capability, self.notes, report)

    # --- 1-2: which calls become steps ------------------------------------------------------

    def _coalesce(self, calls: list[RecordedCall]) -> list[RecordedCall]:
        kept: list[RecordedCall] = []
        for index, call in enumerate(calls):
            following = calls[index + 1] if index + 1 < len(calls) else None
            same_next = (
                following is not None
                and following.member in ("form_input", "type")
                and _same_element(call, following)
            )
            if (
                call.member == "left_click"
                and call.element
                and call.element.is_form_control
                and same_next
            ):
                continue  # clicking into a field is part of entering its value
            if call.member in ("form_input", "type") and same_next:
                continue  # only the last value entered counts
            if call.member == "key" and call.key == "Tab":
                self.notes.append(
                    f"call {call.seq}: a Tab press was dropped (replay acts on fields directly)"
                )
                continue
            kept.append(call)
        return kept

    def _entry(self, first: RecordedCall) -> PageState:
        """Adds the leading navigate when the first action was not one, and returns
        the page the flow starts from (where the negative control must hold)."""
        if first.member == "navigate":
            return first.result_state or _required_state(first.after, first.seq)
        state = _required_state(first.before, first.seq)
        if state.route is None:
            raise CompileError(
                [f"call {first.seq} happened outside the app, so there is no entry point"]
            )
        self.steps.append(
            Step(
                id=self._id("go", _route_slug(state.route)),
                intent=f"Go to {state.route}",
                action="navigate",
                effect="read",
                url=self._url(state.url, first.seq),
                reentry_point=True,
                timeout_ms=15_000,
                retry=RetryPolicy(max_attempts=2, backoff_ms=[500, 1_000]),
                expect=_arrival(state, None, self.values),
            )
        )
        return state

    # --- 3-5, 9: steps -----------------------------------------------------------------------

    def _navigate(self, call: RecordedCall) -> None:
        state = _required_state(call.result_state, call.seq)
        first = not self.steps
        self.steps.append(
            Step(
                id=self._id("go", _route_slug(state.route or "page")),
                intent=f"Go to {state.route or 'the page'}",
                action="navigate",
                effect="read",
                url=self._url(call.url or state.url, call.seq),
                reentry_point=first,
                timeout_ms=15_000,
                retry=RetryPolicy(max_attempts=2, backoff_ms=[500, 1_000]),
                expect=_arrival(state, call.before, self.values),
            )
        )

    def _action(self, call: RecordedCall) -> None:
        element = call.element
        if (
            call.member == "key"
            and call.effect != "commit"
            and (element is None or element.tag == "body")
        ):
            self.notes.append(
                f"call {call.seq}: a {call.key} press on the page itself (no field had focus) "
                "was dropped"
            )
            return
        if call.actor == "human" and (element is None or element.target is None):
            # No element description here: for an element that was gone it is raw page
            # text, and this note goes to review.txt unredacted. trajectory.json has it.
            self.notes.append(
                f"call {call.seq}: a person's {call.member} was not compiled; its element was "
                "gone before its locators could be verified. Check whether the flow needs it"
            )
            return
        if element is None or element.target is None:
            why = "; ".join(
                f"{by}: {reason}" for by, reason in (element.rejected if element else [])
            )
            self.problems.append(
                f"call {call.seq} ({call.member}): no locator was verified for its element"
                + (f" ({why})" if why else "")
            )
            return
        target = element.target
        effect = call.effect or "input"
        if call.member in ("form_input", "type"):
            effect = "commit" if effect == "commit" else "input"
        commit = effect == "commit"
        shape = _Shape(effect, target, 30_000 if commit else 10_000)
        if call.member == "left_click":
            step = Step(
                id=self._id("click", _slug(element.name or element.description)),
                intent=f"Click {target.description}" + ("; this commits" if commit else ""),
                action="click",
                expect=_changes(call.before, call.result_state, self.values),
                **shape.fields(),
            )
        elif call.member == "key":
            step = Step(
                id=self._id("press", _slug(call.key or "key")),
                intent=f"Press {call.key} in {target.description}"
                + ("; this commits" if commit else ""),
                action="press",
                value=LiteralValue(literal=call.key or ""),
                expect=_changes(call.before, call.result_state, self.values),
                **shape.fields(),
            )
        elif element.tag == "select":
            step = self._select(call, element, shape)
        else:
            value = self._value(call)
            check = element.input_type in ("checkbox", "radio")
            name = value.param if isinstance(value, ParamValue) else "the value"
            step = Step(
                id=self._id("set" if check else "fill", _slug(_value_slug(value, element))),
                intent=(
                    f"Set {target.description}"
                    if check
                    else f"Enter {name} into {target.description}"
                ),
                action="check" if check else "fill",
                value=value,
                **shape.fields(),
            )
        if call.actor == "human":
            self.human_steps.append(step.id)
        if commit:
            self.commit_seen = True
            if not step.expect:
                self.notes.append(
                    f"step {step.id} commits, but nothing visibly changed after it was recorded; "
                    "lint will require an expectation before approval"
                )
        self.steps.append(step)

    def _select(self, call: RecordedCall, element: ElementRecord, shape: _Shape) -> Step:
        target = shape.target
        value = self._value(call)
        css = _page_css(target)
        step_id = self._id("select", _slug(_value_slug(value, element)))
        expect: list[Expectation] = []
        wait_before: sig.Signature | None = None
        absent: OutcomeRef | None = None
        if css is not None:
            wait_before = sig.OptionsLoadedCondition(target=css, min=1)
            if isinstance(value, ParamValue):
                expect.append(
                    Expectation(
                        description=f"The list holds the requested {value.param}",
                        when=sig.SelectedCondition(target=css, param=value.param),
                        then=Disposition(do="continue"),
                    )
                )
            elif isinstance(value, LiteralValue):
                expect.append(
                    Expectation(
                        description="The list holds the recorded value",
                        when=sig.SelectedCondition(target=css, value=value.literal),
                        then=Disposition(do="continue"),
                    )
                )
            absent = self._not_found(step_id, value)
        else:
            self.notes.append(
                f"step {step_id}: the list has no page-unique CSS locator, so its readiness and "
                "selection cannot be checked; replay will act as soon as the list is found"
            )
        name = value.param if isinstance(value, ParamValue) else "the value"
        return Step(
            id=step_id,
            intent=f"Choose {name} in {target.description}",
            action="select",
            value=value,
            wait_before=wait_before,
            on_target_absent=absent,
            expect=expect,
            **shape.fields(),
        )

    def _not_found(self, step_id: str, value: ValueRef) -> OutcomeRef | None:
        """A free-form param missing from a loaded list is the app's answer (that is
        not one of yours), not a broken step. Enum params are checked in preflight, so
        an enum value missing from its list would be drift, not an outcome."""
        if not isinstance(value, ParamValue):
            return None
        spec = next((p for p in self.goal.params if p.name == value.param), None)
        if spec is None or spec.type == "enum":
            return None
        code = f"{value.param}_not_found"
        nothing = (
            "Nothing was committed." if not self.commit_seen else "This came after the commit."
        )
        self.outcomes.append(
            OutcomeSpec(
                code=code,
                detected_by=step_id,
                description=(
                    f"The list of choices loaded, but none of them is the requested {value.param}. "
                    + nothing
                ),
            )
        )
        return OutcomeRef(outcome=code)

    def _value(self, call: RecordedCall) -> ValueRef:
        value = call.value
        if value is not None and value.param is not None:
            return ParamValue(param=value.param)
        if value is not None and value.literal is not None:
            return LiteralValue(literal=value.literal)
        ref = f"withheld_{call.seq}"
        self.notes.append(
            f"call {call.seq}: a value that is neither a goal param nor public was entered; it "
            f"was withheld as secret_ref {ref!r}. Make it a param, or name a real secret."
        )
        return SecretValue(secret_ref=ref)

    # --- 7: outputs ------------------------------------------------------------------------------

    def _extract(self, output: RecordedOutput) -> None:
        spec = next((o for o in self.goal.outputs if o.name == output.name), None)
        if spec is None:
            self.problems.append(f"output {output.name!r} is not one the goal requested")
            return
        target = output.element.target
        if target is None:
            self.problems.append(f"output {output.name!r}: no locator was verified for its element")
            return
        extraction = _extraction(output.name, output.sample, spec.type, output.element)
        step_id = self._id("read", output.name)
        if spec.pattern and re.search(spec.pattern, output.sample) is None:
            self.notes.append(
                f"output {output.name}: the discovery sample does not match {spec.pattern}"
            )
        expect: list[Expectation] = []
        css = _page_css(target)
        if css is not None:
            expect.append(
                Expectation(
                    description=f"The element showing {output.name} is visible",
                    when=sig.VisibleCondition(css=css),
                    then=Disposition(do="continue"),
                )
            )
        self.steps.append(
            Step(
                id=step_id,
                intent=f"Read {output.name} from {target.description}",
                action="extract",
                effect="read",
                target=target,
                extract=extraction,
                expect=expect,
                retry=RetryPolicy(max_attempts=2, backoff_ms=[250, 500]),
            )
        )
        self.outputs.append(spec.output_spec(step_id))

    # --- 6: success ------------------------------------------------------------------------------

    def _success(self, entry: PageState) -> Checkpoint:
        assert self.t.success is not None
        start_items = _items(entry.snapshot)
        conditions: list[sig.Signature] = []
        for record in self.t.success.evidence:
            condition = _evidence(record, self.values)
            if condition is None:
                self.notes.append(
                    f"success evidence {record.description!r} has no stable condition"
                )
                continue
            if isinstance(condition, sig.VisibleCondition) and _holds_in(condition, start_items):
                self.notes.append(
                    f"success evidence {record.description!r} was already showing when the flow "
                    "began, so it proves nothing and was left out"
                )
                continue
            if condition not in conditions:
                conditions.append(condition)
        if not conditions:
            self.notes.append("no success evidence survived; lint will flag the checkpoint")
            final = self.t.success.page.route or "/"
            conditions.append(sig.UrlCondition(path_regex=re.escape(final) + "$"))
        signature = conditions[0] if len(conditions) == 1 else sig.AllCondition(all=conditions)
        landmark = _landmark(self.steps[0])
        negated = sig.NotCondition(not_=signature)
        control = sig.AllCondition(all=[landmark, negated]) if landmark else negated
        return Checkpoint(
            signature=signature,
            negative_control=control,
            description=self.t.success.summary or "The flow's confirmation is showing",
        )

    # --- 9: the artifact -----------------------------------------------------------------------

    def _capability(self, success: Checkpoint) -> Capability:
        commit = any(s.effect == "commit" for s in self.steps)
        targets = [s.target for s in self.steps if s.target is not None]
        locators = [loc for t in targets for loc in t.locators]
        first = self.steps[0]
        return Capability(
            id=self.goal.capability,
            version=self.goal.version,
            title=self.goal.title,
            description=" ".join(self.goal.description.split()),
            lifecycle=Lifecycle(status="draft"),
            app=AppBinding(
                vendor_app=self.goal.vendor_app,
                entry_route=_route_of_template(first.url or "") or self.goal.start_route,
            ),
            requires=Requirements(session=self.goal.session),
            inputs=[p.input_spec() for p in self.goal.params],
            outputs=self.outputs,
            side_effects=SideEffects(
                cls="commit" if commit else "read",
                idempotent=not commit,
                approval="per_invocation" if commit else "none",
                description=" ".join((self.goal.side_effects or _default_effects(commit)).split()),
            ),
            steps=self.steps,
            success=success,
            outcomes=self.outcomes,
            recovery=RecoveryPolicy(
                allowed=["retry_transient", "reauthenticate", "dismiss_interstitial"], budget=3
            ),
            provenance=Provenance(
                discovery_run_id=self.t.run_id,
                recorded_against=self.t.tenant,
                goal_template=self.goal.goal,
                models_served=self.models_served,
                human_steps=self.human_steps,
                locator_stats=LocatorStats(
                    targets=len(targets),
                    locators=len(locators),
                    verified_unique=sum(
                        1 for loc in locators if loc.verified and loc.verified.unique
                    ),
                ),
                authored="discovered",
            ),
        )

    # --- helpers ---------------------------------------------------------------------------------

    def _id(self, verb: str, what: str) -> str:
        return f"s{len(self.steps) + 1}_{verb}_{what}"[:40].rstrip("_")

    def _url(self, url: str, seq: int) -> str:
        """`{base_url}` + route + query, with param values templated."""
        url = JSESSIONID.sub("", url)
        route = self.gate.route_of(url)
        if route is None or self.gate.tenant.host != (urlsplit(url).hostname or "").lower():
            self.problems.append(f"call {seq}: the address is outside the tenant's app")
            return "{base_url}/"
        params = {v: k for k, v in self.values.items()}
        query = []
        for key, value in parse_qsl(urlsplit(url).query, keep_blank_values=True):
            if value in params:
                query.append(f"{quote(key, safe='')}={{{params[value]}}}")
            else:
                if _DATA.search(value):
                    self.notes.append(
                        f"call {seq}: query value for {key!r} looks like data and was kept"
                    )
                query.append(f"{quote(key, safe='')}={quote(value, safe='')}")
        return "{base_url}" + route + (("?" + "&".join(query)) if query else "")


# --- page items and conditions -----------------------------------------------------------------


@dataclass(frozen=True)
class _Item:
    role: str | None
    name: str | None
    level: int | None
    text: str | None


def _items(snapshot: str) -> list[_Item]:
    items = []
    for line in snapshot.splitlines():
        role, name = parse_line(line)
        level = _LEVEL.search(line)
        text_match = _TEXT_LINE.match(line)
        text = text_match.group(1).strip().strip('"') if text_match else None
        if role or text:
            items.append(_Item(role, name, int(level.group(1)) if level else None, text))
    return items


def _arrival(
    after: PageState, before: PageState | None, values: Mapping[str, str]
) -> list[Expectation]:
    """Arrived on a page: its route, and its main heading (a new one when there was a
    page before)."""
    conditions: list[sig.Signature] = []
    if after.route is not None:
        conditions.append(sig.UrlCondition(path_regex=re.escape(after.route) + "$"))
    heading = _heading(after, before, values) or _heading(after, None, values)
    if heading is not None:
        conditions.append(heading)
    if not conditions:
        return []
    when = conditions[0] if len(conditions) == 1 else sig.AllCondition(all=conditions)
    loaded = f"{after.route or 'The page'} has loaded"
    return [
        Expectation(
            description=loaded + (f" and {_says(heading)}" if heading is not None else ""),
            when=when,
            then=Disposition(do="continue"),
        )
    ]


def _changes(
    before: PageState | None, after: PageState | None, values: Mapping[str, str]
) -> list[Expectation]:
    """What showed up after an action: a new page, else a new heading, else a new
    line of text, else a new notice or control. Data never counts unless templated."""
    if before is None or after is None:
        return []
    if after.route != before.route:
        return _arrival(after, before, values)
    condition: sig.Signature | None = _heading(after, before, values)
    if condition is None:
        old = {i.text for i in _items(before.snapshot)}
        for item in _items(after.snapshot):
            text = _stable(item.text, values) if item.text else None
            if text and item.text not in old and len(text) <= _TEXT_MAX:
                condition = sig.VisibleCondition(text=text)
                break
    if condition is None:
        old_named = {(i.role, i.name) for i in _items(before.snapshot)}
        for item in _items(after.snapshot):
            name = _stable(item.name, values) if item.name else None
            if item.role in _NOTICE_ROLES and name and (item.role, item.name) not in old_named:
                condition = sig.VisibleCondition(role=item.role, name=name)
                break
    if condition is None:
        return []
    return [
        Expectation(
            description=_first_upper(_says(condition)),
            when=condition,
            then=Disposition(do="continue"),
        )
    ]


def _first_upper(text: str) -> str:
    return text[:1].upper() + text[1:]


def _says(condition: sig.Signature) -> str:
    if isinstance(condition, sig.VisibleCondition):
        if condition.role and condition.name:
            return f"the {condition.role} {condition.name!r} is showing"
        if condition.text:
            return f"the text {condition.text!r} is showing"
    return "the page shows the recorded change"


def _heading(
    after: PageState, before: PageState | None, values: Mapping[str, str]
) -> sig.Signature | None:
    old = {i.name for i in _items(before.snapshot)} if before is not None else set()
    headings = [
        i for i in _items(after.snapshot) if i.role == "heading" and i.name and i.name not in old
    ]
    headings.sort(key=lambda i: i.level if i.level is not None else 7)
    for item in headings:
        name = _stable(item.name, values)
        if name:
            return sig.VisibleCondition(role="heading", name=name)
    return None


def _stable(text: str | None, values: Mapping[str, str]) -> str | None:
    """The text with param values templated, or None when it still looks like data."""
    if not text:
        return None
    out = text
    for name, value in sorted(values.items(), key=lambda kv: -len(kv[1])):
        if len(value) >= 3:
            out = out.replace(value, "{" + name + "}")
    return None if _DATA.search(out) else out


def _evidence(record: ElementRecord, values: Mapping[str, str]) -> sig.Signature | None:
    name = _stable(record.name, values) if record.name else None
    if record.role and name:
        return sig.VisibleCondition(role=record.role, name=name)
    css = _page_css(record.target) if record.target else None
    if css is not None:
        return sig.VisibleCondition(css=css)
    return None


def _holds_in(condition: sig.VisibleCondition, items: list[_Item]) -> bool:
    """Whether a role+name or text condition is visible in a recorded snapshot. CSS
    conditions cannot be judged from a snapshot; replay checks them live."""
    if condition.role and condition.name:
        return any(i.role == condition.role and i.name == condition.name for i in items)
    if condition.text:
        return any(i.text == condition.text or i.name == condition.text for i in items)
    return False


def _landmark(step: Step) -> sig.Signature | None:
    for expectation in step.expect:
        for node in sig.walk(expectation.when):
            if isinstance(node, sig.VisibleCondition):
                return node
    return None


def _page_css(target: Target | None) -> str | None:
    """A CSS locator that is unique on the page itself: signatures are evaluated
    against the page, not inside a target's scope or frame."""
    if target is None or target.scope is not None or target.frame_path:
        return None
    for locator in target.locators:
        if locator.by == "css" and isinstance(locator.value, str) and not locator.templated:
            return locator.value
    return None


def _extraction(name: str, sample: str, kind: str, element: ElementRecord) -> Extraction:
    read = "value" if element.is_form_control else "text"
    if re.fullmatch(r"\d+", sample):
        return Extraction(output=name, read=read, parse="regex", regex=r"^\s*(\d+)\s*$")
    if kind == "integer":
        return Extraction(output=name, read=read, parse="integer")
    if kind == "decimal":
        return Extraction(output=name, read=read, parse="decimal")
    return Extraction(output=name, read=read, parse="regex", regex=r"^\s*(.*?)\s*$")


# --- small pure helpers ------------------------------------------------------------------------


def _same_element(a: RecordedCall, b: RecordedCall) -> bool:
    return a.element is not None and b.element is not None and a.element.key == b.element.key


def _next_seq(actions: list[RecordedCall], call: RecordedCall) -> int:
    later = [c.seq for c in actions if c.seq > call.seq]
    return min(later) if later else 1 << 30


def _required_state(state: PageState | None, seq: int) -> PageState:
    if state is None:
        raise CompileError([f"call {seq} has no recorded page"])
    return state


def _route_of_template(url: str) -> str | None:
    if not url.startswith("{base_url}"):
        return None
    return url.removeprefix("{base_url}").split("?", 1)[0] or "/"


def _route_slug(route: str) -> str:
    stem = route.rstrip("/").rsplit("/", 1)[-1].split(".", 1)[0]
    return _slug(stem or "home")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:24] or "element"


def _value_slug(value: ValueRef, element: ElementRecord) -> str:
    if isinstance(value, ParamValue):
        return value.param
    return element.name or element.tag


def _default_effects(commit: bool) -> str:
    return "Changes state in the app." if commit else "Reads only; changes nothing."
