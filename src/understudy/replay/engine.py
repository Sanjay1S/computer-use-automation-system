"""The replay engine: runs a capability with no model in the loop.

Each step: wait until its target is ready (watching the app's global states while
waiting), act once, then race the step's expectations against the global states
until one holds or the step times out. Nothing sleeps for a fixed time and nothing
treats network idle as success. Recoverable conditions go to a small fixed
registry with a per-run budget.

The commit rule: a commit step is never retried or re-entered, and once its input
may have reached the page, anything short of a definite success or a definite
business answer is `needs_human` with `commit_state=unknown`.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, NoReturn

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator as PwLocator
from playwright.sync_api import Page

from understudy.artifact import hashing
from understudy.artifact import signatures as sig
from understudy.artifact.schema import (
    Capability,
    Disposition,
    Expectation,
    Extraction,
    LiteralValue,
    OutputSpec,
    ParamValue,
    Sensitivity,
    Step,
)
from understudy.chaos import Chaos
from understudy.evidence.run import Actor, EventKind, RunContext
from understudy.policy.gate import PolicyGate
from understudy.profile import AppProfile, GlobalState, Interstitial
from understudy.redaction.redactor import Redactor
from understudy.replay import preflight as pre
from understudy.replay.race import first_hit
from understudy.replay.recovery import RecoveryLedger, backoff_ms, reentry_index
from understudy.replay.result import (
    Attempt,
    BusinessOutcome,
    CommitState,
    DriftEvent,
    Escalation,
    Evidence,
    Failure,
    FailureCategory,
    RecoveryEvent,
    Rejection,
    ReplayResult,
    SideEffectReport,
    Status,
    StepTiming,
    Timings,
)
from understudy.session import Credentials, FormLogin, LoginFailed
from understudy.surface.base import (
    ActAction,
    ActionBlocked,
    ActionTimeout,
    DialogBlocking,
    InfraError,
    OptionAbsent,
    PolicyBlocked,
    Resolution,
    SurfaceError,
    TargetChanged,
)
from understudy.surface.web import WebSurface

Verdict = Literal["continue", "succeed"]

_ACT: dict[str, ActAction] = {
    "click": "click",
    "fill": "fill",
    "select": "select",
    "check": "check",
    "press": "press",
}


# --- how a run stops ---------------------------------------------------------------


class _Stop(Exception):
    """Ends the run. Raised anywhere below `_Execution.run`; turned into the result."""


@dataclass(eq=False)
class _Fail(_Stop):
    category: FailureCategory
    code: str
    step_id: str | None
    expected: str
    observed: str
    attempts: list[Attempt] = field(default_factory=list)


@dataclass(eq=False)
class _Escalate(_Stop):
    reason: str
    step_id: str | None
    detail: str
    cause: _Fail | None = None


@dataclass(eq=False)
class _Business(_Stop):
    code: str
    step_id: str


class _Retry(Exception):
    """Run the current step again (retry_transient)."""


@dataclass(eq=False)
class _Restart(Exception):
    index: int


# --- the public entry point -----------------------------------------------------------


class Replayer:
    def __init__(
        self,
        capability: Capability,
        *,
        gate: PolicyGate,
        profile: AppProfile,
        credentials: Credentials,
        runs_dir: Path,
        redactor: Redactor | None = None,
        poll_ms: int = 100,
        screenshots: bool = True,
    ) -> None:
        self.capability = capability
        self.gate = gate
        self.tenant = gate.tenant
        self.profile = profile
        self.credentials = credentials
        self.runs_dir = runs_dir
        self.redactor = redactor or Redactor.from_env()
        self.poll_ms = poll_ms
        self.screenshots = screenshots

    def preflight(
        self,
        inputs: Mapping[str, str],
        *,
        approval: pre.CommitApproval | None = None,
        allow_draft: bool = False,
        fault_injection: bool = False,
    ) -> Rejection | None:
        return pre.preflight(
            self.capability,
            inputs,
            gate=self.gate,
            profile=self.profile,
            approval=approval,
            allow_draft=allow_draft,
            fault_injection=fault_injection,
        )

    def run(
        self,
        inputs: Mapping[str, str],
        *,
        open_page: Callable[[], Page],
        approval: pre.CommitApproval | None = None,
        allow_draft: bool = False,
        chaos: Chaos | None = None,
    ) -> ReplayResult:
        """`open_page` is called only once preflight has passed."""
        return _Execution(self, dict(inputs), chaos).run(open_page, approval, allow_draft)


# --- one run ------------------------------------------------------------------------


class _Execution:
    def __init__(self, replayer: Replayer, inputs: dict[str, str], chaos: Chaos | None) -> None:
        self.r = replayer
        self.cap = replayer.capability
        self.params = inputs
        self.chaos = chaos
        self.started = datetime.now(UTC)
        self.artifact_hash = hashing.content_hash(self.cap)
        self.sensitivity: dict[str, Sensitivity] = {i.name: i.sensitivity for i in self.cap.inputs}
        known: dict[str, Sensitivity] = {
            value: self.sensitivity[name]
            for name, value in inputs.items()
            if name in self.sensitivity
        }
        known.update(replayer.credentials.sensitivities)
        redactor = replayer.redactor.knowing(known)
        self.run_ctx = RunContext(
            replayer.runs_dir,
            kind="replay",
            redactor=redactor,
            meta={
                "capability": self.cap.id,
                "version": self.cap.version,
                "artifact_hash": self.artifact_hash,
                "tenant": replayer.tenant.id,
                "inputs": {
                    name: redactor.value(value, self.sensitivity.get(name, "pii"))
                    for name, value in inputs.items()
                },
                "faults": [str(f) for f in chaos.faults] if chaos else [],
            },
            verbatim=("capability", "version", "artifact_hash", "tenant"),
        )
        self.session = FormLogin(replayer.profile.login, replayer.tenant, replayer.credentials)
        self.surface: WebSurface | None = None
        self.ledger = RecoveryLedger(self.cap.recovery)
        self.attempts: dict[str, int] = {}
        self.outputs: dict[str, Any] = {}
        self.drift: list[DriftEvent] = []
        self.timings: list[StepTiming] = []
        self.commit_state: CommitState | Literal["pending"] = "not_attempted"
        self.commit_step: str | None = None
        self.index = 0
        self.current: Step | None = None
        """The step being run; None while logging in and while verifying."""

    @property
    def page_surface(self) -> WebSurface:
        assert self.surface is not None
        return self.surface

    # --- top level ---------------------------------------------------------------------

    def run(
        self,
        open_page: Callable[[], Page],
        approval: pre.CommitApproval | None,
        allow_draft: bool,
    ) -> ReplayResult:
        rejection = self.r.preflight(
            self.params,
            approval=approval,
            allow_draft=allow_draft,
            fault_injection=bool(self.chaos and self.chaos.faults),
        )
        if rejection is not None:
            self._event("policy", ok=False, error="; ".join(rejection.problems))
            return self._finish("rejected", rejection=rejection)
        try:
            page = open_page()
            if self.chaos:
                self.chaos.install(page)
            self.surface = WebSurface(
                page,
                self.r.gate,
                mode="replay",
                run_id=self.run_ctx.run_id,
                # Without dismiss_interstitial a known dialog is just an unknown one.
                dialog_rules=(
                    self.r.profile.dialog_rules()
                    if "dismiss_interstitial" in self.cap.recovery.allowed
                    else []
                ),
            )
            self._login()
            self._steps()
            self._verify()
        except _Stop as stop:
            return self._stopped(stop)
        except SurfaceError as error:
            return self._stopped(self._from_surface(error, self._step_id()))
        except PlaywrightError as error:
            return self._stopped(
                _Fail("infra", "infra_error", self._step_id(), "a working browser", _line(error))
            )
        except Exception as error:  # the result must be written whatever broke
            return self._stopped(
                _Fail(
                    "automation",
                    "engine_error",
                    self._step_id(),
                    "the engine to run to a result",
                    f"{type(error).__name__}: {_line(error)}",
                )
            )
        if self.commit_state == "pending":
            self.commit_state = "committed"
        return self._finish("succeeded", outputs=dict(self.outputs))

    def _steps(self) -> None:
        steps = self.cap.steps
        self.index = 0
        while self.index < len(steps):
            step = steps[self.index]
            self.current = step
            try:
                verdict = self._step(step)
            except _Retry:
                continue
            except _Restart as restart:
                self.index = restart.index
                continue
            if verdict == "succeed":
                break
            self.index += 1
        self.current = None  # kept on the way out of a failing step, for its result

    def _step(self, step: Step) -> Verdict:
        surface = self.page_surface
        started = time.monotonic()
        attempt = self.attempts.get(step.id, 0)
        if self.chaos:
            fired = self.chaos.before_step(step, surface.page, self.params)
            if fired:
                self._event("control", actor="system", step_id=step.id, injected=fired)
        try:
            if step.action == "navigate":
                self._navigate(step)
                verdict = self._race(step)
            else:
                resolution = self._await_target(step)
                if step.action == "extract":
                    self._extract(step, resolution)
                    verdict = self._race(step)
                elif step.action == "wait_for":
                    verdict = self._race(step)
                elif step.effect == "commit":
                    with surface.commit_window():
                        self._check_not_vacuous(step)
                        self.commit_step = step.id
                        self._act(step, resolution)
                        verdict = self._race(step)
                else:
                    self._act(step, resolution)
                    verdict = self._race(step)
        finally:
            elapsed = int((time.monotonic() - started) * 1000)
            self.timings.append(StepTiming(step_id=step.id, attempt=attempt, duration_ms=elapsed))
        self._screenshot(step.id)
        return verdict

    # --- the phases of a step ------------------------------------------------------------

    def _navigate(self, step: Step) -> None:
        url = pre.expand_url(step.url or "", self.r.tenant.base_url, self.params)
        started = time.monotonic()
        try:
            status = self.page_surface.navigate(url, timeout_ms=step.timeout_ms)
        except ActionTimeout as error:
            self._event_act(step, started, ok=False, error=str(error))
            self._transient(step, f"the page did not load: {error}", "infra", "timeout")
        self._event_act(step, started, ok=True, status=status)

    def _await_target(self, step: Step) -> Resolution[PwLocator]:
        """Poll until the step's readiness signature holds and its target resolves,
        watching the global states throughout. Resolution never waits by itself."""
        assert step.target is not None
        surface = self.page_surface
        deadline = time.monotonic() + step.timeout_ms / 1000
        ready_seen = False
        last: Resolution[PwLocator] | None = None
        while True:
            self._look(step, [])
            ready = step.wait_before is None or surface.probe(step.wait_before, self.params)
            if ready:
                ready_seen = True
                last = surface.resolve(step.target, self.params)
                if last.element is not None:
                    if last.drifted:
                        self._drift(step, last)
                    return last
                if self._absent_is_outcome(step, last) and step.on_target_absent:
                    raise _Business(step.on_target_absent.outcome, step.id)
            if time.monotonic() >= deadline:
                break
            surface.pause(self.r.poll_ms)
        if not ready_seen:
            raise _Fail(
                "automation",
                "not_ready",
                step.id,
                f"the step's readiness check ({step.wait_before.kind if step.wait_before else ''})",
                f"it never held within {step.timeout_ms} ms",
            )
        attempts = _attempts(last)
        raise _Fail(
            "automation",
            "target_not_found",
            step.id,
            step.target.description,
            "no locator found exactly one visible element with the expected identity: "
            + "; ".join(f"{a.by}#{a.index} {a.outcome}" for a in attempts),
            attempts,
        )

    def _act(self, step: Step, resolution: Resolution[PwLocator]) -> None:
        assert step.target is not None
        action = _ACT[step.action]
        value, sensitivity = self._value(step)
        commit = step.effect == "commit"
        for attempt in (0, 1):
            started = time.monotonic()
            assert resolution.element is not None
            try:
                self.page_surface.act(
                    action,
                    resolution.element,
                    value,
                    fingerprint=step.target.fingerprint,
                    params=self.params,
                    timeout_ms=step.timeout_ms,
                )
            except SurfaceError as error:
                if commit and error.dispatched:
                    self.commit_state = "pending"
                self._event_act(step, started, resolution, value, sensitivity, False, str(error))
                if isinstance(error, TargetChanged) and not error.dispatched and attempt == 0:
                    resolution = self._await_target(step)  # re-rendered: find it again, once
                    continue
                if isinstance(error, OptionAbsent):
                    if step.on_target_absent and error.options_loaded >= 1:
                        raise _Business(step.on_target_absent.outcome, step.id) from None
                    raise _Fail(
                        "automation", "option_absent", step.id, "the option to exist", str(error)
                    ) from None
                if isinstance(error, ActionTimeout) and not commit:
                    self._transient(step, str(error), "automation", "timeout")
                raise
            if commit:
                self.commit_state = "pending"
            self._event_act(step, started, resolution, value, sensitivity, True)
            return

    def _extract(self, step: Step, resolution: Resolution[PwLocator]) -> None:
        assert step.extract is not None and resolution.element is not None
        spec = next(o for o in self.cap.outputs if o.name == step.extract.output)
        started = time.monotonic()
        raw = self.page_surface.read(resolution.element, step.extract)
        value = _typed(spec, _parse(raw, step.extract))
        if value is None:
            self._event_act(step, started, resolution, ok=False, error="unreadable value")
            self._transient(
                step,
                f"the text read for {spec.name!r} does not parse as a valid {spec.type}",
                "automation",
                "extract_failed",
            )
        self.outputs[spec.name] = value
        self.run_ctx.know({str(value): spec.sensitivity})
        self._event_act(step, started, resolution, str(value), spec.sensitivity, True)

    def _race(self, step: Step) -> Verdict:
        surface = self.page_surface
        started = time.monotonic()
        deadline = started + step.timeout_ms / 1000
        while True:
            expectation = self._look(step, step.expect)
            if expectation is not None:
                self._event_race(step, started, expectation.description, expectation.then)
                return self._apply(step, expectation.then, expectation.description)
            if not step.expect and step.effect != "commit":
                return "continue"  # nothing to wait for; the global states were checked once
            if time.monotonic() >= deadline:
                disposition = step.resolved_on_no_match
                why = f"none of the step's expectations held within {step.timeout_ms} ms"
                self._event_race(step, started, None, disposition)
                return self._apply(step, disposition, why)
            surface.pause(self.r.poll_ms)

    def _verify(self) -> None:
        surface = self.page_surface
        success = self.cap.success
        if not surface.probe(success.signature, self.params):
            raise _Fail(
                "automation",
                "success_not_verified",
                self._step_id(),
                success.description,
                "the success checkpoint does not hold after the last step",
            )
        missing = [o.name for o in self.cap.outputs if o.name not in self.outputs]
        if missing:
            raise _Fail(
                "automation",
                "output_missing",
                self._step_id(),
                f"outputs {', '.join(missing)}",
                "the flow finished without extracting them",
            )

    # --- states and recovery ---------------------------------------------------------------

    def _look(self, step: Step, expectations: list[Expectation]) -> Expectation | None:
        """One look at the page by race priority. Global states and dialogs end here
        (a recovery, a failure or an escalation); a dismissed interstitial means
        looking again; otherwise the first expectation that holds, if any."""
        profile = self.r.profile
        while True:
            hit = first_hit(
                self.page_surface,
                global_states=profile.global_states,
                interstitials=profile.interstitials,
                expectations=expectations,
                params=self.params,
            )
            if hit is None:
                return None
            if hit.source == "global":
                state = profile.global_states[hit.index]
                self._acknowledge(state.when)
                self._on_global(step, state)
            elif hit.source == "interstitial":
                self._dismiss(step, profile.interstitials[hit.index])
            elif hit.source == "expect":
                self._acknowledge(expectations[hit.index].when)
                return expectations[hit.index]
            else:
                raise _Escalate("unknown_blocking_state", step.id, hit.description)

    def _acknowledge(self, signature: sig.Signature) -> None:
        """A dialog that a matched signature was waiting for has been dealt with: it must
        stop matching, and stop blocking the next step."""
        regexes = _dialog_regexes(signature)
        if regexes:
            self.page_surface.acknowledge_dialogs(matching=regexes)

    def _on_global(self, step: Step, state: GlobalState) -> NoReturn:
        if state.handling == "reauthenticate":
            self._reauthenticate(step, state)
        self._transient(step, state.description.strip(), "app", "app_error", page_state=True)

    def _transient(
        self,
        step: Step,
        reason: str,
        category: FailureCategory,
        code: str,
        *,
        page_state: bool = False,
    ) -> NoReturn:
        """Retry if that is safe and allowed; otherwise stop.

        A fault of the step itself (a timeout, an unreadable value) retries the step.
        A fault the page shows (a 5xx, an error panel) would still be on screen after
        a retry in place, so it retries from the re-entry point, which reloads the
        page; that step's retry policy is the one spent."""
        if self.commit_state == "pending":
            raise _Escalate(
                "app_error_after_commit" if category == "app" else "commit_outcome_unknown",
                step.id,
                reason,
            )
        index: int | None = self.index
        refusal: str | None = None
        if page_state and step.action != "navigate":
            if self.commit_state != "not_attempted":
                refusal = "the commit has already happened; replay never re-enters past it"
            else:
                index = reentry_index(self.cap.steps, self.index)
                if index is None:
                    refusal = "no step at or before this one is a re-entry point"
        target = self.cap.steps[index] if index is not None else step
        attempt = self.attempts.get(target.id, 0)
        if refusal is None:
            if index == self.index and step.effect == "commit":
                refusal = "a commit step is never retried"
            elif attempt >= target.retry.max_attempts:
                refusal = (
                    f"step {target.id!r} allows {target.retry.max_attempts} retries "
                    f"and has used {attempt}"
                )
            else:
                refusal = self.ledger.refusal("retry_transient")
        if refusal is not None or index is None:
            observed = f"{reason} (not retried: {refusal})"
            raise _Fail(category, code, step.id, _expected(step), observed)
        restarted = target.id if index != self.index else None
        self._recover(
            RecoveryEvent(
                handler="retry_transient", step_id=step.id, reason=reason, restarted_from=restarted
            )
        )
        self.attempts[target.id] = attempt + 1
        self.page_surface.pause(backoff_ms(target.retry, attempt))
        if restarted is None:
            raise _Retry()
        raise _Restart(index)

    def _reauthenticate(self, step: Step, state: GlobalState) -> NoReturn:
        reason = state.description.strip()
        if self.commit_state == "pending":
            raise _Escalate("commit_outcome_unknown", step.id, f"after the commit: {reason}")
        if self.commit_state != "not_attempted":
            raise _Fail(
                "app",
                "session_expired",
                step.id,
                "a live session",
                f"{reason}; the commit has already happened, and replay never re-enters a "
                "flow past its commit",
            )
        entry = reentry_index(self.cap.steps, self.index)
        refusal = self.ledger.refusal("reauthenticate") or (
            "no step at or before this one is a re-entry point" if entry is None else None
        )
        if refusal or entry is None:
            raise _Fail(
                "app", "session_expired", step.id, "a live session", f"{reason} ({refusal})"
            )
        restart = self.cap.steps[entry].id
        self._recover(
            RecoveryEvent(
                handler="reauthenticate", step_id=step.id, reason=reason, restarted_from=restart
            )
        )
        self._login()
        raise _Restart(entry)

    def _dismiss(self, step: Step, interstitial: Interstitial) -> None:
        surface = self.page_surface
        event = RecoveryEvent(
            handler="dismiss_interstitial", step_id=step.id, reason=interstitial.description
        )
        if interstitial.dialog is not None:
            # Already answered by the surface on arrival, by the profile's rule, which it
            # only has when the artifact allows this handler. Reported, not budgeted.
            answered = surface.acknowledge_dialogs(matching=_dialog_regexes(interstitial.when))
            if any(not d.known for d in answered):
                raise _Escalate(
                    "unknown_blocking_state",
                    step.id,
                    f"{interstitial.description}: this artifact does not allow "
                    "dismiss_interstitial, so it got the default answer (dismissed)",
                )
            self._recover(event, budgeted=False)
            return
        refusal = self.ledger.refusal("dismiss_interstitial")
        if refusal:
            raise _Escalate(
                "unknown_blocking_state", step.id, f"{interstitial.description} ({refusal})"
            )
        assert interstitial.dismiss is not None
        resolution = surface.resolve(interstitial.dismiss, self.params)
        if resolution.element is None or resolution.facts is None:
            raise _Escalate(
                "unknown_blocking_state",
                step.id,
                f"{interstitial.description}, and its dismiss control was not found",
            )
        if self.r.gate.classify("left_click", resolution.facts) == "commit":
            raise _Escalate(
                "unknown_blocking_state",
                step.id,
                f"{interstitial.description}, and its dismiss control would commit",
            )
        surface.act(
            "click",
            resolution.element,
            fingerprint=interstitial.dismiss.fingerprint,
            params=self.params,
        )
        self._recover(event)
        # Wait for it to go before looking again: one that fades out would otherwise
        # be dismissed again, spending budget on a recovery that already worked.
        deadline = time.monotonic() + step.timeout_ms / 1000
        while surface.probe(interstitial.when, self.params):
            if time.monotonic() >= deadline:
                raise _Escalate(
                    "unknown_blocking_state",
                    step.id,
                    f"{interstitial.description}, and it did not close when dismissed",
                )
            surface.pause(self.r.poll_ms)

    def _recover(self, event: RecoveryEvent, *, budgeted: bool = True) -> None:
        self.ledger.spend(event, budgeted=budgeted)
        self._event(
            "recovery",
            step_id=event.step_id,
            handler=event.handler,
            reason=event.reason,
            restarted_from=event.restarted_from,
        )

    def _apply(self, step: Step, disposition: Disposition, why: str) -> Verdict:
        if disposition.do == "continue" or disposition.do == "succeed":
            if self.commit_state == "pending":
                self.commit_state = "committed"
            return disposition.do
        code = disposition.code or ""
        if disposition.do == "business":
            raise _Business(code, step.id)
        if disposition.do == "escalate":
            raise _Escalate(code, step.id, why)
        raise _Fail("automation", code, step.id, _expected(step), why)

    def _check_not_vacuous(self, step: Step) -> None:
        """The success checkpoint must be false before the commit, or seeing it
        afterwards would not show that this run committed anything."""
        if self.page_surface.probe(self.cap.success.signature, self.params):
            raise _Fail(
                "automation",
                "checkpoint_vacuous",
                step.id,
                "the success checkpoint is false before the commit",
                "it already holds, so it could not tell whether this run committed",
            )

    @staticmethod
    def _absent_is_outcome(step: Step, resolution: Resolution[PwLocator]) -> bool:
        """A param-bound target that is missing once the step's structural guard holds
        (the list has loaded) is the app's answer, not a broken locator."""
        assert step.target is not None
        return (
            step.on_target_absent is not None
            and step.wait_before is not None
            and any(locator.templated for locator in step.target.locators)
            and bool(resolution.attempts)
            and all(a.outcome == "no_match" for a in resolution.attempts)
        )

    def _login(self) -> None:
        started = time.monotonic()
        with self.chaos.paused() if self.chaos else contextlib.nullcontext():
            self.session.login(self.page_surface, poll_ms=self.r.poll_ms)
        self._event(
            "act",
            actor="system",
            member="login",
            duration_ms=int((time.monotonic() - started) * 1000),
            ok=True,
        )

    # --- ending a run ---------------------------------------------------------------------

    def _stopped(self, stop: _Stop) -> ReplayResult:
        """Applies the commit rule, then builds the result."""
        if self.commit_state == "pending":
            if isinstance(stop, _Fail):
                stop = _Escalate(
                    "commit_outcome_unknown",
                    stop.step_id,
                    f"{stop.code}: {stop.observed}",
                    cause=stop,
                )
            self.commit_state = "refused" if isinstance(stop, _Business) else "unknown"
        if isinstance(stop, _Business):
            spec = next((o for o in self.cap.outcomes if o.code == stop.code), None)
            outcome = BusinessOutcome(
                code=stop.code,
                message=spec.description if spec else stop.code,
                step_id=stop.step_id,
            )
            return self._finish("business_outcome", outcome=outcome)
        assert isinstance(stop, _Fail | _Escalate)
        evidence = self._bundle(stop)
        if isinstance(stop, _Escalate):
            escalation = Escalation(
                reason=stop.reason, step_id=stop.step_id, detail=stop.detail, evidence=evidence
            )
            failure = _failure(stop.cause, evidence) if stop.cause else None
            self._event("escalation", step_id=stop.step_id, reason=stop.reason, ok=False)
            return self._finish("needs_human", escalation=escalation, failure=failure)
        return self._finish("failed", failure=_failure(stop, evidence))

    def _from_surface(self, error: SurfaceError, step_id: str | None) -> _Stop:
        message = str(error)
        if isinstance(error, PolicyBlocked):
            return _Fail(
                "policy",
                "policy_blocked",
                step_id,
                "every action and request within policy",
                f"{error.rule}: {message}",
            )
        if isinstance(error, ActionBlocked | DialogBlocking):
            return _Escalate("unknown_blocking_state", step_id, message)
        if isinstance(error, InfraError | LoginFailed):
            return _Fail("infra", error.code, step_id, "a working session and browser", message)
        return _Fail("automation", error.code, step_id, "the step to complete", message)

    def _finish(
        self,
        status: Status,
        *,
        outputs: dict[str, Any] | None = None,
        outcome: BusinessOutcome | None = None,
        rejection: Rejection | None = None,
        escalation: Escalation | None = None,
        failure: Failure | None = None,
    ) -> ReplayResult:
        if self.chaos:
            self.chaos.release()
        finished = datetime.now(UTC)
        commit = self.commit_state if self.commit_state != "pending" else "unknown"
        result = ReplayResult(
            status=status,
            capability=self.cap.id,
            version=self.cap.version,
            artifact_hash=self.artifact_hash,
            tenant=self.r.tenant.id,
            run_id=self.run_ctx.run_id,
            outputs=outputs,
            outcome=outcome,
            rejection=rejection,
            escalation=escalation,
            failure=failure,
            side_effects=SideEffectReport(
                commit_state=commit,
                commit_step=self.commit_step,
                commit_requests=self.surface.commit_requests if self.surface else 0,
            ),
            recoveries=list(self.ledger.events),
            drift=list(self.drift),
            timings=Timings(
                started_at=self.started,
                finished_at=finished,
                duration_ms=int((finished - self.started).total_seconds() * 1000),
                steps=list(self.timings),
            ),
        )
        self.run_ctx.write_json("result.json", result, verbatim=ReplayResult.IDENTIFIERS)
        self._event(
            "result",
            ok=status == "succeeded",
            status=status,
            commit_state=commit,
            code=(
                (failure.code if failure else None)
                or (escalation.reason if escalation else None)
                or (outcome.code if outcome else None)
                or (rejection.code if rejection else None)
            ),
        )
        return result

    def _bundle(self, stop: _Fail | _Escalate) -> Evidence | None:
        if self.surface is None:
            return None
        reason = stop.code if isinstance(stop, _Fail) else stop.reason
        directory = self.run_ctx.subdir("failure")
        try:
            self.surface.failure_bundle(
                directory,
                self.run_ctx.redactor,
                reason=reason,
                events=self.run_ctx.recent(10),
                extra_masks=self.r.profile.screenshot_masks,
            )
        except Exception as error:  # a dead page still gets a result, without the bundle
            (directory / "bundle.error.txt").write_text(
                self.run_ctx.redactor.text(_line(error)) + "\n", encoding="utf-8"
            )
            return None
        files = {p.name for p in directory.iterdir()}

        def rel(name: str) -> str | None:
            return f"failure/{name}" if name in files else None

        return Evidence(
            screenshot=rel("screenshot.png"),
            dom=rel("dom.html"),
            aria=rel("aria.txt"),
            console=rel("console.txt"),
            dialog=rel("dialog.txt"),
        )

    # --- helpers --------------------------------------------------------------------------

    def _value(self, step: Step) -> tuple[str | None, Sensitivity]:
        if isinstance(step.value, ParamValue):
            return self.params[step.value.param], self.sensitivity[step.value.param]
        if isinstance(step.value, LiteralValue):
            return step.value.literal, "public"
        return None, "public"  # secret refs are refused in preflight

    def _step_id(self) -> str | None:
        return self.current.id if self.current else None

    def _drift(self, step: Step, resolution: Resolution[PwLocator]) -> None:
        assert resolution.locator_index is not None and step.target is not None
        by = step.target.locators[resolution.locator_index].by
        event = DriftEvent(
            step_id=step.id,
            locator_index=resolution.locator_index,
            by=by,
            attempts=_attempts(resolution),
        )
        seen = {(d.step_id, d.locator_index) for d in self.drift}
        if (step.id, event.locator_index) not in seen:
            self.drift.append(event)
            self._event("drift", step_id=step.id, locator_used=f"{by}#{event.locator_index}")

    def _screenshot(self, label: str) -> None:
        if not self.r.screenshots or self.surface is None:
            return
        with contextlib.suppress(PlaywrightError, SurfaceError):  # evidence never fails a run
            self.surface.screenshot(
                self.run_ctx.screenshot_path(label), extra_masks=self.r.profile.screenshot_masks
            )

    def _event(self, kind: EventKind, *, actor: Actor = "engine", **fields: Any) -> None:
        self.run_ctx.event(kind, actor=actor, **fields)

    def _event_act(
        self,
        step: Step,
        started: float,
        resolution: Resolution[PwLocator] | None = None,
        value: str | None = None,
        sensitivity: Sensitivity = "pii",
        ok: bool = True,
        error: str | None = None,
        **data: Any,
    ) -> None:
        locator_used = None
        if resolution is not None and resolution.locator_index is not None and step.target:
            index = resolution.locator_index
            locator_used = f"{step.target.locators[index].by}#{index}"
        self.run_ctx.event(
            "act",
            actor="engine",
            step_id=step.id,
            action=step.action,
            target=step.target,
            locator_used=locator_used,
            value=value,
            value_sensitivity=sensitivity,
            url=self.page_surface.page.url,
            duration_ms=int((time.monotonic() - started) * 1000),
            ok=ok,
            error=error,
            **data,
        )

    def _event_race(
        self, step: Step, started: float, matched: str | None, disposition: Disposition
    ) -> None:
        self.run_ctx.event(
            "race",
            actor="engine",
            step_id=step.id,
            duration_ms=int((time.monotonic() - started) * 1000),
            ok=disposition.do in ("continue", "succeed"),
            matched=matched,
            then=disposition.do,
            code=disposition.code,
        )


# --- pure helpers ---------------------------------------------------------------------------


def _parse(raw: str, extraction: Extraction) -> str | None:
    text = raw.strip()
    if extraction.parse == "regex":
        match = re.search(extraction.regex or "", raw)
        if match is None:
            return None
        return match.group(1) if match.groups() else match.group(0)
    if extraction.parse == "integer":
        return text if re.fullmatch(r"[+-]?\d+", text) else None
    if extraction.parse == "decimal":
        try:
            return text if Decimal(text).is_finite() else None
        except InvalidOperation:
            return None
    return text


def _typed(spec: OutputSpec, value: str | None) -> Any:
    """The value in the output's declared type, or None if it is not one."""
    if value is None or (spec.pattern is not None and re.search(spec.pattern, value) is None):
        return None
    if spec.type == "integer":
        return int(value) if re.fullmatch(r"[+-]?\d+", value) else None
    if spec.type == "decimal":
        try:
            number = Decimal(value)
        except InvalidOperation:
            return None
        return str(number) if number.is_finite() else None
    if spec.type == "boolean":
        return {"true": True, "false": False}.get(value.lower())
    return value


def _dialog_regexes(signature: sig.Signature) -> list[str | None]:
    return [n.message_regex for n in sig.walk(signature) if isinstance(n, sig.DialogCondition)]


def _attempts(resolution: Resolution[PwLocator] | None) -> list[Attempt]:
    if resolution is None:
        return []
    return [
        Attempt(index=a.index, by=a.by, outcome=a.outcome, count=a.count, detail=a.detail)
        for a in resolution.attempts
    ]


def _failure(stop: _Fail, evidence: Evidence | None) -> Failure:
    return Failure(
        category=stop.category,
        code=stop.code,
        step_id=stop.step_id,
        expected=stop.expected,
        observed=stop.observed,
        attempts=stop.attempts,
        evidence=evidence,
    )


def _expected(step: Step) -> str:
    if step.expect:
        return "one of: " + "; ".join(e.description for e in step.expect)
    return step.intent


def _line(error: BaseException) -> str:
    return str(error).strip().split("\n", 1)[0][:300]
