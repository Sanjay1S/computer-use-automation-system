"""One discovery run, end to end: checks, login, the model loop, compile, draft.

Nothing reaches the model before the session provider has signed in, so the model
never sees credentials. The run folder gets the redacted trajectory, the event log,
masked screenshots, token usage, the draft artifact and its review summary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar, Literal

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page
from pydantic import BaseModel, ConfigDict, Field

from understudy.artifact import hashing
from understudy.artifact.review import render
from understudy.artifact.schema import Sensitivity
from understudy.discovery.agent import MODEL, Budget, DiscoveryAgent, ModelTurn
from understudy.discovery.compiler import CompileError, compile_trajectory
from understudy.discovery.driver import Approver, DiscoveryBrowser, refuse_all
from understudy.discovery.goal import GoalSpec
from understudy.discovery.prompts import goal_message
from understudy.discovery.recorder_tools import RecorderTools
from understudy.discovery.trajectory import Recorder
from understudy.evidence.run import RunContext
from understudy.evidence.usage import Usage
from understudy.policy.gate import PolicyGate
from understudy.profile import AppProfile
from understudy.redaction.redactor import Redactor
from understudy.session import Credentials, FormLogin
from understudy.surface.base import SurfaceError
from understudy.surface.web import WebSurface

Status = Literal["compiled", "rejected", "gave_up", "stuck", "failed"]


class DiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    IDENTIFIERS: ClassVar[tuple[str, ...]] = (
        "status",
        "run_id",
        "capability",
        "tenant",
        "draft",
        "artifact_hash",
    )

    status: Status
    run_id: str
    capability: str
    tenant: str
    code: str | None = None
    reason: str | None = None
    draft: str | None = None
    artifact_hash: str | None = None
    lint_ok: bool | None = None
    lint: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    turns: int = 0
    seconds: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    models_served: list[str] = Field(default_factory=list)


def discover(
    goal: GoalSpec,
    values: Mapping[str, str],
    *,
    gate: PolicyGate,
    profile: AppProfile,
    credentials: Credentials,
    model: ModelTurn,
    open_page: Callable[[], Page],
    runs_dir: Path,
    approve: Approver = refuse_all,
    draft_path: Path | None = None,
    redactor: Redactor | None = None,
    budget: Budget | None = None,
    model_id: str = MODEL,
    effort: str = "high",
    screenshots: bool = True,
) -> DiscoveryResult:
    """`open_page` is called only once the run's checks have passed."""
    sensitivity = {p.name: p.sensitivity for p in goal.params}
    known: dict[str, Sensitivity] = {v: sensitivity.get(k, "pii") for k, v in values.items() if v}
    known.update(credentials.sensitivities)
    redactor = (redactor or Redactor.from_env()).knowing(known)
    run = RunContext(
        runs_dir,
        kind="discovery",
        redactor=redactor,
        meta={
            "capability": goal.capability,
            "tenant": gate.tenant.id,
            "model": model_id,
            "effort": effort,
            "params": {k: redactor.value(v, sensitivity.get(k, "pii")) for k, v in values.items()},
        },
        verbatim=("capability", "tenant", "model", "effort"),
    )
    base = DiscoveryResult(
        status="rejected", run_id=run.run_id, capability=goal.capability, tenant=gate.tenant.id
    )

    problems = _checks(goal, values, gate, profile)
    if problems:
        return _finish(
            run, base.model_copy(update={"code": "rejected", "reason": "; ".join(problems)})
        )

    recorder: Recorder | None = None
    agent: DiscoveryAgent | None = None
    try:
        surface = WebSurface(open_page(), gate, mode="discovery", run_id=run.run_id)
        FormLogin(profile.login, gate.tenant, credentials).login(surface)
        run.event("act", actor="system", member="login", ok=True)
        recorder = Recorder(
            run, surface, goal, values, masks=profile.screenshot_masks, screenshots=screenshots
        )
        recorder.start()
        browser = DiscoveryBrowser(surface, recorder, approve=approve)
        agent = DiscoveryAgent(
            model=model,
            browser=browser,
            tools=RecorderTools(goal, recorder, surface),
            goal_text=goal_message(goal, values, gate.tenant.base_url),
            run=run,
            on_turn=recorder.begin_turn,
            budget=budget or Budget(),
            model_id=model_id,
            effort=effort,
        )
        outcome = agent.run_loop()
    except Exception as error:  # the result must be written whatever broke
        if isinstance(error, SurfaceError | PlaywrightError):
            code = error.code if isinstance(error, SurfaceError) else "infra_error"
            reason = _line(error)
        else:
            code, reason = "engine_error", f"{type(error).__name__}: {_line(error)}"
        spent = (
            {"turns": agent.turns, "usage": agent.usage, "models_served": agent.usage.models}
            if agent is not None
            else {}
        )
        failed = base.model_copy(
            update={"status": "failed", "code": code, "reason": reason, **spent}
        )
        return _finish(run, failed, recorder)

    summary = base.model_copy(
        update={
            "status": outcome.status if outcome.status != "succeeded" else "failed",
            "code": outcome.code,
            "reason": outcome.reason,
            "turns": outcome.turns,
            "seconds": outcome.seconds,
            "usage": outcome.usage,
            "models_served": outcome.models_served,
        }
    )
    if outcome.status != "succeeded":
        return _finish(run, summary, recorder)

    try:
        compiled = compile_trajectory(
            recorder.trajectory, goal, gate, models_served=outcome.models_served
        )
    except CompileError as error:
        return _finish(
            run,
            summary.model_copy(
                update={"code": "uncompilable", "reason": str(error), "notes": error.problems}
            ),
            recorder,
        )

    capability = compiled.capability
    destination = draft_path or Path("capabilities") / "drafts" / f"{capability.id}.json"
    artifact_hash = hashing.save(capability, destination)
    hashing.save(capability, run.dir / "draft.json")
    review = render(capability, compiled.report)
    if compiled.notes:
        review += "\nCompiler notes\n" + "".join(f"  - {note}\n" for note in compiled.notes)
    (run.dir / "review.txt").write_text(review, encoding="utf-8")
    return _finish(
        run,
        summary.model_copy(
            update={
                "status": "compiled",
                "draft": str(destination),
                "artifact_hash": artifact_hash,
                "lint_ok": compiled.report.ok,
                "lint": [str(f) for f in compiled.report.findings],
                "notes": compiled.notes,
            }
        ),
        recorder,
    )


def _checks(
    goal: GoalSpec, values: Mapping[str, str], gate: PolicyGate, profile: AppProfile
) -> list[str]:
    problems = []
    decision = gate.check_discovery_allowed()
    if not decision.allowed:
        problems.append(decision.reason)
    if goal.vendor_app != gate.policy.vendor_app or goal.vendor_app != profile.vendor_app:
        problems.append(
            f"the goal is for {goal.vendor_app!r}; the policy or profile is for another app"
        )
    if goal.session != profile.login.session:
        problems.append(
            f"the goal needs session {goal.session!r}; "
            f"the profile provides {profile.login.session!r}"
        )
    problems += goal.value_problems(values)
    return problems


def _finish(
    run: RunContext, result: DiscoveryResult, recorder: Recorder | None = None
) -> DiscoveryResult:
    """Every run, however it ended, leaves its result, its trajectory and what it cost."""
    if recorder is not None:
        recorder.persist()
    if result.usage.by_model:
        run.write_json("usage.json", result.usage)
    run.write_json("result.json", result, verbatim=DiscoveryResult.IDENTIFIERS)
    run.event(
        "result",
        actor="system",
        ok=result.status == "compiled",
        status=result.status,
        code=result.code,
    )
    return result


def _line(error: BaseException) -> str:
    return str(error).strip().split("\n", 1)[0][:300]
