"""The review summary: what a human reads before approving an artifact.

Reviewing raw JSON does not scale and hides the things approval is actually
about. This renders the decisions: what commits, what the caller is promised,
how each element will be found, and what lint thinks.
"""

from __future__ import annotations

import textwrap

from understudy.artifact.hashing import content_hash
from understudy.artifact.lint import LintReport, lint
from understudy.artifact.schema import (
    Capability,
    LiteralValue,
    ParamValue,
    SecretValue,
    Step,
    ValueRef,
)

EFFECT_MARK = {"read": " ", "input": "·", "commit": "!"}


def render(capability: Capability, report: LintReport | None = None) -> str:
    report = report or lint(capability)
    lines: list[str] = []

    lines += _header(capability)
    lines += _contract(capability)
    lines += _steps(capability)
    lines += _lint(report)
    return "\n".join(lines) + "\n"


def _header(capability: Capability) -> list[str]:
    life = capability.lifecycle
    status: str = life.status
    if life.status == "approved":
        status += f" by {life.reviewed_by}"
    stale = (
        life.status == "approved"
        and life.approved_hash is not None
        and life.approved_hash != content_hash(capability)
    )
    return [
        f"{capability.id}@{capability.version}  {capability.title}",
        f"  status      {status}" + ("  [EDITED SINCE APPROVAL]" if stale else ""),
        f"  hash        {content_hash(capability)}",
        f"  app         {capability.app.vendor_app} ({capability.app.surface}), "
        f"entry {capability.app.entry_route}",
        f"  session     {capability.requires.session}",
        f"  discovered  run {capability.provenance.discovery_run_id} "
        f"against {capability.provenance.recorded_against} "
        f"({capability.provenance.authored})",
        *(
            [f"  by a person {', '.join(capability.provenance.human_steps)}  [REVIEW EACH]"]
            if capability.provenance.human_steps
            else []
        ),
        "",
    ]


def _contract(capability: Capability) -> list[str]:
    lines = ["CONTRACT (what a calling agent sees)", f"  {capability.description.strip()}", ""]

    lines.append("  inputs")
    for spec in capability.inputs:
        bits: list[str] = [spec.type]
        if spec.enum:
            bits.append("|".join(spec.enum))
        if spec.pattern:
            bits.append(f"~{spec.pattern}")
        if not spec.required:
            bits.append("optional")
        if spec.sensitivity != "public":
            bits.append(spec.sensitivity)
        lines.append(f"    {spec.name:<22} {', '.join(bits)}")
        lines.append(f"    {'':<22} {spec.description.strip()}")

    lines.append("  outputs")
    for output in capability.outputs:
        pattern = f" ~{output.pattern}" if output.pattern else ""
        lines.append(f"    {output.name:<22} {output.type}{pattern}  from {output.source_step}")

    effects = capability.side_effects
    lines += [
        "  side effects",
        f"    class {effects.cls}, "
        f"{'idempotent' if effects.idempotent else 'not idempotent'}, "
        f"approval {effects.approval}",
        f"    {effects.description.strip()}",
    ]

    if capability.outcomes:
        lines.append("  business outcomes")
        for outcome in capability.outcomes:
            lines.append(
                f"    {outcome.code:<30} at {outcome.detected_by}: {outcome.description.strip()}"
            )

    lines += [
        "  recovery",
        f"    {', '.join(capability.recovery.allowed) or 'none'} "
        f"(budget {capability.recovery.budget})",
        "",
    ]
    return lines


def _steps(capability: Capability) -> list[str]:
    lines = ["STEPS  ('!' commits, '·' writes input, blank reads)"]
    width = max(len(step.id) for step in capability.steps)
    for step in capability.steps:
        mark = EFFECT_MARK[step.effect]
        value = _value(step.value)
        lines.append(
            f"  {mark} {step.id:<{width}}  {step.action:<9} {value:<24} {step.intent.strip()}"
        )
        if step.id in capability.provenance.human_steps:
            lines.append(
                "      recorded from a person's action during a takeover: confirm it belongs "
                "in the flow"
            )
        lines += _step_detail(step)
    lines.append("")
    return lines


def _step_detail(step: Step) -> list[str]:
    lines = []
    pad = " " * 6
    if step.url:
        lines.append(f"{pad}url      {step.url}")
    if step.target:
        lines.append(f"{pad}target   {step.target.description.strip()}")
        for index, locator in enumerate(step.target.locators):
            tick = "ok " if locator.verified and locator.verified.unique else "?? "
            rank = _ordinal(index + 1)
            lines.append(
                f"{pad}  {tick}{rank:<4} {locator.by:<10} {locator.stability:<6} {locator.value}"
            )
            lines += _wrap(locator.note, pad + " " * 6)
    if step.wait_before:
        lines.append(f"{pad}waits    {step.wait_before.kind}")
    for expectation in step.expect:
        disposition = expectation.then.do + (
            f"({expectation.then.code})" if expectation.then.code else ""
        )
        lines.append(f"{pad}expect   {disposition:<32} {expectation.description.strip()}")
    no_match = step.resolved_on_no_match
    lines.append(f"{pad}else     {no_match.do}" + (f"({no_match.code})" if no_match.code else ""))
    if step.on_target_absent:
        lines.append(f"{pad}absent   business({step.on_target_absent.outcome})")
    if step.extract:
        lines.append(f"{pad}extract  {step.extract.output} <- {step.extract.read}")
    if step.effect == "commit" and step.retry.max_attempts == 0:
        lines.append(f"{pad}         commit is never retried")
    if step.reentry_point:
        lines.append(f"{pad}         safe re-entry point after session recovery")
    return lines


def _lint(report: LintReport) -> list[str]:
    if not report.findings:
        return ["LINT  clean"]
    lines = [f"LINT  {len(report.errors)} error(s), {len(report.warnings)} warning(s)"]
    lines += [f"  {finding}" for finding in report.findings]
    return lines


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _wrap(text: str, indent: str, width: int = 92) -> list[str]:
    return [indent + line for line in textwrap.wrap(" ".join(text.split()), width)]


def _value(value: ValueRef | None) -> str:
    match value:
        case ParamValue():
            return f"{{{value.param}}}"
        case SecretValue():
            return f"secret:{value.secret_ref}"
        case LiteralValue():
            return repr(value.literal)
        case _:
            return ""
