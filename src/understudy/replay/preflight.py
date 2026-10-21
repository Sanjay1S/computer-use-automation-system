"""Preflight: every check that can be made before replay touches the UI. A run that
fails one is `rejected` and nothing is executed."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

from understudy.artifact import hashing
from understudy.artifact.lint import LintReport, bound_params, lint
from understudy.artifact.schema import Capability, InputSpec, LiteralValue, SecretValue
from understudy.policy.gate import REPLAY_MEMBER, PolicyGate
from understudy.profile import AppProfile
from understudy.replay.result import Rejection

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_INTEGER = re.compile(r"[+-]?\d+")


class CommitApproval(BaseModel):
    """A grant for one commit: this capability, at this content hash, with these
    inputs, on this tenant. A stand-in for a signed grant from an approver; the CLI
    mints one per invocation from `--approve`, and the console (M6) will too."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: str
    version: str
    artifact_hash: str
    inputs_hash: str
    tenant: str
    granted_by: str


def inputs_hash(inputs: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(inputs), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def grant(
    capability: Capability, inputs: Mapping[str, str], tenant: str, granted_by: str
) -> CommitApproval:
    return CommitApproval(
        capability=capability.id,
        version=capability.version,
        artifact_hash=hashing.content_hash(capability),
        inputs_hash=inputs_hash(inputs),
        tenant=tenant,
        granted_by=granted_by,
    )


def commits(capability: Capability) -> bool:
    return capability.side_effects.cls == "commit" or any(
        s.effect == "commit" for s in capability.steps
    )


def expand_url(template: str, base_url: str, params: Mapping[str, str]) -> str:
    """`{base_url}` from the tenant; any other placeholder from the inputs, encoded."""

    def one(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "base_url":
            return base_url
        if name not in params:
            raise ValueError(f"url placeholder {{{name}}} has no input")
        return quote(params[name], safe="")

    return _PLACEHOLDER.sub(one, template)


def preflight(
    capability: Capability,
    inputs: Mapping[str, str],
    *,
    gate: PolicyGate,
    profile: AppProfile,
    approval: CommitApproval | None,
    allow_draft: bool,
    fault_injection: bool = False,
) -> Rejection | None:
    """Checks run in order, and the first group with problems is the rejection."""
    report = lint(capability, gate)
    artifact_hash = hashing.content_hash(capability)
    problems = _artifact_problems(capability, report, profile)
    if problems:
        return Rejection(code="invalid_artifact", problems=problems)
    problems = _lifecycle_problems(capability, artifact_hash, allow_draft)
    if problems:
        return Rejection(code="not_approved", problems=problems)
    problems = check_inputs(capability, inputs)
    if problems:
        return Rejection(code="invalid_input", problems=problems)
    problems = _policy_problems(capability, report, inputs, gate)
    if fault_injection and not gate.tenant.sandbox:
        problems.append(f"fault injection runs only on sandbox tenants; {gate.tenant.id!r} is not")
    if problems:
        return Rejection(code="policy", problems=problems)
    problems = _approval_problems(capability, artifact_hash, inputs, gate.tenant.id, approval)
    if problems:
        return Rejection(code="approval_required", problems=problems)
    return None


def check_inputs(capability: Capability, inputs: Mapping[str, str]) -> list[str]:
    """Problems name the input and the rule, never the value: a rejected value may
    still be sensitive."""
    specs = {spec.name: spec for spec in capability.inputs}
    problems = [f"unknown input {name!r}" for name in sorted(set(inputs) - set(specs))]
    for name, spec in specs.items():
        if name not in inputs:
            if spec.required:
                problems.append(f"input {name!r} is required")
            continue
        problems += [f"input {name!r} {p}" for p in value_problems(spec, inputs[name])]
    for where, names in bound_params(capability).items():
        for name in sorted(names - set(inputs)):
            if name in specs and not specs[name].required:
                problems.append(f"input {name!r} is optional but {where} needs it")
    return problems


def value_problems(spec: InputSpec, value: str) -> list[str]:
    problems: list[str] = []
    number: Decimal | None = None
    if spec.type == "enum" and value not in (spec.enum or []):
        problems.append(f"must be one of {', '.join(spec.enum or [])}")
    elif spec.type == "integer":
        if _INTEGER.fullmatch(value) is None:
            problems.append("must be an integer")
        else:
            number = Decimal(value)
    elif spec.type == "decimal":
        try:
            number = Decimal(value)
        except InvalidOperation:
            problems.append("must be a decimal number")
        else:
            if not number.is_finite():
                problems.append("must be a finite number")
                number = None
    elif spec.type == "boolean" and value.lower() not in ("true", "false"):
        problems.append("must be true or false")
    elif spec.type == "date":
        try:
            date.fromisoformat(value)
        except ValueError:
            problems.append("must be an ISO date (YYYY-MM-DD)")
    if number is not None:
        if spec.min is not None and number < Decimal(str(spec.min)):
            problems.append(f"must be at least {spec.min:g}")
        if spec.max is not None and number > Decimal(str(spec.max)):
            problems.append(f"must be at most {spec.max:g}")
    if spec.pattern is not None and re.search(spec.pattern, value) is None:
        problems.append(f"does not match {spec.pattern}")
    return problems


def _artifact_problems(
    capability: Capability, report: LintReport, profile: AppProfile
) -> list[str]:
    problems = [str(f) for f in report.errors if f.code != "L007"]
    if capability.app.vendor_app != profile.vendor_app:
        problems.append(
            f"the artifact is for {capability.app.vendor_app!r}, the profile for "
            f"{profile.vendor_app!r}"
        )
    if capability.requires.session != profile.login.session:
        problems.append(
            f"the artifact needs session {capability.requires.session!r}; the app profile "
            f"provides {profile.login.session!r}"
        )
    for step in capability.steps:
        if isinstance(step.value, SecretValue):
            problems.append(
                f"step {step.id!r} takes a secret; replay has no secret store yet "
                "(credentials go through the session provider)"
            )
    return problems


def _lifecycle_problems(capability: Capability, artifact_hash: str, allow_draft: bool) -> list[str]:
    lifecycle = capability.lifecycle
    if lifecycle.status == "deprecated":
        return ["the artifact is deprecated"]
    if lifecycle.status == "approved":
        if lifecycle.approved_hash == artifact_hash:
            return []
        if not allow_draft:
            return ["the artifact changed after it was approved: its hash no longer matches"]
        return []
    return [] if allow_draft else ["the artifact is a draft; approve it, or pass --allow-draft"]


def _policy_problems(
    capability: Capability, report: LintReport, inputs: Mapping[str, str], gate: PolicyGate
) -> list[str]:
    """L007 is lint's policy-drift rule; the rest are what lint cannot see: enabled
    members, allowed keys, and the full URL (scheme and host) after expansion."""
    problems = [str(f) for f in report.errors if f.code == "L007"]
    for step in capability.steps:
        decision = gate.check_member(REPLAY_MEMBER[step.action])
        if not decision.allowed:
            problems.append(f"step {step.id!r}: {decision.reason}")
        if step.action == "press" and isinstance(step.value, LiteralValue):
            decision = gate.check_key(step.value.literal)
            if not decision.allowed:
                problems.append(f"step {step.id!r}: {decision.reason}")
        if step.url:
            try:
                url = expand_url(step.url, gate.tenant.base_url, inputs)
            except ValueError as error:
                problems.append(f"step {step.id!r}: {error}")
                continue
            decision = gate.check_url(url)
            if not decision.allowed:
                problems.append(f"step {step.id!r}: {decision.reason}")
    return problems


def _approval_problems(
    capability: Capability,
    artifact_hash: str,
    inputs: Mapping[str, str],
    tenant: str,
    approval: CommitApproval | None,
) -> list[str]:
    if not commits(capability):
        return []
    if approval is None:
        return ["this capability commits; it needs an approval for this invocation (--approve)"]
    expected = CommitApproval(
        capability=capability.id,
        version=capability.version,
        artifact_hash=artifact_hash,
        inputs_hash=inputs_hash(inputs),
        tenant=tenant,
        granted_by=approval.granted_by,
    )
    return [
        f"the approval was granted for a different {field}"
        for field in ("capability", "version", "artifact_hash", "inputs_hash", "tenant")
        if getattr(approval, field) != getattr(expected, field)
    ]
