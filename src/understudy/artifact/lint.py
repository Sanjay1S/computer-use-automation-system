"""Artifact lint: the rules that make an artifact safe to approve.

These run in three places: inside the compiler (so a discovered draft is checked
before anyone reads it), in `understudy lint`, and at approval time. L007 also
runs when replay loads an approved artifact, because a policy can tighten after
approval and must then block the artifact rather than let it run.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from understudy.artifact import signatures as sig
from understudy.artifact.schema import (
    Capability,
    LiteralValue,
    Locator,
    ParamValue,
    Target,
)

Severity = Literal["error", "warning"]

SENSITIVE_PARAM_KEYS = ("jsessionid", "sessionid", "token", "password", "secret")


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    where: str

    def __str__(self) -> str:
        return f"{self.code} [{self.severity}] {self.where}: {self.message}"


@dataclass(frozen=True)
class LintReport:
    findings: list[Finding]

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> set[str]:
        return {f.code for f in self.findings}


class PolicyView:
    """What lint needs from a policy to check L007. The real gate (`PolicyGate`) implements
    this; lint takes the narrow view so the artifact package stays standalone."""

    def allows_route(self, route: str) -> bool:  # pragma: no cover - protocol default
        return True

    def allows_locator_value(self, value: str) -> bool:  # pragma: no cover
        return True


def lint(
    capability: Capability,
    policy: PolicyView | None = None,
    run_values: Mapping[str, str] | None = None,
) -> LintReport:
    """`run_values` maps the values one run happened to use or read (a free-form
    input, an output's sample) to what they were ("input funding_account_id").
    Only the compiler knows them, so L012 runs there."""
    findings: list[Finding] = []
    findings += _l001_vacuous_success(capability)
    findings += _l002_sensitive_literals(capability)
    findings += _l003_commit_retries(capability)
    findings += _l004_param_binding(capability)
    findings += _l005_l006_locators(capability)
    findings += _l007_policy_drift(capability, policy)
    findings += _l008_outputs(capability)
    findings += _l009_commit_approval(capability)
    findings += _l010_baked_in_url(capability)
    findings += _l011_outcome_refs(capability)
    findings += _l012_run_values_in_text(capability, run_values or {})
    return LintReport(findings=findings)


def _l001_vacuous_success(capability: Capability) -> list[Finding]:
    """A success checkpoint that holds before the flow runs proves nothing."""
    findings = []
    where = "success"
    if not sig.has_element_condition(capability.success.signature):
        findings.append(
            Finding(
                "L001",
                "error",
                "the success checkpoint has no element-level condition; a URL alone "
                "also matches an error page served on the same route",
                where,
            )
        )
    if capability.success.negative_control is None:
        findings.append(
            Finding(
                "L001",
                "error",
                "the success checkpoint has no negative control, so there is no "
                "evidence it is false on the start state",
                where,
            )
        )
    return findings


def _l002_sensitive_literals(capability: Capability) -> list[Finding]:
    """Literals are public by definition; anything else is a param or a secret ref."""
    findings = []
    sensitivity = {i.name: i.sensitivity for i in capability.inputs}
    for step in capability.steps:
        where = f"steps.{step.id}"
        if isinstance(step.value, LiteralValue) and _looks_sensitive(step.value.literal):
            findings.append(
                Finding(
                    "L002",
                    "error",
                    f"literal value {_preview(step.value.literal)} looks sensitive; "
                    "use a param or a secret_ref",
                    where,
                )
            )
        if isinstance(step.value, ParamValue):
            declared = sensitivity.get(step.value.param)
            if declared == "secret":
                findings.append(
                    Finding(
                        "L002",
                        "error",
                        f"param {step.value.param!r} is declared 'secret'; secrets must "
                        "travel as secret_ref, not as an input param",
                        where,
                    )
                )
    return findings


def _l003_commit_retries(capability: Capability) -> list[Finding]:
    """Retrying a commit can move money twice."""
    return (
        [
            Finding(
                "L003",
                "error",
                f"commit step has retry.max_attempts={step.retry.max_attempts}; a commit "
                "is never retried",
                f"steps.{step.id}",
            )
            for step in capability.steps
            if step.effect == "commit" and step.retry.max_attempts > 0
        ]
        + [
            Finding(
                "L003",
                "error",
                f"commit step treats an unmatched outcome as "
                f"{step.on_no_match.do!r}; after a commit, anything that is not a "
                "definite outcome is unknown and goes to a human",
                f"steps.{step.id}",
            )
            for step in capability.steps
            if step.effect == "commit"
            and step.on_no_match is not None
            and step.on_no_match.do in ("succeed", "continue")
        ]
        + [
            Finding(
                "L003",
                "error",
                "commit step has no expectations, so nothing would show whether it committed",
                f"steps.{step.id}",
            )
            for step in capability.steps
            if step.effect == "commit" and not step.expect
        ]
    )


def bound_params(capability: Capability) -> dict[str, set[str]]:
    """The params each part of the artifact uses, keyed by a readable location
    ("step 's3'", "the success checkpoint"). L004 and replay's preflight share it."""
    where: dict[str, set[str]] = {}
    for step in capability.steps:
        used: set[str] = set()
        if isinstance(step.value, ParamValue):
            used.add(step.value.param)
        used |= _params_in_target(step.target)
        if step.wait_before:
            used |= sig.params_used(step.wait_before)
        for expectation in step.expect:
            used |= sig.params_used(expectation.when)
        if step.url:
            used |= _template_params(step.url) - {"base_url"}
        where[f"step {step.id!r}"] = used
    success = sig.params_used(capability.success.signature)
    if capability.success.negative_control:
        success |= sig.params_used(capability.success.negative_control)
    where["the success checkpoint"] = success
    return where


def _l004_param_binding(capability: Capability) -> list[Finding]:
    """Every declared param is used, and every used param is declared."""
    declared = {i.name for i in capability.inputs}
    used: set[str] = set().union(*bound_params(capability).values())

    findings = [
        Finding("L004", "error", f"input {name!r} is declared but never bound", "inputs")
        for name in sorted(declared - used)
    ]
    findings += [
        Finding("L004", "error", f"param {name!r} is bound but never declared", "steps")
        for name in sorted(used - declared)
    ]
    return findings


def _l005_l006_locators(capability: Capability) -> list[Finding]:
    """Every target needs a locator verified at discovery.

    L006 judges the locator replay will reach for *first*, which is the first
    verified one in declared order, not the most stable one in the list: the
    fallback chain is tried in order, so a low-stability head is what drifts.
    """
    findings = []
    for step in capability.steps:
        if step.target is None:
            continue
        where = f"steps.{step.id}.target"
        verified = [
            loc
            for loc in step.target.locators
            if loc.verified and loc.verified.unique and loc.verified.same_element
        ]
        if not verified:
            findings.append(
                Finding(
                    "L005",
                    "error",
                    "no locator was verified unique and same-element at discovery",
                    where,
                )
            )
            continue
        if verified[0].stability == "low":
            findings.append(
                Finding(
                    "L006",
                    "warning",
                    f"the first verified locator is low stability ({verified[0].by}); expect drift",
                    where,
                )
            )
    return findings


def _l007_policy_drift(capability: Capability, policy: PolicyView | None) -> list[Finding]:
    """A policy can tighten after approval. The artifact must then stop loading."""
    if policy is None:
        return []
    findings = []
    for step in capability.steps:
        if step.url and not policy.allows_route(_route_of(step.url)):
            findings.append(
                Finding(
                    "L007",
                    "error",
                    f"url {step.url!r} is outside the current policy",
                    f"steps.{step.id}",
                )
            )
        for locator in _all_locators(step.target):
            for value in _strings_in(locator.value):
                if not policy.allows_locator_value(value):
                    findings.append(
                        Finding(
                            "L007",
                            "error",
                            f"locator {value!r} is outside the current policy",
                            f"steps.{step.id}.target",
                        )
                    )
    return findings


def _l008_outputs(capability: Capability) -> list[Finding]:
    """An output nothing extracts is a promise to the caller that replay cannot keep."""
    findings = []
    extracted: dict[str, list[str]] = {}
    for step in capability.steps:
        if step.extract:
            extracted.setdefault(step.extract.output, []).append(step.id)

    for output in capability.outputs:
        where = f"outputs.{output.name}"
        writers = extracted.get(output.name, [])
        if not writers:
            findings.append(Finding("L008", "error", "no step extracts this output", where))
            continue
        if len(writers) > 1:
            findings.append(
                Finding(
                    "L008",
                    "error",
                    f"steps {', '.join(writers)} all extract this output; which value "
                    "the caller receives would depend on step order",
                    where,
                )
            )
        if output.source_step not in writers:
            findings.append(
                Finding(
                    "L008",
                    "error",
                    f"source_step is {output.source_step!r} but step {writers[0]!r} extracts it",
                    where,
                )
            )
    for step in capability.steps:
        if step.extract and step.extract.output not in {o.name for o in capability.outputs}:
            findings.append(
                Finding(
                    "L008",
                    "error",
                    f"step extracts {step.extract.output!r}, which is not a declared output",
                    f"steps.{step.id}",
                )
            )
    return findings


def _l009_commit_approval(capability: Capability) -> list[Finding]:
    """A commit that nobody has to approve is a commit nobody is accountable for."""
    findings = []
    commits = [s for s in capability.steps if s.effect == "commit"]
    if commits and capability.side_effects.cls != "commit":
        findings.append(
            Finding(
                "L009",
                "error",
                f"step {commits[0].id!r} commits, but side_effects.class is "
                f"{capability.side_effects.cls!r}",
                "side_effects",
            )
        )
    if capability.side_effects.cls == "commit" and capability.side_effects.approval == "none":
        findings.append(
            Finding(
                "L009",
                "error",
                "a commit capability needs approval='per_invocation'",
                "side_effects",
            )
        )
    return findings


def _l010_baked_in_url(capability: Capability) -> list[Finding]:
    """A host belongs to the tenant binding; a session id belongs to nothing on disk.

    Locators are checked as well as urls: a selector matching on a form action or
    an href can carry a host or a jsessionid just as easily.
    """
    findings = []
    for step in capability.steps:
        for locator in _all_locators(step.target):
            for value in _strings_in(locator.value):
                if "jsessionid" in value.lower():
                    findings.append(
                        Finding(
                            "L010",
                            "error",
                            f"locator {value!r} carries a session id",
                            f"steps.{step.id}.target",
                        )
                    )
                if _has_host(value):
                    findings.append(
                        Finding(
                            "L010",
                            "error",
                            f"locator {value!r} carries a host; the host is a tenant "
                            "binding, not part of the capability",
                            f"steps.{step.id}.target",
                        )
                    )
        if not step.url:
            continue
        where = f"steps.{step.id}"
        if not step.url.startswith("{base_url}"):
            findings.append(
                Finding(
                    "L010",
                    "error",
                    f"url {step.url!r} does not start with {{base_url}}; the host is a "
                    "tenant binding, not part of the capability",
                    where,
                )
            )
        if "jsessionid" in step.url.lower():
            findings.append(
                Finding(
                    "L010",
                    "error",
                    f"url {step.url!r} carries a session id",
                    where,
                )
            )
    return findings


def _l011_outcome_refs(capability: Capability) -> list[Finding]:
    """Outcomes are part of the agent-facing contract, so every reference to one
    must resolve: a caller told about an outcome no step can raise is misled, and a
    step raising an outcome the caller was never told about is worse."""
    findings = []
    declared = {outcome.code for outcome in capability.outcomes}
    step_ids = {step.id for step in capability.steps}

    for outcome in capability.outcomes:
        if outcome.detected_by not in step_ids:
            findings.append(
                Finding(
                    "L011",
                    "error",
                    f"detected_by names step {outcome.detected_by!r}, which does not exist",
                    f"outcomes.{outcome.code}",
                )
            )

    for step in capability.steps:
        if step.on_target_absent and step.on_target_absent.outcome not in declared:
            findings.append(
                Finding(
                    "L011",
                    "error",
                    f"on_target_absent names outcome {step.on_target_absent.outcome!r}, "
                    "which is not declared",
                    f"steps.{step.id}",
                )
            )
        for expectation in step.expect:
            code = expectation.then.code
            if expectation.then.do == "business" and code not in declared:
                findings.append(
                    Finding(
                        "L011",
                        "error",
                        f"expect raises business outcome {code!r}, which is not declared",
                        f"steps.{step.id}",
                    )
                )
    return findings


FREE_TEXT_KEYS = frozenset({"title", "description", "intent", "note"})


def _l012_run_values_in_text(capability: Capability, values: Mapping[str, str]) -> list[Finding]:
    """A value from the discovery run in a field people and agents read as the
    contract: a sample account id in the success description would be shown to
    every caller as if it were part of the capability, and it is customer data.
    Values shorter than three characters are skipped: they match by accident. The
    message names the value's source, never the value."""
    patterns = [
        (label, re.compile(rf"(?<![\w]){re.escape(value)}(?![\w])"))
        for value, label in sorted(values.items())
        if len(value) >= 3
    ]
    if not patterns:
        return []
    findings: list[Finding] = []
    for where, text in _free_text(capability.model_dump(mode="json", by_alias=True), ""):
        for label, pattern in patterns:
            if pattern.search(text):
                findings.append(
                    Finding(
                        "L012",
                        "error",
                        f"free text holds the discovery run's value of {label}; "
                        "describe the capability, not one run of it",
                        where,
                    )
                )
    return findings


def _free_text(node: Any, path: str) -> list[tuple[str, str]]:
    if isinstance(node, dict):
        found: list[tuple[str, str]] = []
        for key, value in node.items():
            where = f"{path}.{key}" if path else key
            if key in FREE_TEXT_KEYS and isinstance(value, str):
                found.append((where, value))
            else:
                found += _free_text(value, where)
        return found
    if isinstance(node, list):
        return [pair for i, item in enumerate(node) for pair in _free_text(item, f"{path}[{i}]")]
    return []


def _has_host(value: str) -> bool:
    return bool(re.search(r"https?://|(?<![\w.])localhost(?::\d+)?(?![\w.])", value.lower()))


def _all_locators(target: Target | None) -> list[Locator]:
    """Every locator a target carries. The scope locator is one of them: it is
    resolved against the live page like any other, so every rule that inspects
    locators must see it too."""
    if target is None:
        return []
    return [*target.locators, *([target.scope] if target.scope else [])]


def _params_in_target(target: Target | None) -> set[str]:
    """Params referenced by a templated locator.

    Only `templated` locators are scanned, and only their string parts: a role
    locator's value is a dict whose repr is full of braces, which would otherwise
    read as param names.
    """
    params: set[str] = set()
    for locator in _all_locators(target):
        if not locator.templated:
            continue
        for text in _strings_in(locator.value):
            params |= _template_params(text)
    return params


def _strings_in(value: str | dict[str, Any]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [part for part in value.values() if isinstance(part, str)]


def _template_params(text: str) -> set[str]:
    params: set[str] = set()
    rest = text
    while "{" in rest:
        start = rest.index("{")
        if "}" not in rest[start:]:
            break
        end = rest.index("}", start)
        params.add(rest[start + 1 : end])
        rest = rest[end + 1 :]
    return params


def _route_of(url: str) -> str:
    route = url.removeprefix("{base_url}")
    return route.split("?", 1)[0] or "/"


def _looks_sensitive(value: str) -> bool:
    """Keywords match whole tokens, not substrings: 'Secretary' and 'Password Reset'
    are ordinary button labels in a banking UI, and L002 is an error."""
    tokens = {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}
    if tokens & set(SENSITIVE_PARAM_KEYS):
        return True
    digits = [c for c in value if c.isdigit()]
    return len(digits) >= 9 and len(digits) >= len(value) - 3


def _preview(value: str) -> str:
    return repr(value if len(value) <= 12 else value[:9] + "...")
