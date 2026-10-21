"""The goal spec: what discovery is asked to do, and the contract the capability will offer.

A goal names the capability it should become, its typed inputs (each with a demo
value to use during discovery) and the outputs the caller wants back. Demo values
should be distinctive: the compiler recognises a param by finding its demo value
in what the model typed, so two params sharing a value could not be told apart.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from understudy.artifact.schema import InputSpec, OutputSpec, Sensitivity, ValueType
from understudy.replay.preflight import value_problems

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoalParam(_Model):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    type: ValueType
    description: str
    demo: str
    """The value discovery uses. A sandbox value; on a shared sandbox, one that is safe to use."""
    sensitivity: Sensitivity = "public"
    enum: list[str] | None = None
    pattern: str | None = None
    example: str | None = None
    """A fake value for the agent-facing tool definition."""

    def input_spec(self) -> InputSpec:
        return InputSpec(
            name=self.name,
            type=self.type,
            description=self.description,
            sensitivity=self.sensitivity,
            enum=self.enum,
            pattern=self.pattern,
            example=self.example,
        )


class GoalOutput(_Model):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    type: ValueType
    description: str
    sensitivity: Sensitivity = "public"
    pattern: str | None = None

    def output_spec(self, source_step: str) -> OutputSpec:
        return OutputSpec(
            name=self.name,
            type=self.type,
            description=self.description,
            sensitivity=self.sensitivity,
            pattern=self.pattern,
            source_step=source_step,
        )


class GoalSpec(_Model):
    schema_version: Literal["understudy.goal/v1"] = "understudy.goal/v1"
    capability: str = Field(pattern=r"^[a-z0-9_]+\.[a-z0-9_]+$")
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    title: str
    description: str
    """For the calling agent: what the capability does and what it needs."""
    vendor_app: str
    session: str
    """The session provider the capability requires (the app profile's login)."""
    goal: str
    """The task, with {param} placeholders. Shown to the model with demo values filled in."""
    start_route: str = Field(pattern=r"^/")
    """Where the task begins, relative to the tenant's base URL."""
    params: list[GoalParam] = Field(default_factory=list)
    outputs: list[GoalOutput] = Field(default_factory=list)
    side_effects: str | None = None
    """What the flow changes, for the artifact's side_effects.description."""

    @model_validator(mode="after")
    def _consistent(self) -> GoalSpec:
        names = [p.name for p in self.params] + [o.name for o in self.outputs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate param or output names: {', '.join(duplicates)}")
        unknown = set(_PLACEHOLDER.findall(self.goal)) - {p.name for p in self.params}
        if unknown:
            raise ValueError(f"goal names undeclared params: {', '.join(sorted(unknown))}")
        return self

    def values(self, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
        values = {p.name: p.demo for p in self.params}
        values.update(overrides or {})
        return values

    def value_problems(self, values: Mapping[str, str]) -> list[str]:
        """Problems name the param and the rule, never the value."""
        specs = {p.name: p for p in self.params}
        problems = [f"unknown param {name!r}" for name in sorted(set(values) - set(specs))]
        for name, spec in specs.items():
            if name not in values:
                problems.append(f"param {name!r} has no value")
                continue
            problems += [
                f"param {name!r} {p}" for p in value_problems(spec.input_spec(), values[name])
            ]
        seen: dict[str, str] = {}
        for name in sorted(values):
            other = seen.setdefault(values[name], name)
            if other != name:
                problems.append(
                    f"params {other!r} and {name!r} share a value, so what was typed could "
                    "not be traced back to one of them"
                )
        return problems

    def render(self, values: Mapping[str, str]) -> str:
        return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), self.goal)


def load_goal(path: Path) -> GoalSpec:
    return GoalSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
