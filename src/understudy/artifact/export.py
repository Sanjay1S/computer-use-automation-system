"""Export: the JSON Schema for artifacts, and the tool definition a calling agent sees.

The tool definition is the contract-first rule made concrete. It is built from
`description`, `inputs`, `outputs`, `side_effects` and `outcomes` only; `steps`
never appear, because how the capability does its job is not the caller's
business and would change under it on every re-discovery.
"""

from __future__ import annotations

from typing import Any

from understudy.artifact.schema import Capability, InputSpec, OutputSpec

JSON_TYPES: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "decimal": "number",
    "boolean": "boolean",
    "enum": "string",
    "date": "string",
}


def capability_json_schema() -> dict[str, Any]:
    """The JSON Schema for the artifact format itself."""
    return Capability.model_json_schema(by_alias=True)


def tool_definition(capability: Capability) -> dict[str, Any]:
    """The capability as an Anthropic-style tool definition."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for spec in capability.inputs:
        properties[spec.name] = _input_schema(spec)
        if spec.required:
            required.append(spec.name)

    return {
        "name": capability.id.replace(".", "_"),
        "description": _tool_description(capability),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def _input_schema(spec: InputSpec) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": JSON_TYPES[spec.type], "description": spec.description}
    if spec.enum:
        schema["enum"] = spec.enum
    if spec.pattern:
        schema["pattern"] = spec.pattern
    if spec.min is not None:
        schema["minimum"] = spec.min
    if spec.max is not None:
        schema["maximum"] = spec.max
    if spec.type == "date":
        schema["format"] = "date"
    if spec.example is not None:
        schema["examples"] = [spec.example]
    return schema


def _tool_description(capability: Capability) -> str:
    lines = [capability.description.strip(), ""]

    effects = capability.side_effects
    idempotent = "idempotent" if effects.idempotent else "not idempotent"
    lines.append(f"Side effects: {effects.cls} ({idempotent}). {effects.description.strip()}")
    if effects.approval == "per_invocation":
        lines.append("Each invocation requires human approval before the committing step.")

    if capability.outputs:
        lines += ["", "Returns on success:"]
        lines += [f"- {_output_line(output)}" for output in capability.outputs]

    if capability.outcomes:
        lines += ["", "Business outcomes (the app completed and declined; not errors):"]
        lines += [f"- {o.code}: {o.description.strip()}" for o in capability.outcomes]

    lines += [
        "",
        "Other results: 'needs_human' when the outcome is unverified (never assume the "
        "action did or did not happen), 'rejected' for invalid input or a missing "
        "approval, and 'failed' for app, automation, infrastructure or policy errors.",
    ]
    return "\n".join(lines)


def _output_line(output: OutputSpec) -> str:
    pattern = f", matching {output.pattern}" if output.pattern else ""
    return f"{output.name} ({output.type}{pattern}): {output.description.strip()}"
