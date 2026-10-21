"""The recorder tools: how the model tells discovery what it found.

They sit next to the browser toolset as ordinary custom tools with strict schemas.
`record_output` and `declare_success` point at elements by ref, so the compiler
gets verified locators for the output and the success checkpoint rather than the
model's description of them. A success claim is refused until every requested
output has been recorded: there is no vacuous success.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from anthropic.types.beta import BetaToolResultBlockParam, BetaToolUseBlock

from understudy.artifact.schema import Extraction
from understudy.discovery.goal import GoalSpec
from understudy.discovery.trajectory import ElementRecord, Recorder
from understudy.surface.base import SurfaceError
from understudy.surface.web import WebSurface

NAMES = frozenset({"record_output", "declare_success", "request_human", "give_up"})


@dataclass(frozen=True)
class Finish:
    status: Literal["succeeded", "gave_up", "stuck"]
    code: str
    reason: str


class RecorderTools:
    def __init__(self, goal: GoalSpec, recorder: Recorder, surface: WebSurface) -> None:
        self.goal = goal
        self.recorder = recorder
        self.surface = surface

    def definitions(self) -> list[dict[str, Any]]:
        outputs = [o.name for o in self.goal.outputs]
        return [
            _tool(
                "record_output",
                "Record one requested output once its value is on screen. Call it for each "
                "requested output before declare_success, with the ref of the element that "
                "shows the value itself (not a label next to it).",
                {
                    "name": {
                        "type": "string",
                        "enum": outputs or ["none"],
                        "description": "Which requested output this is.",
                    },
                    "ref": {"type": "string", "description": "The element's ref, e.g. ref_12."},
                    "description": {
                        "type": "string",
                        "description": "One line: what the element is.",
                    },
                },
            ),
            _tool(
                "declare_success",
                "Declare the task complete. Call it once, after recording every requested "
                "output, when the page shows that the task succeeded.",
                {
                    "summary": {"type": "string", "description": "One line: what was done."},
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Refs of the elements that prove success, such as a confirmation "
                            "heading and the element showing the result."
                        ),
                    },
                },
            ),
            _tool(
                "request_human",
                "Ask a person to help when you cannot make progress but a person could: an "
                "unexpected dialog, a step you are unsure is safe, a page you cannot read.",
                {
                    "reason": {"type": "string", "description": "What is blocking you."},
                    "need": {"type": "string", "description": "What you need the person to do."},
                },
            ),
            _tool(
                "give_up",
                "Stop because the task cannot be done in this application.",
                {"reason": {"type": "string", "description": "Why it cannot be done."}},
            ),
        ]

    def handle(self, tool_use: BetaToolUseBlock) -> tuple[BetaToolResultBlockParam, Finish | None]:
        args = tool_use.input if isinstance(tool_use.input, dict) else {}
        try:
            if tool_use.name == "record_output":
                return _ok(tool_use, self._record_output(args)), None
            if tool_use.name == "declare_success":
                return _ok(tool_use, "Success recorded."), self._declare_success(args)
            if tool_use.name == "request_human":
                reason = f"{_str(args, 'reason')} (needs: {_str(args, 'need')})"
                return _ok(tool_use, "A person has been asked."), Finish(
                    "stuck", "human_requested", reason
                )
            if tool_use.name == "give_up":
                return _ok(tool_use, "Stopped."), Finish("gave_up", "gave_up", _str(args, "reason"))
        except _Refused as refusal:
            return _error(tool_use, str(refusal)), None
        except SurfaceError as error:
            return _error(tool_use, str(error)), None
        return _error(tool_use, f"unknown tool {tool_use.name!r}"), None

    def _record_output(self, args: dict[str, Any]) -> str:
        name = _str(args, "name")
        spec = next((o for o in self.goal.outputs if o.name == name), None)
        if spec is None:
            raise _Refused(f"{name!r} is not a requested output")
        element = self.surface.element_for_ref(_str(args, "ref"))
        facts = self.surface.facts(element)
        raw = (
            self.surface.current_value(element)
            if facts.is_form_control
            else self.surface.read(element, Extraction(output=name, read="text"))
        )
        sample = raw.strip()
        if not sample:
            raise _Refused("that element shows no value")
        if spec.pattern and re.search(spec.pattern, sample) is None:
            raise _Refused(
                f"that element's text does not match {name}'s pattern {spec.pattern}; choose "
                "the element that shows only the value"
            )
        record = self.recorder.element(
            element, description=f"the element showing {name}", strip=sample
        )
        if record.target is None:
            raise _Refused("no stable way to find that element again was found; choose another")
        self.recorder.output(name, _str(args, "description"), record, sample)
        return f"Recorded {name}."

    def _declare_success(self, args: dict[str, Any]) -> Finish:
        recorded = {o.name for o in self.recorder.trajectory.outputs}
        missing = [o.name for o in self.goal.outputs if o.name not in recorded]
        if missing:
            raise _Refused(f"record_output first for: {', '.join(missing)}")
        refs = args.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            raise _Refused("give at least one evidence ref")
        evidence: list[ElementRecord] = []
        for ref in refs:
            record = self.recorder.element(self.surface.element_for_ref(str(ref)))
            if record.target is None:
                raise _Refused(f"no stable way to find {ref} again was found; choose another")
            evidence.append(record)
        summary = _str(args, "summary")
        self.recorder.success(summary, evidence, self.recorder.capture())
        return Finish("succeeded", "declared_success", summary)


class _Refused(Exception):
    pass


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


def _str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    return value if isinstance(value, str) else ""


def _ok(tool_use: BetaToolUseBlock, text: str) -> BetaToolResultBlockParam:
    return {"type": "tool_result", "tool_use_id": tool_use.id, "content": text}


def _error(tool_use: BetaToolUseBlock, text: str) -> BetaToolResultBlockParam:
    return {"type": "tool_result", "tool_use_id": tool_use.id, "content": text, "is_error": True}
