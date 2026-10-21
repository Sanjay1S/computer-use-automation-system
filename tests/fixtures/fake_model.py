"""A scripted stand-in for the discovery model: no network, no key.

It returns real SDK `BetaMessage` objects, so the loop, the driver and the
recorder see exactly the shapes the API sends. A script is a generator: it yields
the blocks of one assistant turn and is sent back the text of the tool results
that turn produced, which is where it finds refs, as a model would.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Generator, Iterator, Mapping
from typing import Any

from anthropic.types.beta import BetaMessage

MODEL = "claude-opus-5-5"

Block = dict[str, Any]
Script = Generator[list[Block], str, None]

_ids = itertools.count(1)


def browser(member: str, /, **input: Any) -> list[Block]:
    return [
        {
            "type": "tool_use",
            "id": f"toolu_{next(_ids):04d}",
            "name": member,
            "input": input,
            "toolset_name": "browser",
        }
    ]


def tool(tool_name: str, /, **input: Any) -> list[Block]:
    return [
        {"type": "tool_use", "id": f"toolu_{next(_ids):04d}", "name": tool_name, "input": input}
    ]


def say(text: str) -> list[Block]:
    return [{"type": "text", "text": text}]


def ref_target(ref: str) -> dict[str, str]:
    return {"type": "ref", "ref": ref}


def message(
    blocks: list[Block],
    *,
    stop_reason: str | None = None,
    model: str = MODEL,
    thinking: str | None = "Looking at the page to decide the next step.",
    usage: Mapping[str, int] | None = None,
    stop_details: Mapping[str, Any] | None = None,
) -> BetaMessage:
    content: list[Block] = []
    if thinking:
        content.append({"type": "thinking", "thinking": thinking, "signature": "sig"})
    content += blocks
    if stop_reason is None:
        stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    return BetaMessage.model_validate(
        {
            "id": f"msg_{next(_ids):04d}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": dict(stop_details) if stop_details else None,
            "usage": dict(usage or {"input_tokens": 2_000, "output_tokens": 150}),
            "content": content,
        }
    )


def result_text(request: Mapping[str, Any]) -> str:
    """The text the model would read from the last user turn."""
    content = request["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        inner = block.get("content") if isinstance(block, dict) else None
        if isinstance(inner, str):
            parts.append(inner)
        elif isinstance(inner, list):
            parts += [b["text"] for b in inner if isinstance(b, dict) and b.get("type") == "text"]
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
    return "\n".join(parts)


class ScriptedModel:
    def __init__(self, script: Script | Callable[[], Script], **turn: Any) -> None:
        self._script = script() if callable(script) else script
        self._started = False
        self._turn = turn
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> BetaMessage:
        self.requests.append(request)
        text = result_text(request)
        try:
            blocks = self._script.send(text) if self._started else next(self._script)
        except StopIteration:
            blocks = say("I am done.")
        self._started = True
        return message(blocks, **self._turn)


class Replies:
    """A fixed list of prepared messages, for loop tests that need exact responses."""

    def __init__(self, messages: list[BetaMessage]) -> None:
        self._messages: Iterator[BetaMessage] = iter(messages)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> BetaMessage:
        self.requests.append(request)
        return next(self._messages)


# --- reading a snapshot as a model would ------------------------------------------------------


def ref_of(snapshot: str, needle: str) -> str:
    for line in snapshot.splitlines():
        if needle in line:
            match = re.search(r"\[ref=(ref_\d+)\]", line)
            if match:
                return match.group(1)
    raise AssertionError(f"no line with {needle!r} in:\n{snapshot}")


def comboboxes(snapshot: str) -> list[tuple[str, list[str]]]:
    """(ref, option labels) for each combobox, in page order."""
    found: list[tuple[str, list[str]]] = []
    for line in snapshot.splitlines():
        box = re.search(r"- combobox(?: \"[^\"]*\")? \[ref=(ref_\d+)\]", line)
        if box:
            found.append((box.group(1), []))
            continue
        option = re.search(r'- option "([^"]*)"', line)
        if option and found:
            found[-1][1].append(option.group(1))
    return found


def numeric_link(snapshot: str, after: str) -> str:
    seen = False
    for line in snapshot.splitlines():
        seen = seen or after in line
        match = re.search(r'- link "(\d+)" \[ref=(ref_\d+)\]', line)
        if seen and match:
            return match.group(2)
    raise AssertionError(f"no numeric link after {after!r}")


# --- the open-account flow ------------------------------------------------------------------


def open_account(
    base_url: str, account_type: str, funding: str, *, via_link: bool = False
) -> Callable[[], Script]:
    """What a careful model does on the open-account form, ref by ref."""

    def script() -> Script:
        if via_link:
            text = yield browser("navigate", url=f"{base_url}/overview.htm")
            text = yield browser("read_page", filter="interactive")
            text = yield browser(
                "left_click", target=ref_target(ref_of(text, 'link "Open New Account"'))
            )
        else:
            text = yield browser("navigate", url=f"{base_url}/openaccount.htm")
        text = yield browser("read_page")
        while len(comboboxes(text)) < 2 or not comboboxes(text)[1][1]:
            text = yield browser("wait", duration=0.3)
            text = yield browser("read_page")
        (type_ref, _), (funding_ref, _) = comboboxes(text)[:2]
        submit = ref_of(text, 'button "Open New Account"')
        text = yield browser("form_input", target=ref_target(type_ref), value=account_type)
        text = yield browser("form_input", target=ref_target(funding_ref), value=funding)
        text = yield browser("left_click", target=ref_target(submit))
        if "did not grant permission" in text:
            yield tool("give_up", reason="A person did not approve opening the account.")
            return
        text = yield browser("read_page")
        heading = ref_of(text, 'heading "Account Opened!"')
        link = numeric_link(text, "Account Opened!")
        text = yield tool(
            "record_output", name="new_account_id", ref=link, description="the new account number"
        )
        yield tool(
            "declare_success",
            summary="Opened the account and read its number.",
            evidence_refs=[heading, link],
        )

    return script
