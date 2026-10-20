"""Accessibility snapshots and the refs the model uses to point at elements.

Playwright's AI-mode snapshot tags elements `[ref=e12]`, or `[ref=f3e12]` inside a
frame or after a navigation (each document gets a new prefix). Those refs resolve
only against Playwright's most recent snapshot, and a stale one makes
`get_by_ref` wait out its timeout instead of failing.

The model sees `ref_N` instead. N comes from a run-wide counter and is never
reused, so a ref the model kept from an older snapshot either still names the
same element or is reported stale at once. It can never alias a new element.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from understudy.redaction.detectors import JSESSIONID
from understudy.surface.base import StaleRef

PW_REF = re.compile(r"\[ref=((?:f\d+)?e\d+)\]")
_LINE = re.compile(r'^[ \t]*- (?P<role>[a-z][a-z-]*)(?: "(?P<name>(?:[^"\\]|\\.)*)")?(?P<rest>.*)$')

NAMELESS_ROLES = frozenset({"generic", "none", "presentation", "text", "document"})

_FIELD_VALUE = re.compile(
    r'^(?P<head>[ \t]*- (?:textbox|searchbox)(?: "(?:[^"\\]|\\.)*")?(?: \[[^\]\n]*\])*)'
    r":[ \t]+(?P<value>.*)$",
    re.MULTILINE,
)


def scrub_for_model(snapshot: str, secrets: Iterable[str]) -> str:
    """What the model may see. Discovery shows it page structure, never credentials:
    a password field's value is recognised by value (the field may have no name to
    go by) and the session id is dropped from every link."""
    hidden = {value for value in secrets if value}
    out = JSESSIONID.sub("", snapshot)
    out = _FIELD_VALUE.sub(
        lambda m: (
            m.group("head") + ": [SECRET]" if m.group("value").strip() in hidden else m.group(0)
        ),
        out,
    )
    for value in hidden:
        if len(value) >= 4:
            out = out.replace(value, "[SECRET]")
    return out


@dataclass(frozen=True)
class SnapshotNode:
    model_ref: str
    pw_ref: str
    role: str | None
    name: str | None


def parse_line(line: str) -> tuple[str | None, str | None]:
    """Role and accessible name from one snapshot line, in Playwright's own terms
    (the same computation `get_by_role` matches against)."""
    match = _LINE.match(line)
    if not match:
        return None, None
    role = match.group("role")
    name = match.group("name")
    if name is not None:
        name = name.replace('\\"', '"').replace("\\\\", "\\")
    return (None if role in NAMELESS_ROLES else role), (name or None)


class RefTable:
    def __init__(self) -> None:
        self._counter = 0
        self._model_by_pw: dict[str, str] = {}
        self._current: dict[str, SnapshotNode] = {}
        self.generation = 0

    def rewrite(self, raw: str) -> str:
        """Swap Playwright refs for model refs and make this snapshot the current one."""
        current: dict[str, SnapshotNode] = {}

        def swap(line: str) -> str:
            match = PW_REF.search(line)
            if not match:
                return line
            pw_ref = match.group(1)
            model_ref = self._model_by_pw.get(pw_ref)
            if model_ref is None:
                self._counter += 1
                model_ref = f"ref_{self._counter}"
                self._model_by_pw[pw_ref] = model_ref
            role, name = parse_line(line)
            current[model_ref] = SnapshotNode(model_ref, pw_ref, role, name)
            return line[: match.start()] + f"[ref={model_ref}]" + line[match.end() :]

        rewritten = "\n".join(swap(line) for line in raw.splitlines())
        self._current = current
        self.generation += 1
        return rewritten

    def resolve(self, model_ref: str) -> SnapshotNode:
        node = self._current.get(model_ref)
        if node is None:
            known = model_ref in self._model_by_pw.values()
            why = "not in the latest snapshot" if known else "not a ref this run issued"
            raise StaleRef(f"{model_ref} is {why}; call read_page for fresh refs")
        return node

    @property
    def current(self) -> dict[str, SnapshotNode]:
        return dict(self._current)
