"""Captures what a person does in the live page.

A page script (`recorder.js`) reports trusted click, change and submit events through
an exposed binding, with a description of the element taken at the moment of the
event. Each report is judged by who held the session at that moment:
- a person (state HUMAN): their action. When the element is still on the page and
  matches the report, it is located so the same synthesizer discovery uses can
  give it verified locators;
- automation (inside one of its own action windows): its own input, already recorded;
- otherwise: `unexpected_input`, a warning, never an action.

Reports are only queued when they arrive (the binding runs inside whatever browser
call is in flight) and are read later on the automation thread. The page can call
the binding itself, so these are reports, not proof: human steps are flagged for
review wherever they are used.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator as PwLocator

from understudy.control.plane import ControlPlane
from understudy.surface.base import ElementFacts, SurfaceError
from understudy.surface.web import WebSurface

BINDING = "__understudyInput"
SCRIPT = (Path(__file__).parent / "recorder.js").read_text(encoding="utf-8")
MAX_QUEUED = 1_000

Attribution = Literal["human", "automation", "unexpected"]
InputKind = Literal["click", "change", "submit"]


@dataclass(frozen=True)
class ElementReport:
    """The element as the page described it. Raw page text: redact before writing."""

    tag: str
    input_type: str | None = None
    id_attr: str | None = None
    name_attr: str | None = None
    role: str | None = None
    aria_label: str | None = None
    text: str | None = None
    value_attr: str | None = None
    label: str | None = None
    href: str | None = None
    autocomplete: str | None = None
    test_id: str | None = None
    path: str | None = None

    def facts(self) -> ElementFacts:
        """Facts from the report alone, for an element that is no longer there."""
        return ElementFacts(
            tag=self.tag,
            role=self.role or _implicit_role(self.tag, self.input_type, self.href),
            name=self.aria_label or self.value_attr or self.text or self.label or None,
            input_type=self.input_type,
            label=self.label,
            test_id=self.test_id,
            id_attr=self.id_attr,
            name_attr=self.name_attr,
            value_attr=self.value_attr,
            text=self.text,
            href=self.href,
            autocomplete=self.autocomplete,
        )


@dataclass(frozen=True)
class HumanInput:
    kind: InputKind
    at: float
    """Wall clock, seconds, when the page saw it."""
    url: str
    """Raw. Redact before writing."""
    in_frame: bool
    element: ElementReport
    attribution: Attribution
    facts: ElementFacts
    """From the live element when it was located, else from the report."""
    locator: PwLocator | None = None
    """The live element, located and matched against the report."""
    value: str | None = None
    """Raw, for a change. Never sent for a password field."""
    option_label: str | None = None
    checked: bool | None = None

    @property
    def member(self) -> str | None:
        """The toolset member this input amounts to; None when it is part of another
        (a click into a field is part of the change that follows)."""
        if self.kind == "change":
            return "form_input"
        # A click on a field or its label is followed by the change that carries the value.
        if self.kind == "click" and not self.facts.is_form_control and self.facts.tag != "label":
            return "left_click"
        return None

    def describe(self) -> str:
        facts = self.facts
        what = facts.role or facts.tag
        named = facts.name or facts.label or facts.id_attr or facts.name_attr
        return f"{what} {named!r}" if named else what


class HumanRecorder:
    def __init__(self, surface: WebSurface, plane: ControlPlane) -> None:
        """Install before the first navigation: the script then runs in every
        document the session opens."""
        self.surface = surface
        self.plane = plane
        self._queued: list[dict[str, Any]] = []
        self.dropped = 0
        page = surface.page
        page.expose_binding(BINDING, self._on_report)
        page.add_init_script(script=SCRIPT)
        with contextlib.suppress(PlaywrightError):  # a closed or crashed page reports nothing
            page.evaluate(SCRIPT)

    def drain(self) -> list[HumanInput]:
        """Every report queued so far, judged and, for a person's, located."""
        queued, self._queued = self._queued, []
        inputs = []
        for raw in queued:
            parsed = self._parse(raw)
            if parsed is not None:
                inputs.append(parsed)
        return inputs

    def flush(self) -> list[HumanInput]:
        """A round trip to the page first, so reports it already sent are queued."""
        with contextlib.suppress(PlaywrightError):
            self.surface.page.evaluate("0")
        return self.drain()

    # --- internals ------------------------------------------------------------------

    def _on_report(self, source: Any, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        if len(self._queued) >= MAX_QUEUED:
            self.dropped += 1
            return
        self._queued.append(payload)

    def _parse(self, raw: dict[str, Any]) -> HumanInput | None:
        kind = raw.get("kind")
        element = raw.get("element")
        ts = raw.get("ts")
        if kind not in ("click", "change", "submit") or not isinstance(element, dict):
            return None
        if not isinstance(ts, int | float):
            return None
        report = ElementReport(
            tag=_str(element.get("tag")) or "unknown",
            input_type=_str(element.get("type")),
            id_attr=_str(element.get("id")),
            name_attr=_str(element.get("name")),
            role=_str(element.get("role")),
            aria_label=_str(element.get("aria_label")),
            text=_str(element.get("text")),
            value_attr=_str(element.get("value_attr")),
            label=_str(element.get("label")),
            href=_str(element.get("href")),
            autocomplete=_str(element.get("autocomplete")),
            test_id=_str(element.get("test_id")),
            path=_str(element.get("path")),
        )
        at = float(ts) / 1000
        in_frame = raw.get("top") is not True
        attribution = self._attribute(at)
        locator, facts = None, report.facts()
        if attribution == "human" and not in_frame:
            located = self._locate(report)
            if located is not None:
                locator, facts = located
        checked = raw.get("checked")
        return HumanInput(
            kind=kind,
            at=at,
            url=_str(raw.get("url")) or "",
            in_frame=in_frame,
            element=report,
            attribution=attribution,
            facts=facts,
            locator=locator,
            value=None if report.input_type == "password" else _str(raw.get("value"), keep=True),
            option_label=_str(raw.get("option_label")),
            checked=checked if isinstance(checked, bool) else None,
        )

    def _attribute(self, at: float) -> Attribution:
        state = self.plane.state_at(at)
        if state == "HUMAN":
            return "human"
        # Automation acts only while it holds the session: once it has stopped for a
        # person, input just after its last action is not its own.
        if state == "AGENT" and self.surface.automation_at(at):
            return "automation"
        return "unexpected"

    def _locate(self, report: ElementReport) -> tuple[PwLocator, ElementFacts] | None:
        """The element the report names, if it is still there and still the same one:
        the path is recomputed by position, so a page that changed since can put a
        different element at it."""
        if not report.path:
            return None
        try:
            element = self.surface.page.locator(f"css={report.path}")
            if element.count() != 1:
                return None
            facts = self.surface.facts(element)
        except (PlaywrightError, SurfaceError):
            return None
        same = (
            facts.tag == report.tag
            and facts.id_attr == report.id_attr
            and facts.name_attr == report.name_attr
            and (facts.input_type or None) == report.input_type
        )
        if same and not (report.id_attr or report.name_attr) and report.text:
            same = " ".join((facts.text or "").split()).startswith(report.text[:40])
        return (element, facts) if same else None


def _str(value: Any, *, keep: bool = False) -> str | None:
    """A string field from the page, or None. Empty strings count as absent unless kept."""
    if not isinstance(value, str):
        return None
    if keep:
        return value
    return value or None


def _implicit_role(tag: str, input_type: str | None, href: str | None) -> str | None:
    if tag == "button" or (tag == "input" and input_type in ("button", "submit", "reset")):
        return "button"
    if tag == "a" and href:
        return "link"
    if tag == "select":
        return "combobox"
    if tag == "input" and input_type in ("checkbox", "radio"):
        return input_type
    if tag in ("input", "textarea"):
        return "textbox"
    return None
