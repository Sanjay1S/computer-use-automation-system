"""The surface seam: what discovery and replay may ask of a UI, independent of how
the UI is driven. A new surface (legacy IE, Windows UIA, a terminal) is a new
implementation of `Surface` plus new locator kinds; the artifact does not change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from understudy.artifact.schema import Extraction, Fingerprint, Target
from understudy.artifact.signatures import Signature


class SurfaceError(Exception):
    code = "surface_error"
    dispatched = False
    """Set when the error came from the browser call that delivers the input, so the
    input may have reached the page. Replay treats a commit that failed this way as
    having an unknown outcome."""


class StaleRef(SurfaceError):
    """The ref is not in the latest snapshot. Refs are never reused, so a stale one
    can only fail; it can never land on a different element."""

    code = "stale_ref"


class PolicyBlocked(SurfaceError):
    code = "policy_blocked"

    def __init__(self, message: str, rule: str) -> None:
        super().__init__(message)
        self.rule = rule


class ActionBlocked(SurfaceError):
    """Something else on the page receives the input: an overlay, a modal. Replay
    treats this as an unknown state for a human, not as a missing target."""

    code = "blocked"


class OptionAbsent(SurfaceError):
    """The select's options are loaded and none matches. With a structural guard
    this is a business outcome (the account is not the customer's), not a fault."""

    code = "option_absent"

    def __init__(self, message: str, options_loaded: int) -> None:
        super().__init__(message)
        self.options_loaded = options_loaded


class DialogBlocking(SurfaceError):
    """A native dialog is open; the page cannot be read or driven until it is handled."""

    code = "dialog_blocking"


class ActionFailed(SurfaceError):
    code = "action_failed"


class TargetChanged(SurfaceError):
    """The element under the locator stopped matching the target's fingerprint
    between resolution and the action (a re-render swapped it). Nothing was done."""

    code = "target_changed"


class ProbeError(SurfaceError):
    """A signature could not be evaluated: an unsafe param, a malformed selector."""

    code = "probe_error"


class ActionTimeout(SurfaceError):
    """Kept apart from ActionFailed because a timeout on a read or input step is
    transient (retry_transient), where other failures are not."""

    code = "timeout"


class InfraError(SurfaceError):
    """The browser, the network or DNS failed. Never counted as an app or automation failure."""

    code = "infra_error"


@dataclass(frozen=True)
class TableContext:
    table: str
    """A CSS selector unique on the page (or within the target's scope)."""
    column: str | int
    """The column's header text, or its 0-based index when the header is blank
    (legacy grids leave action columns unlabelled)."""
    key_column: str
    row_key: str


@dataclass(frozen=True)
class NearText:
    """Visible text placed next to a control instead of a <label for>. Legacy pages
    (ParaBank included) label fields this way, so it is the only anchor some have."""

    anchor_tag: str
    text: str
    hops: Literal[0, 1]
    """0: the text is the control's preceding sibling. 1: it precedes the control's parent."""
    parent_tag: str | None = None


@dataclass(frozen=True)
class Ancestor:
    tag: str
    nth_of_type: int
    id: str | None


@dataclass(frozen=True)
class ElementFacts:
    """What the surface could establish about one element. Used for fingerprints,
    locator synthesis and policy classification; never for page text in logs."""

    tag: str
    role: str | None = None
    name: str | None = None
    input_type: str | None = None
    label: str | None = None
    test_id: str | None = None
    id_attr: str | None = None
    name_attr: str | None = None
    placeholder: str | None = None
    value_attr: str | None = None
    text: str | None = None
    href: str | None = None
    autocomplete: str | None = None
    form_action: str | None = None
    form_buttons: tuple[str, ...] = ()
    """Labels of the submit-capable controls in the element's form, for judging
    whether a key press there can commit."""
    near: NearText | None = None
    table: TableContext | None = None
    nth_of_type: int = 1
    ancestry: tuple[Ancestor, ...] = ()
    """From the element's parent upwards, at most eight levels."""
    frame_path: tuple[str, ...] = ()

    @property
    def is_password(self) -> bool:
        return self.tag == "input" and self.input_type == "password"

    @property
    def is_form_control(self) -> bool:
        return self.tag in ("select", "textarea") or (
            self.tag == "input"
            and self.input_type not in ("button", "submit", "image", "reset", "hidden")
        )

    @property
    def is_button(self) -> bool:
        return (
            self.role == "button"
            or self.tag == "button"
            or (self.tag == "input" and self.input_type in ("button", "submit", "image"))
        )

    def fingerprint(self) -> Fingerprint:
        return Fingerprint(
            role=self.role,
            name=self.name,
            tag=self.tag,
            input_type=self.input_type,
            label=self.label,
        )


AttemptOutcome = Literal[
    "matched", "no_match", "ambiguous", "hidden", "fingerprint_mismatch", "error"
]


@dataclass(frozen=True)
class LocatorAttempt:
    index: int
    by: str
    outcome: AttemptOutcome
    count: int = 0
    detail: str = ""


@dataclass(frozen=True)
class Resolution[E]:
    """One pass over a target's fallback chain. `element` is set only when a
    candidate matched exactly one visible element whose fingerprint agrees."""

    attempts: list[LocatorAttempt]
    element: E | None = None
    facts: ElementFacts | None = None
    locator_index: int | None = None

    @property
    def ok(self) -> bool:
        return self.element is not None

    @property
    def drifted(self) -> bool:
        """The primary locator failed and a fallback won: proceed, but flag the artifact."""
        return self.locator_index is not None and self.locator_index > 0


@dataclass(frozen=True)
class Observation:
    url: str
    """Raw. Pass it through the redactor before it is written anywhere."""
    title: str
    snapshot: str
    """The accessibility snapshot with model-facing refs (`ref_N`)."""
    status: int | None = None
    dialog: str | None = None
    notes: list[str] = field(default_factory=list)


ActAction = Literal["click", "fill", "select", "check", "press"]


class Surface[E](Protocol):
    def observe(self) -> Observation: ...

    def navigate(self, url: str) -> int | None: ...

    def resolve(
        self, target: Target, params: Mapping[str, str], *, require_visible: bool = True
    ) -> Resolution[E]: ...

    def act(self, action: ActAction, element: E, value: str | None = None) -> None: ...

    def read(self, element: E, extraction: Extraction) -> str: ...

    def probe(self, signature: Signature, params: Mapping[str, str] | None = None) -> bool: ...

    def screenshot(self, path: Path) -> Path: ...
