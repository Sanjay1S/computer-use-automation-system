"""The outcome race: which known state the page is in right now, by priority.

One look, no waiting; the engine polls it every 100 ms until a step's timeout. The
order is fixed:
(a) the app profile's global states, then its known interstitials;
(b) the step's own expectations, in declared order;
(c) a native dialog nobody knows (the surface has dismissed it), which goes to a
    human.
Global states come first so that an expired session or an app error is never
mistaken for a step's expected page, which may still be partly on screen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from understudy.artifact.schema import Expectation
from understudy.artifact.signatures import Signature
from understudy.profile import GlobalState, Interstitial


class Prober(Protocol):
    def probe(self, signature: Signature, params: Mapping[str, str] | None = None) -> bool: ...

    @property
    def pending_dialog(self) -> tuple[str, str] | None: ...


@dataclass(frozen=True)
class Hit:
    source: Literal["global", "interstitial", "expect", "dialog"]
    index: int
    """Position in its own list. -1 for an unknown dialog."""
    description: str


def first_hit(
    prober: Prober,
    *,
    global_states: Sequence[GlobalState],
    interstitials: Sequence[Interstitial],
    expectations: Sequence[Expectation],
    params: Mapping[str, str],
) -> Hit | None:
    for index, state in enumerate(global_states):
        if prober.probe(state.when, params):
            return Hit("global", index, state.description)
    for index, interstitial in enumerate(interstitials):
        if prober.probe(interstitial.when, params):
            return Hit("interstitial", index, interstitial.description)
    for index, expectation in enumerate(expectations):
        if prober.probe(expectation.when, params):
            return Hit("expect", index, expectation.description)
    dialog = prober.pending_dialog
    if dialog is not None:
        return Hit("dialog", -1, f"an unexpected {dialog[0]} dialog is open")
    return None
