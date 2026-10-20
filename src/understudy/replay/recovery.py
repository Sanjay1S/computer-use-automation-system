"""The recovery registry: three fixed handlers, the ones the artifact allows, and a
per-run budget. Recovery is never improvised: a condition with no handler here is
a failure or an escalation, not something to try a way around."""

from __future__ import annotations

from collections.abc import Sequence

from understudy.artifact.schema import RecoveryPolicy, RetryPolicy, Step
from understudy.replay.result import RecoveryEvent

DEFAULT_BACKOFF_MS = (500, 1_000, 2_000)


class RecoveryLedger:
    def __init__(self, policy: RecoveryPolicy) -> None:
        self.allowed = frozenset(policy.allowed)
        self.remaining = policy.budget
        self.events: list[RecoveryEvent] = []

    def refusal(self, handler: str) -> str | None:
        """Why the handler may not run now, or None if it may."""
        if handler not in self.allowed:
            return f"the artifact does not allow {handler}"
        if self.remaining < 1:
            return "the recovery budget for this run is spent"
        return None

    def spend(self, event: RecoveryEvent, *, budgeted: bool = True) -> RecoveryEvent:
        """`budgeted=False` records a recovery that a standing rule already made (a known
        dialog, answered by the surface on arrival): it is reported, but it is not a
        try the engine chose, so it does not use up the budget."""
        if budgeted and self.refusal(event.handler) is not None:
            raise RuntimeError(f"{event.handler} is not available")
        if budgeted:
            self.remaining -= 1
        self.events.append(event)
        return event


def backoff_ms(retry: RetryPolicy, attempt: int) -> int:
    """The step's own backoff if it declares one (its last entry repeats), else
    0.5 s, 1 s, 2 s."""
    table = retry.backoff_ms or list(DEFAULT_BACKOFF_MS)
    return table[min(attempt, len(table) - 1)]


def reentry_index(steps: Sequence[Step], current: int) -> int | None:
    """The latest safe restart point at or before the current step."""
    for index in range(min(current, len(steps) - 1), -1, -1):
        if steps[index].reentry_point:
            return index
    return None
