"""The discovery loop: our own loop around the Messages API, not the SDK's tool runner.

Each turn is budgeted, every browser call goes through the driver (policy, approval,
recording), and the loop can stop for a person at any point. The conversation is
append-only: earlier turns are never edited or pruned, which preserved thinking
requires and which keeps the prompt cache warm.

Stuck signals stop the run for a person rather than letting it wander: the model
asks for one, repeats an action that changes nothing, keeps failing, makes no
progress, nears its budget without succeeding, or is refused by the API's
safeguards. With an `on_stuck` handler the person takes over the live session; when
they hand it back, what they did is told to the model in the same user turn as the
tool results (never a separate, edited or reordered one), and the loop goes on.
Stuck signals are then judged only over what happens after the hand-back.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import anthropic
from anthropic.types.beta import BetaMessage, BetaToolResultBlockParam, BetaToolUseBlock
from pydantic import BaseModel, ConfigDict

from understudy.discovery.prompts import CUT_OFF, NUDGE, SYSTEM_PROMPT
from understudy.discovery.recorder_tools import Finish
from understudy.discovery.trajectory import ACTION_MEMBERS
from understudy.evidence.run import RunContext
from understudy.evidence.usage import Usage

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
NOT_EXECUTED = "Not executed: an earlier action in this turn failed."
REFUSED = "Not executed: this turn was declined."
MAX_NUDGES = 2

Status = Literal["succeeded", "gave_up", "stuck", "failed"]


@dataclass(frozen=True)
class Budget:
    max_turns: int = 40
    max_seconds: float = 600.0
    """Wall clock, not counting time spent waiting for a person's approval."""
    max_cost_usd: float = 5.0
    stuck_at: float = 0.8
    """The share of any budget after which a run that has not succeeded is stuck."""


class ModelTurn(Protocol):
    def __call__(self, request: dict[str, Any]) -> BetaMessage: ...


class AnthropicModel:
    """The real model: one Messages API call per turn."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def __call__(self, request: dict[str, Any]) -> BetaMessage:
        try:
            response: BetaMessage = self._client.beta.messages.create(**request)
        except TypeError as error:
            # The SDK reports missing credentials as a TypeError when the request is built.
            if "authentication" not in str(error):
                raise
            raise anthropic.AnthropicError(
                "no API credentials: set ANTHROPIC_API_KEY (in the environment or .env)"
            ) from None
        return response


class Toolset(Protocol):
    """What the loop needs from the browser driver."""

    approval_seconds: float

    @property
    def last_state_key(self) -> str: ...

    def to_dict(self) -> Any: ...

    def tool_result(self, tool_use: BetaToolUseBlock) -> BetaToolResultBlockParam: ...


class Tools(Protocol):
    """What the loop needs from the recorder tools."""

    def definitions(self) -> list[dict[str, Any]]: ...

    def handle(
        self, tool_use: BetaToolUseBlock
    ) -> tuple[BetaToolResultBlockParam, Finish | None]: ...


@dataclass(frozen=True)
class Handback:
    """What a person's handling of a stuck run came to."""

    resumed: bool
    text: str = ""
    """For the model, when resumed: what the person did and said."""
    resolution: str = ""
    """Why the run stops, when not resumed."""


StuckHandler = Callable[[str, str], Handback]
"""Called with the stuck code and reason."""


class AgentOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Status
    code: str
    reason: str
    turns: int
    seconds: float
    usage: Usage
    models_served: list[str]


@dataclass(frozen=True)
class _Call:
    member: str
    input_key: str
    before: str
    after: str
    error: bool


@dataclass
class _Stop(Exception):
    status: Status
    code: str
    reason: str


@dataclass
class DiscoveryAgent:
    model: ModelTurn
    browser: Toolset
    tools: Tools
    goal_text: str
    run: RunContext | None = None
    on_turn: Callable[[int, str | None], None] | None = None
    on_stuck: StuckHandler | None = None
    budget: Budget = field(default_factory=Budget)
    model_id: str = MODEL
    effort: str = "high"
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": self.goal_text}]
        self.usage = Usage()
        self.turns = 0
        self._nudges = 0
        self._history: list[_Call] = []
        """Every tool call this run. Bounded by the turn budget."""
        self._floor = 0
        """Stuck signals look only at calls from here on: since the last hand-back."""
        self._soft_budget = True
        self.handoff_seconds = 0.0
        self._started = self.clock()
        self._tool_defs = [self.browser.to_dict(), *self.tools.definitions()]

    def run_loop(self) -> AgentOutcome:
        try:
            while True:
                self._turn()
        except _Stop as stop:
            return self._outcome(stop)

    # --- one turn -----------------------------------------------------------------------

    def request(self) -> dict[str, Any]:
        """The full request for the next turn. Only `messages` changes between turns."""
        return {
            "model": self.model_id,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": list(self.messages),
            "tools": self._tool_defs,
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self.effort},
            "cache_control": {"type": "ephemeral"},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def _turn(self) -> None:
        self._check_budget()
        try:
            response = self.model(self.request())
        except anthropic.AnthropicError as error:
            raise _Stop(
                "failed", "model_api_error", f"{type(error).__name__}: {_line(error)}"
            ) from None
        self.turns += 1
        self.usage.add(response.model, response.usage)
        self.messages.append({"role": "assistant", "content": response.content})
        rationale = _rationale(response)
        if self.on_turn:
            self.on_turn(self.turns, rationale)
        tool_uses = [b for b in response.content if isinstance(b, BetaToolUseBlock)]
        self._event(response, rationale, tool_uses)

        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            reason = f"the model declined to continue (category: {category})"
            # Nothing in a refused turn runs; its calls are answered in case a person
            # hands the run back, since every tool_use needs its result.
            unrun = [_result(t, REFUSED, error=True) for t in tool_uses]
            self._end_turn(unrun, _Stop("stuck", "refusal", reason))
            return
        if response.stop_reason == "max_tokens" and tool_uses:
            results = [_result(t, CUT_OFF, error=True) for t in tool_uses]
            self._history.append(_Call("cut_off", "", "", "", True))
            self._end_turn(results, self._stuck())
            return
        if not tool_uses:
            self._nudges += 1
            if self._nudges > MAX_NUDGES:
                self._end_turn(
                    [], _Stop("stuck", "no_tool_call", "the model stopped calling tools")
                )
                return
            self._reply([{"type": "text", "text": NUDGE}])
            return
        self._nudges = 0
        results, finish = self._execute(tool_uses)
        if finish is not None:
            self._end_turn(results, _Stop(finish.status, finish.code, finish.reason))
            return
        self._end_turn(results, self._stuck() or self._nearly_spent())

    def _end_turn(self, content: list[Any], stop: _Stop | None) -> None:
        """The user turn that answers this one, and the stop, if any. A stuck run goes
        to a person first: when they hand it back, their account joins this same turn."""
        if stop is not None and stop.status == "stuck" and self.on_stuck is not None:
            started = self.clock()
            try:
                answer = self.on_stuck(stop.code, stop.reason)
            finally:
                self.handoff_seconds += self.clock() - started
            if answer.resumed:
                self._reply([*content, {"type": "text", "text": answer.text}])
                self._nudges = 0
                self._floor = len(self._history)
                if stop.code == "budget_nearly_spent":
                    self._soft_budget = False  # the hard budget still applies
                return
            stop = _Stop(stop.status, stop.code, f"{stop.reason}; {answer.resolution}")
        if content:
            self._reply(content)
        if stop is not None:
            raise stop

    def _execute(
        self, tool_uses: list[BetaToolUseBlock]
    ) -> tuple[list[BetaToolResultBlockParam], Finish | None]:
        results: list[BetaToolResultBlockParam] = []
        finish: Finish | None = None
        failed = False
        for tool_use in tool_uses:
            if failed or finish is not None:
                results.append(_result(tool_use, NOT_EXECUTED, error=True))
                continue
            before = self.browser.last_state_key
            if tool_use.toolset_name is not None:
                result = self.browser.tool_result(tool_use)
            else:
                result, finish = self.tools.handle(tool_use)
            error = bool(result.get("is_error"))
            self._history.append(
                _Call(
                    tool_use.name,
                    _input_key(tool_use.input),
                    before,
                    self.browser.last_state_key,
                    error,
                )
            )
            failed = failed or error
            results.append(result)
        return results, finish

    def _reply(self, content: list[Any]) -> None:
        self.messages.append({"role": "user", "content": content})

    # --- stopping ----------------------------------------------------------------------------

    def _stuck(self) -> _Stop | None:
        calls = self._history[self._floor :]
        if len(calls) >= 3 and all(c.error for c in calls[-3:]):
            return _Stop("stuck", "consecutive_errors", "three calls in a row failed")
        # Repetition is judged over actions alone, so reading the page in between (the
        # usual rhythm) cannot hide a loop; three identical looks in a row count too.
        actions = [c for c in calls if c.member in ACTION_MEMBERS]
        for recent in (calls[-3:], actions[-3:]):
            if (
                len(recent) == 3
                and len({(c.member, c.input_key) for c in recent}) == 1
                and all(c.before == c.after for c in recent)
            ):
                return _Stop(
                    "stuck", "repeated_action", f"{recent[0].member} three times, nothing changed"
                )
        if len(actions) >= 5 and all(c.before == c.after for c in actions[-5:]):
            return _Stop(
                "stuck", "no_progress", "five actions in a row changed nothing on the page"
            )
        return None

    def _nearly_spent(self) -> _Stop | None:
        if not self._soft_budget:
            return None
        over = self._over(self.budget.stuck_at)
        if not over:
            return None
        return _Stop(
            "stuck",
            "budget_nearly_spent",
            f"{int(self.budget.stuck_at * 100)}% of the {over} budget is spent without success",
        )

    def _check_budget(self) -> None:
        over = self._over(1.0)
        if over:
            raise _Stop("failed", "budget_exhausted", f"the {over} budget is spent")

    def _over(self, share: float) -> str | None:
        budget = self.budget
        spent = (
            self.turns / budget.max_turns,
            self._elapsed() / budget.max_seconds,
            self.usage.cost_usd / budget.max_cost_usd,
        )
        over = [
            name
            for name, used in zip(("turns", "time", "cost"), spent, strict=True)
            if used >= share
        ]
        return over[0] if over else None

    def _elapsed(self) -> float:
        """Time waiting for people (approvals, takeovers) is not the run's to spend."""
        return self.clock() - self._started - self.browser.approval_seconds - self.handoff_seconds

    def _outcome(self, stop: _Stop) -> AgentOutcome:
        return AgentOutcome(
            status=stop.status,
            code=stop.code,
            reason=stop.reason,
            turns=self.turns,
            seconds=round(self._elapsed(), 2),
            usage=self.usage,
            models_served=self.usage.models,
        )

    def _event(
        self, response: BetaMessage, rationale: str | None, tool_uses: list[BetaToolUseBlock]
    ) -> None:
        if self.run is None:
            return
        self.run.event(
            "decide",
            actor="agent",
            member=tool_uses[0].name if tool_uses else None,
            rationale=rationale,
            turn=self.turns,
            stop_reason=response.stop_reason,
            model=response.model,
            cost_usd=round(self.usage.cost_usd, 4),
        )


def _rationale(response: BetaMessage) -> str | None:
    """The turn's summarized thinking: the "why" logged with its action."""
    parts = [b.thinking.strip() for b in response.content if b.type == "thinking" and b.thinking]
    text = " ".join(p for p in parts if p)
    return text[:1_000] or None


def _result(tool_use: BetaToolUseBlock, text: str, *, error: bool) -> BetaToolResultBlockParam:
    block: BetaToolResultBlockParam = {
        "type": "tool_result",
        "tool_use_id": tool_use.id,
        "content": text,
    }
    if tool_use.toolset_name is not None:
        block["toolset_name"] = tool_use.toolset_name
    if error:
        block["is_error"] = True
    return block


def _input_key(value: object) -> str:
    if isinstance(value, Mapping):
        value = {k: v for k, v in value.items() if k != "tab_id"}
    return json.dumps(value, sort_keys=True, default=str)


def _line(error: BaseException) -> str:
    return str(error).strip().split("\n", 1)[0][:300]
