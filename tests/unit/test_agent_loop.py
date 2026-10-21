"""The discovery loop on its own: request shape, append-only history, nudges, stop
reasons, stuck signals and budgets. The browser and the recorder tools are stubs,
so each signal can be produced exactly."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from anthropic import APIConnectionError
from anthropic.types.beta import BetaMessage, BetaToolResultBlockParam, BetaToolUseBlock

from tests.fixtures.fake_model import Replies, browser, message, say, tool
from understudy.discovery.agent import NOT_EXECUTED, Budget, DiscoveryAgent
from understudy.discovery.prompts import CUT_OFF, NUDGE
from understudy.discovery.recorder_tools import Finish


class StubBrowser:
    def __init__(self, react: Callable[[BetaToolUseBlock], tuple[bool, str]] | None = None) -> None:
        """`react(tool_use)` gives (is_error, new state key); default: ok, page changes."""
        self.approval_seconds = 0.0
        self.calls: list[str] = []
        self._state = "s0"
        self._react = react or (lambda t: (False, f"s{len(self.calls)}"))

    @property
    def last_state_key(self) -> str:
        return self._state

    def to_dict(self) -> Any:
        return {"type": "browser_toolset_20260801"}

    def tool_result(self, tool_use: BetaToolUseBlock) -> BetaToolResultBlockParam:
        self.calls.append(tool_use.name)
        error, self._state = self._react(tool_use)
        block: BetaToolResultBlockParam = {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "toolset_name": "browser",
            "content": "failed" if error else "ok",
        }
        if error:
            block["is_error"] = True
        return block


class StubTools:
    def __init__(self) -> None:
        self.handled: list[str] = []

    def definitions(self) -> list[dict[str, Any]]:
        return [{"name": "declare_success"}]

    def handle(self, tool_use: BetaToolUseBlock) -> tuple[BetaToolResultBlockParam, Finish | None]:
        self.handled.append(tool_use.name)
        finish = {
            "declare_success": Finish("succeeded", "declared_success", "done"),
            "give_up": Finish("gave_up", "gave_up", "cannot"),
            "request_human": Finish("stuck", "human_requested", "help"),
        }.get(tool_use.name)
        return {"type": "tool_result", "tool_use_id": tool_use.id, "content": "ok"}, finish


def agent(
    replies: list[BetaMessage],
    browser_stub: StubBrowser | None = None,
    budget: Budget | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[DiscoveryAgent, Replies, StubBrowser, StubTools]:
    model = Replies(replies)
    stub = browser_stub or StubBrowser()
    tools = StubTools()
    loop = DiscoveryAgent(
        model=model,
        browser=stub,
        tools=tools,
        goal_text="Do the task.",
        budget=budget or Budget(),
        **({"clock": clock} if clock else {}),
    )
    return loop, model, stub, tools


def clicks(n: int) -> list[BetaMessage]:
    return [
        message(browser("left_click", target={"type": "ref", "ref": f"ref_{i}"})) for i in range(n)
    ]


def test_a_declared_success_ends_the_run() -> None:
    loop, _, stub, tools = agent([*clicks(2), message(tool("declare_success"))])

    outcome = loop.run_loop()

    assert (outcome.status, outcome.code, outcome.turns) == ("succeeded", "declared_success", 3)
    assert stub.calls == ["left_click", "left_click"] and tools.handled == ["declare_success"]
    assert outcome.models_served == ["claude-opus-5-5"]


def test_the_history_is_append_only_and_tools_never_change() -> None:
    loop, model, _, _ = agent([*clicks(3), message(tool("declare_success"))])
    loop.run_loop()

    for earlier, later in zip(model.requests, model.requests[1:], strict=False):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]
        assert later["tools"] is earlier["tools"]
    first = model.requests[0]
    assert first["messages"] == [{"role": "user", "content": "Do the task."}]
    assert first["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert "budget_tokens" not in str(first["thinking"])


def test_turns_without_a_tool_call_are_nudged_then_stuck() -> None:
    loop, _, stub, _ = agent([message(say("thinking aloud")) for _ in range(3)])

    outcome = loop.run_loop()

    assert (outcome.status, outcome.code, outcome.turns) == ("stuck", "no_tool_call", 3)
    nudges = [m for m in loop.messages if m["role"] == "user" and m["content"] != "Do the task."]
    assert [m["content"][0]["text"] for m in nudges] == [NUDGE, NUDGE]
    assert stub.calls == []


def test_a_nudge_count_resets_after_a_tool_call() -> None:
    replies = [message(say("hm")), message(say("hm")), *clicks(1)]
    replies += [message(say("hm")), message(say("hm")), message(tool("give_up"))]
    loop, _, _, _ = agent(replies)

    assert loop.run_loop().status == "gave_up"


def test_a_refusal_stops_for_a_person_and_runs_nothing() -> None:
    refused = message(
        browser("navigate", url="https://x.test"),
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": None},
    )
    loop, _, stub, _ = agent([refused])

    outcome = loop.run_loop()

    assert (outcome.status, outcome.code) == ("stuck", "refusal")
    assert "cyber" in outcome.reason
    assert stub.calls == []


def test_a_cut_off_turn_answers_its_tool_calls_without_running_them() -> None:
    cut = message(
        browser("left_click", target={"type": "ref", "ref": "ref_1"}), stop_reason="max_tokens"
    )
    loop, _, stub, _ = agent([cut, message(tool("declare_success"))])

    outcome = loop.run_loop()

    assert outcome.status == "succeeded" and stub.calls == []
    reply = loop.messages[2]["content"][0]
    assert reply["is_error"] and reply["content"] == CUT_OFF and reply["toolset_name"] == "browser"


def test_after_a_failed_call_the_rest_of_the_turn_is_not_executed() -> None:
    both = message(
        [
            *browser("left_click", target={"type": "ref", "ref": "ref_1"}),
            *browser("left_click", target={"type": "ref", "ref": "ref_2"}),
        ]
    )
    stub = StubBrowser(react=lambda t: (True, "s0"))
    loop, _, stub, _ = agent([both, message(tool("give_up"))], browser_stub=stub)

    loop.run_loop()

    assert stub.calls == ["left_click"]
    second = loop.messages[2]["content"][1]
    assert second["is_error"] and second["content"] == NOT_EXECUTED
    assert second["toolset_name"] == "browser"


def test_three_failures_in_a_row_are_stuck() -> None:
    stub = StubBrowser(react=lambda t: (True, "s0"))
    loop, _, _, _ = agent([*clicks(3), message(tool("give_up"))], browser_stub=stub)

    outcome = loop.run_loop()

    assert (outcome.status, outcome.code, outcome.turns) == ("stuck", "consecutive_errors", 3)


def test_the_same_action_three_times_with_no_change_is_stuck() -> None:
    same = [
        message(browser("left_click", target={"type": "ref", "ref": "ref_9"})) for _ in range(3)
    ]
    stub = StubBrowser(react=lambda t: (False, "s0"))
    loop, _, _, _ = agent([*same, message(tool("give_up"))], browser_stub=stub)

    outcome = loop.run_loop()

    assert (outcome.status, outcome.code) == ("stuck", "repeated_action")


def test_five_actions_that_change_nothing_are_stuck() -> None:
    stub = StubBrowser(react=lambda t: (False, "s0"))
    loop, _, _, _ = agent([*clicks(5), message(tool("give_up"))], browser_stub=stub)

    outcome = loop.run_loop()

    assert (outcome.status, outcome.code, outcome.turns) == ("stuck", "no_progress", 5)


def test_reading_the_page_between_actions_is_progress_enough() -> None:
    """Looking does not count as an action, and actions that change the page reset it."""
    replies: list[BetaMessage] = []
    for _ in range(4):
        replies += [message(browser("read_page")), clicks(1)[0]]
    loop, _, _, _ = agent([*replies, message(tool("declare_success"))])

    assert loop.run_loop().status == "succeeded"


def test_asking_for_a_person_stops_the_run() -> None:
    loop, _, _, _ = agent([message(tool("request_human", reason="a modal", need="close it"))])

    outcome = loop.run_loop()

    assert (outcome.status, outcome.code) == ("stuck", "human_requested")


def test_most_of_the_turn_budget_without_success_is_stuck() -> None:
    loop, _, _, _ = agent(clicks(10), budget=Budget(max_turns=5))

    outcome = loop.run_loop()

    assert (outcome.status, outcome.code, outcome.turns) == ("stuck", "budget_nearly_spent", 4)


def test_the_cost_budget_counts_real_tokens() -> None:
    expensive = [
        message(
            browser("left_click", target={"type": "ref", "ref": f"ref_{i}"}),
            usage={"input_tokens": 300_000, "output_tokens": 10_000},
        )
        for i in range(5)
    ]
    loop, _, _, _ = agent(expensive, budget=Budget(max_cost_usd=5.0))

    outcome = loop.run_loop()

    # 1.4 USD a turn: past 80% of 5 USD after the third.
    assert (outcome.status, outcome.code, outcome.turns) == ("stuck", "budget_nearly_spent", 3)
    assert outcome.usage.cost_usd == pytest.approx(3 * 1.4)


def test_time_waiting_for_a_person_is_not_counted() -> None:
    now = [0.0]
    stub = StubBrowser()

    def react(tool_use: BetaToolUseBlock) -> tuple[bool, str]:
        now[0] += 100.0  # every call takes 100 s, all of it waiting for approval
        stub.approval_seconds += 100.0
        return False, f"s{len(stub.calls)}"

    stub._react = react
    loop, _, _, _ = agent(
        [*clicks(6), message(tool("declare_success"))],
        browser_stub=stub,
        budget=Budget(max_seconds=300),
        clock=lambda: now[0],
    )

    outcome = loop.run_loop()

    assert outcome.status == "succeeded" and outcome.seconds == 0.0


def test_an_api_error_fails_the_run_as_infrastructure() -> None:
    class Down:
        def __call__(self, request: dict[str, Any]) -> BetaMessage:
            raise APIConnectionError(request=httpx2.Request("POST", "https://api.test/v1/messages"))

    loop = DiscoveryAgent(model=Down(), browser=StubBrowser(), tools=StubTools(), goal_text="x")

    outcome = loop.run_loop()

    assert (outcome.status, outcome.code, outcome.turns) == ("failed", "model_api_error", 0)


def test_a_dead_click_between_reads_is_still_a_repeated_action() -> None:
    """The review round 1 case: looking between attempts must not hide the loop."""
    same = {"type": "ref", "ref": "ref_7"}
    replies: list[BetaMessage] = []
    for _ in range(4):
        replies += [message(browser("left_click", target=same)), message(browser("read_page"))]
    stub = StubBrowser(react=lambda t: (False, "s0"))
    loop, _, _, _ = agent([*replies, message(tool("give_up"))], browser_stub=stub)

    outcome = loop.run_loop()

    assert (outcome.status, outcome.code) == ("stuck", "repeated_action")
    assert stub.calls.count("left_click") == 3
