"""Live check of the discovery request shape against the real Messages API.

Skipped by default (`live` marker) and without credentials. Two turns of the real
loop with a stub browser: the toolset entry, the strict recorder tools, adaptive
thinking, effort, automatic caching and server-side fallbacks must all be accepted,
and the assistant turn (thinking blocks, a toolset tool_use) must go back unchanged
on the second request. Costs a few cents. Run: `make test-live` with
ANTHROPIC_API_KEY in .env.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic
import pytest
from anthropic.types.beta import BetaToolResultBlockParam, BetaToolUseBlock

from tests.conftest import REPO
from understudy.discovery.agent import MODEL, SYSTEM_PROMPT, AnthropicModel, Budget, DiscoveryAgent
from understudy.discovery.driver import IMPLEMENTED, member_configs
from understudy.discovery.goal import load_goal
from understudy.discovery.recorder_tools import RecorderTools

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
        reason="needs ANTHROPIC_API_KEY",
    ),
]

TOOLSET: dict[str, Any] = {
    "type": "browser_toolset_20260801",
    "configs": member_configs(IMPLEMENTED),
}


class StubBrowser:
    approval_seconds = 0.0
    last_state_key = "unchanging"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def to_dict(self) -> Any:
        return TOOLSET

    def tool_result(self, tool_use: BetaToolUseBlock) -> BetaToolResultBlockParam:
        self.calls.append(tool_use.name)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "toolset_name": "browser",
            "content": [{"type": "text", "text": '- heading "Wiring check" [level=1] [ref=ref_1]'}],
        }


def test_the_api_accepts_the_discovery_request_and_its_follow_up() -> None:
    client = anthropic.Anthropic()
    tools = RecorderTools(load_goal(REPO / "goals" / "open_account.yaml"), None, None)  # type: ignore[arg-type]
    counted = client.messages.count_tokens(
        model=MODEL,
        system=SYSTEM_PROMPT,
        tools=[TOOLSET, *tools.definitions()],  # type: ignore[list-item]
        messages=[{"role": "user", "content": "hi"}],
    )
    assert counted.input_tokens > 1_000  # the toolset's definitions are in there

    browser = StubBrowser()
    loop = DiscoveryAgent(
        model=AnthropicModel(client),
        browser=browser,
        tools=tools,
        goal_text=(
            "This is a check of the tool wiring, not a real task. Call read_page once, "
            "then call give_up with the reason 'wiring check'."
        ),
        budget=Budget(max_turns=4, max_cost_usd=0.50),
    )

    outcome = loop.run_loop()

    assert (outcome.status, outcome.code) == ("gave_up", "gave_up"), outcome
    assert browser.calls == ["read_page"]
    assert outcome.turns == 2 and outcome.usage.cost_usd < 0.50
