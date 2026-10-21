"""The goal spec, and the rule that discovery's system prompt is not tuned to any app."""

from __future__ import annotations

import re

import pytest
import yaml
from pydantic import ValidationError

from tests.conftest import REPO
from understudy.discovery.goal import GoalSpec, load_goal
from understudy.discovery.prompts import CUT_OFF, NUDGE, SYSTEM_PROMPT, goal_message
from understudy.discovery.recorder_tools import RecorderTools
from understudy.profile import load_profile

GOAL_PATH = REPO / "goals" / "open_account.yaml"


def goal() -> GoalSpec:
    return load_goal(GOAL_PATH)


def test_the_open_account_goal_loads_and_renders() -> None:
    spec = goal()
    values = spec.values()

    assert values == {"account_type": "SAVINGS", "funding_account_id": "13344"}
    assert spec.value_problems(values) == []
    assert spec.render(values) == "Open a new SAVINGS account, funded from account 13344."
    assert spec.start_route == "/openaccount.htm"
    assert [p.input_spec().name for p in spec.params] == ["account_type", "funding_account_id"]


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"account_type": "GOLD"}, "param 'account_type' must be one of"),
        ({"funding_account_id": "12a45"}, r"param 'funding_account_id' does not match"),
        ({"note": "x"}, "unknown param 'note'"),
        ({"account_type": "13344"}, "share a value"),
    ],
)
def test_bad_values_are_named_by_param_not_by_value(
    overrides: dict[str, str], problem: str
) -> None:
    problems = goal().value_problems(goal().values(overrides))

    assert any(problem in p for p in problems), problems
    assert not any("12a45" in p for p in problems)


def test_a_goal_naming_an_undeclared_param_is_invalid() -> None:
    raw = yaml.safe_load(GOAL_PATH.read_text())
    raw["goal"] = "Open a {colour} account"

    with pytest.raises(ValidationError, match="undeclared params: colour"):
        GoalSpec.model_validate(raw)


def test_the_goal_message_carries_the_task_and_the_values() -> None:
    message = goal_message(goal(), goal().values(), "https://bank.test/app")

    assert "Open a new SAVINGS account, funded from account 13344." in message
    assert "Begin at https://bank.test/app/openaccount.htm" in message
    assert "- new_account_id:" in message


# --- prompt leak: the system prompt must stay generic ------------------------------------


def app_specific_strings() -> set[str]:
    """Everything the repo knows that is particular to the app or the goal."""
    profile = load_profile(REPO / "apps" / "parabank" / "profile.yaml")
    policy = yaml.safe_load((REPO / "apps" / "parabank" / "policy.yaml").read_text())
    spec = goal()
    words = {profile.vendor_app, spec.capability, spec.title, "bank", "account", "deposit"}
    words |= {p.name for p in spec.params} | {o.name for o in spec.outputs}
    words |= {v for p in spec.params for v in (p.enum or [])} | {p.demo for p in spec.params}
    routes = [
        *profile.recon.get("routes", {}).values(),
        *policy["routes"]["deny"],
        spec.start_route,
    ]
    words |= {re.sub(r"[/*]", "", r).removesuffix(".htm") for r in routes}
    words |= set(re.findall(r"#(\w+)", " ".join(profile.screenshot_masks)))
    words |= {"fromAccountId", "newAccountId", "openaccount", "Open New Account"}
    return {w for w in words if len(w) >= 4}


@pytest.mark.parametrize(
    "text", [SYSTEM_PROMPT, NUDGE, CUT_OFF], ids=["system", "nudge", "cut_off"]
)
def test_the_fixed_prompts_name_nothing_about_the_app(text: str) -> None:
    leaks = sorted(w for w in app_specific_strings() if w.lower() in text.lower())

    assert leaks == []


def test_the_leak_check_does_flag_app_specific_text() -> None:
    """Negative control: the task message is meant to be app-specific, so the same
    check must find leaks in it, or the tests above would pass vacuously."""
    message = goal_message(goal(), goal().values(), "https://bank.test/app").lower()

    leaks = {w for w in app_specific_strings() if w.lower() in message}

    assert {"account", "openaccount", "savings", "13344", "new_account_id"} <= {
        w.lower() for w in leaks
    }


def test_the_recorder_tool_descriptions_are_generic_too() -> None:
    tools = RecorderTools(goal(), recorder=None, surface=None).definitions()  # type: ignore[arg-type]
    for definition in tools:
        definition["input_schema"]["properties"].get("name", {}).pop("enum", None)
    text = repr(tools).lower()

    leaks = sorted(w for w in app_specific_strings() if w.lower() in text)

    assert leaks == []
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in tools)
