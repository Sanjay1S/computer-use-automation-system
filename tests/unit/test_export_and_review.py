"""The agent-facing tool definition and the human-facing review summary."""

from __future__ import annotations

import json

from tests.fixtures.example_capability import example_capability
from understudy.artifact.export import tool_definition
from understudy.artifact.hashing import content_hash
from understudy.artifact.review import _ordinal, render
from understudy.artifact.schema import Lifecycle


def test_tool_definition_carries_the_contract_and_not_the_steps() -> None:
    definition = tool_definition(example_capability())
    serialized = json.dumps(definition)

    assert definition["name"] == "parabank_open_account"
    assert "s4_submit" not in serialized
    assert "newAccountId" not in serialized
    assert "locator" not in serialized.lower()


def test_tool_definition_types_the_inputs() -> None:
    schema = tool_definition(example_capability())["input_schema"]

    assert schema["properties"]["account_type"]["enum"] == ["CHECKING", "SAVINGS"]
    assert schema["properties"]["funding_account_id"]["pattern"] == r"^\d+$"
    assert sorted(schema["required"]) == ["account_type", "funding_account_id"]
    assert schema["additionalProperties"] is False


def test_tool_description_warns_the_caller_about_commits_and_outcomes() -> None:
    description = tool_definition(example_capability())["description"]

    assert "not idempotent" in description
    assert "requires human approval" in description
    assert "funding_account_not_found" in description
    assert "needs_human" in description
    assert "new_account_id" in description


def test_optional_inputs_are_not_required() -> None:
    capability = example_capability()
    capability.inputs[0].required = False

    assert tool_definition(capability)["input_schema"]["required"] == ["funding_account_id"]


def test_review_shows_what_approval_turns_on() -> None:
    summary = render(example_capability())

    assert "parabank.open_account@1.0.0" in summary
    assert content_hash(example_capability()) in summary
    assert "! s4_submit" in summary, "the committing step must be marked"
    assert "commit is never retried" in summary
    assert "escalate(commit_outcome_unknown)" in summary
    assert "business(funding_account_not_found)" in summary
    assert "LINT  clean" in summary


def test_review_shows_the_locator_fallback_chain_in_order() -> None:
    summary = render(example_capability())

    assert "ok 1st  role" in summary
    assert "ok 2nd  css" in summary
    assert "nav link" in summary, "the reviewer needs the disambiguation reasoning"


def test_locator_ranks_read_as_ordinals() -> None:
    assert [_ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21)] == [
        "1st",
        "2nd",
        "3rd",
        "4th",
        "11th",
        "12th",
        "13th",
        "21st",
    ]


def test_review_flags_an_artifact_edited_after_approval() -> None:
    capability = example_capability()
    capability.lifecycle = Lifecycle(
        status="approved",
        reviewed_by="a reviewer",
        approved_at="2026-10-20T00:00:00Z",  # type: ignore[arg-type]
        approved_hash="sha256:" + "0" * 64,
    )

    assert "[EDITED SINCE APPROVAL]" in render(capability)


def test_review_reports_lint_findings() -> None:
    capability = example_capability()
    capability.success.negative_control = None

    assert "L001" in render(capability)
