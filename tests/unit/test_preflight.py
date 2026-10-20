"""Preflight: what is refused before replay touches the UI, and why."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.conftest import FIXTURE_TENANT, PARABANK_POLICY, REPO
from tests.fixtures.example_capability import example_capability
from understudy.artifact import hashing
from understudy.artifact.schema import Capability, InputSpec, Lifecycle, SecretValue
from understudy.artifact.signatures import AllCondition, SelectedCondition
from understudy.policy.gate import PolicyGate
from understudy.profile import load_profile
from understudy.replay.preflight import (
    CommitApproval,
    check_inputs,
    expand_url,
    grant,
    inputs_hash,
    preflight,
)
from understudy.replay.result import Rejection

GATE = PolicyGate.load(PARABANK_POLICY, FIXTURE_TENANT)
PROFILE = load_profile(REPO / "apps" / "parabank" / "profile.yaml")
INPUTS = {"account_type": "SAVINGS", "funding_account_id": "12345"}


def run(
    capability: Capability,
    inputs: dict[str, str] = INPUTS,
    *,
    approval: CommitApproval | None = None,
    allow_draft: bool = True,
) -> Rejection | None:
    return preflight(
        capability,
        inputs,
        gate=GATE,
        profile=PROFILE,
        approval=approval or grant(capability, inputs, "fixture", "t"),
        allow_draft=allow_draft,
    )


def approved(capability: Capability) -> Capability:
    capability.lifecycle = Lifecycle(
        status="approved",
        reviewed_by="r",
        approved_at=datetime.now(UTC),
        approved_hash=hashing.content_hash(capability),
    )
    return capability


def test_the_example_passes_preflight() -> None:
    assert run(example_capability()) is None


def test_an_approved_artifact_runs_without_allow_draft() -> None:
    assert run(approved(example_capability()), allow_draft=False) is None


def test_an_artifact_edited_after_approval_is_not_approved() -> None:
    capability = approved(example_capability())
    capability.title = "Changed after review"
    rejection = run(capability, allow_draft=False)
    assert rejection is not None
    assert rejection.code == "not_approved"
    assert "hash" in rejection.problems[0]


def test_a_deprecated_artifact_is_refused_even_with_allow_draft() -> None:
    capability = example_capability()
    capability.lifecycle = Lifecycle(status="deprecated")
    rejection = run(capability)
    assert rejection is not None and rejection.code == "not_approved"


@pytest.mark.parametrize(
    ("spec", "value", "problem"),
    [
        (InputSpec(name="n", type="integer", description="d"), "1.5", "must be an integer"),
        (InputSpec(name="n", type="integer", description="d", min=1), "0", "at least 1"),
        (InputSpec(name="n", type="decimal", description="d", max=10), "10.01", "at most 10"),
        (InputSpec(name="n", type="decimal", description="d"), "NaN", "finite"),
        (InputSpec(name="n", type="decimal", description="d"), "ten", "decimal"),
        (InputSpec(name="n", type="boolean", description="d"), "yes", "true or false"),
        (InputSpec(name="n", type="date", description="d"), "20/10/2026", "ISO date"),
        (InputSpec(name="n", type="enum", enum=["A"], description="d"), "B", "one of A"),
        (InputSpec(name="n", type="string", pattern=r"^\d+$", description="d"), "1a", "match"),
    ],
)
def test_each_input_type_is_checked(spec: InputSpec, value: str, problem: str) -> None:
    capability = example_capability()
    capability.inputs.append(spec.model_copy(update={"required": False}))
    problems = check_inputs(capability, {**INPUTS, "n": value})
    assert len(problems) == 1 and problem in problems[0], problems
    assert value not in problems[0]


@pytest.mark.parametrize(
    ("spec", "value"),
    [
        (InputSpec(name="n", type="integer", description="d", min=1, max=3), "3"),
        (InputSpec(name="n", type="decimal", description="d", min=0.5), "0.50"),
        (InputSpec(name="n", type="boolean", description="d"), "False"),
        (InputSpec(name="n", type="date", description="d"), "2026-10-20"),
    ],
)
def test_valid_values_pass(spec: InputSpec, value: str) -> None:
    capability = example_capability()
    capability.inputs.append(spec.model_copy(update={"required": False}))
    assert check_inputs(capability, {**INPUTS, "n": value}) == []


def test_an_optional_input_a_step_needs_must_be_given() -> None:
    capability = example_capability()
    capability.inputs[1].required = False
    problems = check_inputs(capability, {"account_type": "SAVINGS"})
    assert problems == [
        "input 'funding_account_id' is optional but step 's3_funding_account' needs it"
    ]


def test_a_secret_ref_is_refused_until_there_is_a_secret_store() -> None:
    capability = example_capability()
    capability.steps[1].value = SecretValue(secret_ref="token")
    capability.requires.secrets = ["token"]
    rejection = run(capability)
    assert rejection is not None and rejection.code == "invalid_artifact"


def test_a_capability_for_another_session_is_refused() -> None:
    capability = example_capability()
    capability.requires.session = "parabank.admin"
    rejection = run(capability)
    assert rejection is not None and rejection.code == "invalid_artifact"


def test_fault_injection_is_refused_on_a_tenant_that_is_not_a_sandbox() -> None:
    production = FIXTURE_TENANT.model_copy(update={"sandbox": False})
    capability = example_capability()
    rejection = preflight(
        capability,
        INPUTS,
        gate=PolicyGate.load(PARABANK_POLICY, production),
        profile=PROFILE,
        approval=grant(capability, INPUTS, "fixture", "t"),
        allow_draft=True,
        fault_injection=True,
    )
    assert rejection is not None and rejection.code == "policy"
    assert "sandbox" in rejection.problems[0]


@pytest.mark.parametrize("field", ["capability", "version", "artifact_hash", "tenant"])
def test_an_approval_binds_to_every_field(field: str) -> None:
    capability = example_capability()
    approval = grant(capability, INPUTS, "fixture", "t").model_copy(update={field: "other"})
    rejection = run(capability, approval=approval)
    assert rejection is not None and rejection.code == "approval_required"
    assert rejection.problems == [f"the approval was granted for a different {field}"]


def test_the_inputs_hash_ignores_order_but_not_values() -> None:
    assert inputs_hash({"a": "1", "b": "2"}) == inputs_hash({"b": "2", "a": "1"})
    assert inputs_hash({"a": "1"}) != inputs_hash({"a": "2"})


def test_url_placeholders_take_the_base_url_and_encoded_inputs() -> None:
    url = expand_url("{base_url}/activity.htm?id={id}", "https://x.test/app", {"id": "1 2&3"})
    assert url == "https://x.test/app/activity.htm?id=1%202%263"
    with pytest.raises(ValueError, match="no input"):
        expand_url("{base_url}/{missing}", "https://x.test", {})


def test_an_optional_input_the_success_checkpoint_needs_must_be_given() -> None:
    capability = example_capability()
    capability.inputs[0].required = False
    capability.success.signature = AllCondition(
        all=[
            capability.success.signature,
            SelectedCondition(target="select#type", param="account_type"),
        ]
    )
    problems = check_inputs(capability, {"funding_account_id": "12345"})
    assert "input 'account_type' is optional but the success checkpoint needs it" in problems


def test_decimal_outputs_must_be_finite() -> None:
    from understudy.artifact.schema import OutputSpec
    from understudy.replay.engine import _typed

    spec = OutputSpec(name="balance", type="decimal", description="d", source_step="s")
    assert _typed(spec, "12.50") == "12.50"
    assert _typed(spec, "NaN") is None
    assert _typed(spec, "Infinity") is None
