"""Every lint rule, positive and negative.

Each test starts from the valid example and breaks exactly one thing, so a
finding can only come from the mutation under test.
"""

from __future__ import annotations

from collections.abc import Callable

from tests.fixtures.example_capability import example_capability
from understudy.artifact.lint import Finding, LintReport, PolicyView, lint
from understudy.artifact.schema import (
    Capability,
    Checkpoint,
    Disposition,
    Expectation,
    Extraction,
    InputSpec,
    LiteralValue,
    Locator,
    OutcomeRef,
    OutputSpec,
    ParamValue,
    RetryPolicy,
    Verification,
)
from understudy.artifact.signatures import (
    SelectedCondition,
    UrlCondition,
    VisibleCondition,
)


def test_the_example_artifact_is_clean() -> None:
    report = lint(example_capability())

    assert report.ok, [str(f) for f in report.errors]
    assert report.warnings == []


def test_l001_a_success_checkpoint_without_an_element_condition() -> None:
    capability = example_capability()
    capability.success = Checkpoint(
        description="we are on the right page",
        signature=UrlCondition(path_regex=r"/openaccount\.htm"),
        negative_control=VisibleCondition(role="heading", name="Open New Account"),
    )

    assert "L001" in _codes(capability)


def test_l001_a_success_checkpoint_without_a_negative_control() -> None:
    capability = example_capability()
    capability.success.negative_control = None

    assert "L001" in _codes(capability)


def test_l002_a_sensitive_literal_is_refused() -> None:
    capability = example_capability()
    capability.steps[2].value = LiteralValue(literal="123456789012")

    assert "L002" in _codes(capability)


def test_l002_a_secret_may_not_travel_as_an_input_param() -> None:
    capability = example_capability()
    capability.inputs[1].sensitivity = "secret"

    assert "L002" in _codes(capability)


def test_l002_a_public_literal_is_fine() -> None:
    capability = example_capability()
    capability.inputs.append(
        InputSpec(name="note", type="string", description="unused placeholder")
    )
    capability.steps[1].value = LiteralValue(literal="CHECKING")
    # account_type is now unbound, which is L004's business, not L002's.

    assert "L002" not in _codes(capability)


def test_l003_a_commit_step_may_not_retry() -> None:
    capability = example_capability()
    capability.steps[3].retry = RetryPolicy(max_attempts=2, backoff_ms=[500, 1000])

    assert "L003" in _codes(capability)


def test_l003_a_read_step_may_retry() -> None:
    capability = example_capability()
    capability.steps[0].retry = RetryPolicy(max_attempts=3, backoff_ms=[500, 1000, 2000])

    assert "L003" not in _codes(capability)


def test_l004_a_declared_param_that_is_never_bound() -> None:
    capability = example_capability()
    capability.inputs.append(InputSpec(name="nickname", type="string", description="never used"))

    assert "L004" in _codes(capability)


def test_l004_a_bound_param_that_was_never_declared() -> None:
    capability = example_capability()
    capability.steps[1].value = ParamValue(param="undeclared_type")

    assert "L004" in _codes(capability)


def test_l004_counts_params_used_only_inside_a_templated_locator() -> None:
    capability = example_capability()
    capability.steps[1].value = LiteralValue(literal="CHECKING")
    locator = capability.steps[1].target.locators[0]  # type: ignore[union-attr]
    locator.value = "Type {account_type}"
    locator.templated = True

    assert "L004" not in _codes(capability)


def test_l005_a_target_with_no_verified_locator() -> None:
    capability = example_capability()
    for locator in capability.steps[3].target.locators:  # type: ignore[union-attr]
        locator.verified = None

    assert "L005" in _codes(capability)


def test_l005_a_locator_verified_as_non_unique_does_not_count() -> None:
    capability = example_capability()
    for locator in capability.steps[3].target.locators:  # type: ignore[union-attr]
        locator.verified = Verification(unique=False, same_element=True, run_id="r")

    assert "L005" in _codes(capability)


def test_l006_warns_when_the_first_locator_is_low_stability() -> None:
    """Only the head of the chain is lowered: replay tries locators in declared
    order, so a stable candidate behind a low one does not rescue it."""
    capability = example_capability()
    locators = capability.steps[3].target.locators  # type: ignore[union-attr]
    locators[0].stability = "low"
    locators[1].stability = "high"

    report = lint(capability)

    assert "L006" in report.codes()
    assert report.ok, "L006 is a warning; it must not block approval"


def test_l007_flags_a_step_outside_the_current_policy() -> None:
    class DenyOpenAccount(PolicyView):
        def allows_route(self, route: str) -> bool:
            return route != "/openaccount.htm"

    report = lint(example_capability(), policy=DenyOpenAccount())

    assert "L007" in report.codes()


def test_l007_flags_a_locator_outside_the_current_policy() -> None:
    class DenySubmitButton(PolicyView):
        def allows_locator_value(self, value: str) -> bool:
            return "Open New Account" not in value

    report = lint(example_capability(), policy=DenySubmitButton())

    assert "L007" in report.codes()


def test_l007_is_silent_without_a_policy() -> None:
    assert "L007" not in _codes(example_capability())


def test_l008_an_output_nothing_extracts() -> None:
    capability = example_capability()
    capability.outputs.append(
        OutputSpec(
            name="balance",
            type="decimal",
            description="never extracted",
            source_step="s5_read_new_id",
        )
    )

    assert "L008" in _codes(capability)


def test_l008_a_source_step_that_does_not_match_the_extracting_step() -> None:
    capability = example_capability()
    capability.outputs[0].source_step = "s1_open_form"

    assert "L008" in _codes(capability)


def test_l008_an_extract_for_an_undeclared_output() -> None:
    capability = example_capability()
    capability.steps[4].extract = Extraction(output="not_declared", read="text")

    assert "L008" in _codes(capability)


def test_l009_a_commit_step_under_a_read_capability() -> None:
    capability = example_capability()
    capability.side_effects.cls = "read"

    assert "L009" in _codes(capability)


def test_l009_a_commit_capability_without_approval() -> None:
    capability = example_capability()
    capability.side_effects.approval = "none"

    assert "L009" in _codes(capability)


def test_l010_a_baked_in_host() -> None:
    capability = example_capability()
    capability.steps[0].url = "http://localhost:8089/parabank/openaccount.htm"

    assert "L010" in _codes(capability)


def test_l010_a_baked_in_session_id() -> None:
    capability = example_capability()
    capability.steps[0].url = "{base_url}/openaccount.htm;jsessionid=9A8B7C6D"

    assert "L010" in _codes(capability)


def test_severities_come_from_the_rules_not_from_a_hardcoded_list() -> None:
    """Derived from real findings: if a rule's severity were downgraded, this fails."""
    severities: dict[str, set[str]] = {}
    for mutate in _ALL_MUTATIONS:
        capability = example_capability()
        mutate(capability)
        for finding in lint(capability).findings:
            severities.setdefault(finding.code, set()).add(finding.severity)

    assert severities.pop("L006", {"warning"}) == {"warning"}
    assert severities, "the mutations must actually produce findings"
    for code, found in sorted(severities.items()):
        assert found == {"error"}, f"{code} should block approval, got {found}"


def test_report_separates_errors_from_warnings() -> None:
    report = LintReport(
        findings=[
            Finding("L001", "error", "bad", "success"),
            Finding("L006", "warning", "meh", "steps.s1"),
        ]
    )

    assert not report.ok
    assert len(report.errors) == 1
    assert len(report.warnings) == 1


def _codes(capability: object) -> set[str]:
    return lint(capability).codes()  # type: ignore[arg-type]


def _warning_only_codes() -> set[str]:
    return {"L006"}


def test_l003_a_commit_may_not_assume_success_when_nothing_matches() -> None:
    """The escalate-on-unknown default is a floor, not just a default: an artifact
    must not be able to declare that an unverified commit succeeded."""
    capability = example_capability()
    capability.steps[3].on_no_match = Disposition(do="succeed")

    assert "L003" in _codes(capability)


def test_l003_a_commit_may_still_declare_an_explicit_escalation() -> None:
    capability = example_capability()
    capability.steps[3].on_no_match = Disposition(do="escalate", code="custom_unknown")

    assert "L003" not in _codes(capability)


def test_l004_walks_the_negative_control_too() -> None:
    capability = example_capability()
    capability.success.negative_control = SelectedCondition(target="#x", param="ghost_param")

    assert "L004" in _codes(capability)


def test_l007_checks_the_scope_locator_as_well() -> None:
    class DenyScope(PolicyView):
        def allows_locator_value(self, value: str) -> bool:
            return "scoped-form" not in value

    capability = example_capability()
    capability.steps[3].target.scope = _scope_locator("form#scoped-form")  # type: ignore[union-attr]

    assert "L007" in lint(capability, policy=DenyScope()).codes()


def test_l008_two_steps_extracting_one_output() -> None:
    """Order-independent: whichever step is declared first, the clash is reported."""
    capability = example_capability()
    duplicate = capability.steps[4].model_copy(deep=True)
    duplicate.id = "s6_read_again"
    capability.steps.insert(4, duplicate)

    assert "L008" in _codes(capability)


def test_l010_catches_a_session_id_hidden_in_a_locator() -> None:
    capability = example_capability()
    capability.steps[3].target.scope = _scope_locator(  # type: ignore[union-attr]
        "form[action*=';jsessionid=9A8B7C6D']"
    )

    assert "L010" in _codes(capability)


def test_l010_catches_a_host_hidden_in_a_locator() -> None:
    capability = example_capability()
    capability.steps[3].target.locators[0] = _scope_locator(  # type: ignore[union-attr]
        "a[href^='http://localhost:8089/parabank']"
    )

    assert "L010" in _codes(capability)


def test_l011_an_outcome_detected_by_a_step_that_does_not_exist() -> None:
    capability = example_capability()
    capability.outcomes[0].detected_by = "s99_nonexistent"

    assert "L011" in _codes(capability)


def test_l011_on_target_absent_naming_an_undeclared_outcome() -> None:
    capability = example_capability()
    capability.steps[2].on_target_absent = OutcomeRef(outcome="no_such_outcome")

    assert "L011" in _codes(capability)


def test_l011_an_expect_raising_an_undeclared_business_outcome() -> None:
    capability = example_capability()
    capability.steps[2].expect.append(
        Expectation(
            description="ghost",
            when=VisibleCondition(text="whatever"),
            then=Disposition(do="business", code="ghost_outcome"),
        )
    )

    assert "L011" in _codes(capability)


def test_l002_does_not_fire_on_ordinary_words_containing_a_keyword() -> None:
    """L002 is an error, so a false positive blocks a legitimate artifact.
    'Secretary' and 'tokenize' are plausible labels in a banking UI; only a
    whole-token match counts, so neither trips the 'secret'/'token' keywords.
    A literal that really is a standalone 'password' or 'secret' still does."""
    capability = example_capability()

    for label in ("Secretary", "tokenize", "My Garden Tour", "Secretaries Office"):
        capability.steps[1].value = LiteralValue(literal=label)
        assert "L002" not in _codes(capability), label


def test_l002_still_catches_a_real_secret_token() -> None:
    capability = example_capability()

    for leaked in ("session token", "jsessionid=9A8B7C6D", "4111 1111 1111 1111"):
        capability.steps[1].value = LiteralValue(literal=leaked)
        assert "L002" in _codes(capability), leaked


RUN = {"13344": "input funding_account_id", "13566": "output new_account_id"}


def test_l012_a_run_value_in_the_success_description() -> None:
    capability = example_capability()
    assert "L012" not in lint(capability, run_values=RUN).codes()

    capability.success.description = "Opened account 13566 from 13344."
    findings = [f for f in lint(capability, run_values=RUN).findings if f.code == "L012"]

    assert {(f.where, f.severity) for f in findings} == {("success.description", "error")}
    assert len(findings) == 2
    assert all("13344" not in f.message and "13566" not in f.message for f in findings)


def test_l012_looks_in_every_free_text_field() -> None:
    capability = example_capability()
    capability.steps[2].intent = "Choose 13344"
    capability.steps[2].target.locators[0].note = "read 13566 here"  # type: ignore[union-attr]

    wheres = {f.where for f in lint(capability, run_values=RUN).findings if f.code == "L012"}

    assert wheres == {"steps[2].intent", "steps[2].target.locators[0].note"}


def test_l012_needs_a_whole_value_and_is_off_without_run_values() -> None:
    capability = example_capability()
    capability.success.description = "Opened account 913344, not the funding one."

    assert "L012" not in lint(capability, run_values=RUN).codes()  # 13344 inside 913344
    capability.success.description = "Opened 13344."
    assert "L012" not in lint(capability).codes()  # no run values: lint at load time


def _scope_locator(value: str) -> Locator:
    return Locator(
        by="css",
        value=value,
        stability="medium",
        verified=Verification(unique=True, same_element=True, run_id="t"),
        note="test locator",
    )


_ALL_MUTATIONS: list[Callable[[Capability], None]] = [
    lambda c: setattr(c.success, "negative_control", None),
    lambda c: setattr(c.steps[1], "value", LiteralValue(literal="jsessionid=ABC123")),
    lambda c: setattr(c.steps[3], "retry", RetryPolicy(max_attempts=2, backoff_ms=[1, 2])),
    lambda c: c.inputs.append(InputSpec(name="unused", type="string", description="x")),
    lambda c: [setattr(loc, "verified", None) for loc in c.steps[3].target.locators],  # type: ignore[union-attr]
    lambda c: setattr(c.steps[3].target.locators[0], "stability", "low"),  # type: ignore[union-attr]
    lambda c: setattr(c.outputs[0], "source_step", "s1_open_form"),
    lambda c: setattr(c.side_effects, "approval", "none"),
    lambda c: setattr(c.steps[0], "url", "http://localhost:8089/openaccount.htm"),
    lambda c: setattr(c.outcomes[0], "detected_by", "s99_missing"),
]


def test_l013_a_fallback_that_ignores_the_input_beside_one_that_uses_it() -> None:
    capability = example_capability()
    step = next(s for s in capability.steps if s.target is not None)
    assert step.target is not None
    keyed = Locator(
        by="table_cell",
        value={
            "table": "table#t",
            "column": "Balance",
            "key_column": "Account",
            "row_key": "{funding_account_id}",
        },
        stability="medium",
        note="by the row key",
        templated=True,
        verified=Verification(unique=True, same_element=True, run_id="t"),
    )
    positional = Locator(
        by="css",
        value="#t > tbody > tr:nth-of-type(1) > td:nth-of-type(2)",
        stability="low",
        note="by position",
        verified=Verification(unique=True, same_element=True, run_id="t"),
    )
    step.target.locators = [keyed, positional]

    findings = [f for f in lint(capability).findings if f.code == "L013"]
    assert [(f.severity, f.where) for f in findings] == [
        ("error", f"steps.{step.id}.target.locators[1]")
    ]

    step.target.locators = [keyed]
    assert "L013" not in _codes(capability)
