"""Schema validation, round-tripping and hashing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from tests.fixtures.example_capability import example_capability
from understudy.artifact.export import capability_json_schema
from understudy.artifact.hashing import canonical_json, content_hash, load, save
from understudy.artifact.schema import (
    Capability,
    Disposition,
    Extraction,
    Fingerprint,
    InputSpec,
    Lifecycle,
    Locator,
    Step,
    Target,
)
from understudy.artifact.signatures import (
    AbsentCondition,
    AllCondition,
    HttpStatusCondition,
    NotCondition,
    SelectedCondition,
    Signature,
    UrlCondition,
    VisibleCondition,
    has_element_condition,
    params_used,
)


def test_example_round_trips_through_json(tmp_path: Path) -> None:
    original = example_capability()
    path = tmp_path / "cap.json"
    save(original, path)

    assert load(path) == original


def test_canonical_json_is_stable_under_key_order() -> None:
    capability = example_capability()
    reparsed = Capability.model_validate(json.loads(canonical_json(capability)))

    assert canonical_json(reparsed) == canonical_json(capability)
    assert content_hash(reparsed) == content_hash(capability)


def test_hash_ignores_lifecycle_so_approval_does_not_invalidate_itself() -> None:
    draft = example_capability()
    before = content_hash(draft)

    draft.lifecycle = Lifecycle(
        status="approved",
        reviewed_by="a reviewer",
        approved_at="2026-10-20T00:00:00Z",  # type: ignore[arg-type]
        approved_hash=before,
    )

    assert content_hash(draft) == before


def test_hash_is_stable_across_processes() -> None:
    """Approval binds to the hash, so it must survive a restart: no set iteration,
    no PYTHONHASHSEED sensitivity, no dict ordering luck."""
    script = (
        "import sys; sys.path.insert(0, '.');"
        "from tests.fixtures.example_capability import example_capability;"
        "from understudy.artifact.hashing import content_hash;"
        "print(content_hash(example_capability()))"
    )
    hashes = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }

    assert hashes == {content_hash(example_capability())}


def test_hash_does_not_collide_when_a_default_is_made_explicit() -> None:
    """exclude_none drops unset fields from the dump; a field set to the value it
    would have defaulted to must still be distinguishable from one left unset."""
    implicit = example_capability()
    explicit = example_capability()
    explicit.steps[3].on_no_match = Disposition(do="escalate", code="commit_outcome_unknown")

    assert content_hash(implicit) != content_hash(explicit)


def test_hash_changes_when_behaviour_changes() -> None:
    capability = example_capability()
    before = content_hash(capability)
    capability.steps[3].timeout_ms += 1

    assert content_hash(capability) != before


def test_json_schema_exports() -> None:
    schema = capability_json_schema()

    assert schema["title"] == "Capability"
    assert "steps" in schema["properties"]


def test_approved_lifecycle_needs_its_evidence() -> None:
    with pytest.raises(ValidationError, match="approved artifact needs"):
        Lifecycle(status="approved", reviewed_by="someone")


def test_navigate_and_target_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="navigate needs a url"):
        Step(id="s1", intent="go", action="navigate", effect="read")


def test_navigate_takes_no_target() -> None:
    with pytest.raises(ValidationError, match="navigate takes no target"):
        Step(
            id="s1",
            intent="go",
            action="navigate",
            effect="read",
            url="{base_url}/x.htm",
            target=Target(
                description="stray",
                fingerprint=Fingerprint(role="button"),
                locators=[Locator(by="css", value="#x", stability="high", note="n")],
            ),
        )


def test_non_navigate_steps_need_a_target() -> None:
    with pytest.raises(ValidationError, match="click needs a target"):
        Step(id="s1", intent="click", action="click", effect="commit")


def test_extract_spec_belongs_only_to_extract_steps() -> None:
    with pytest.raises(ValidationError, match="only extract steps"):
        Step(
            id="s1",
            intent="go",
            action="navigate",
            effect="read",
            url="{base_url}/x.htm",
            extract=Extraction(output="x"),
        )


def test_duplicate_step_ids_are_rejected() -> None:
    capability = example_capability()
    data = json.loads(canonical_json(capability))
    data["steps"][1]["id"] = data["steps"][0]["id"]

    with pytest.raises(ValidationError, match="duplicate step id"):
        Capability.model_validate(data)


def test_enum_inputs_need_values() -> None:
    with pytest.raises(ValidationError, match="needs enum values"):
        InputSpec(name="kind", type="enum", description="x")


def test_a_fingerprint_cannot_be_empty() -> None:
    with pytest.raises(ValidationError, match="at least one field"):
        Fingerprint()


def test_disposition_codes_follow_the_disposition() -> None:
    with pytest.raises(ValidationError, match="needs a code"):
        Disposition(do="business")
    with pytest.raises(ValidationError, match="takes no code"):
        Disposition(do="continue", code="nope")


def test_commit_step_defaults_to_escalating_an_unknown_outcome() -> None:
    commit = example_capability().step("s4_submit")
    assert commit is not None

    assert commit.resolved_on_no_match.do == "escalate"
    assert commit.resolved_on_no_match.code == "commit_outcome_unknown"


def test_read_step_defaults_to_failing_on_an_unknown_state() -> None:
    read = example_capability().step("s1_open_form")
    assert read is not None

    assert read.resolved_on_no_match.do == "fail"
    assert read.resolved_on_no_match.code == "unknown_state"


def test_visible_needs_a_matcher() -> None:
    with pytest.raises(ValidationError, match="visible needs at least one"):
        VisibleCondition()


def test_selected_takes_exactly_one_value_source() -> None:
    with pytest.raises(ValidationError, match="exactly one of"):
        SelectedCondition(target="#x", value="a", param="b")


def test_http_status_range_is_ordered() -> None:
    with pytest.raises(ValidationError, match="min must be <= max"):
        HttpStatusCondition(min=500, max=400)


def test_url_only_signatures_have_no_element_condition() -> None:
    assert not has_element_condition(UrlCondition(path_regex="/x"))
    assert has_element_condition(
        AllCondition(all=[UrlCondition(path_regex="/x"), VisibleCondition(text="hi")])
    )
    assert has_element_condition(NotCondition(not_=VisibleCondition(text="hi")))


def test_absent_is_constructable_and_round_trips() -> None:
    """`absent` shares its matcher list with `visible`; a class-level lookup bug
    once made every artifact using it fail to load with a TypeError."""
    condition = TypeAdapter(Signature).validate_python({"kind": "absent", "text": "Error"})

    assert isinstance(condition, AbsentCondition)
    assert condition.text == "Error"
    assert has_element_condition(condition)


def test_absent_needs_a_matcher_like_visible_does() -> None:
    with pytest.raises(ValidationError, match="absent needs at least one"):
        AbsentCondition()


def test_params_used_walks_the_whole_tree() -> None:
    signature = AllCondition(
        all=[
            SelectedCondition(target="#a", param="one"),
            NotCondition(not_=SelectedCondition(target="#b", param="two")),
        ]
    )

    assert params_used(signature) == {"one", "two"}


def test_not_round_trips_through_its_alias() -> None:
    condition = NotCondition.model_validate(
        {"kind": "not", "not": {"kind": "url", "path_regex": "/x"}}
    )

    assert condition.model_dump(by_alias=True)["not"]["path_regex"] == "/x"


def test_a_reentry_point_must_load_a_page() -> None:
    """Replay restarts there after a recovery; only a navigate starts from a fresh page."""
    step = example_capability().steps[1].model_dump()
    step["reentry_point"] = True
    with pytest.raises(ValidationError, match="re-entry point must be a navigate"):
        Step.model_validate(step)
