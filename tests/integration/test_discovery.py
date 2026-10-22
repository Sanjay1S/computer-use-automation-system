"""Discovery end to end, offline: a scripted model drives the real driver, recorder and
compiler against the routed fixture site, and the compiled draft is then replayed by
the replay engine. This is the whole chain with only the model faked.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Browser, Page

from tests.conftest import REPO
from tests.fixtures.fake_model import ScriptedModel, open_account, result_text
from tests.fixtures.site import BASE_URL, PASSWORD, SSN, USERNAME, FixtureSite
from understudy.artifact import hashing
from understudy.artifact import signatures as sig
from understudy.artifact.schema import Capability, ParamValue
from understudy.discovery.agent import FALLBACK_BETA
from understudy.discovery.driver import ApprovalRequest
from understudy.discovery.goal import load_goal
from understudy.discovery.prompts import SYSTEM_PROMPT
from understudy.discovery.run import DiscoveryResult, discover
from understudy.policy.gate import PolicyGate
from understudy.profile import load_profile
from understudy.redaction.redactor import Redactor
from understudy.replay.engine import Replayer
from understudy.replay.preflight import grant
from understudy.replay.result import ReplayResult
from understudy.session import Credentials
from understudy.surface.web import CONTEXT_OPTIONS

GOAL = load_goal(REPO / "goals" / "open_account.yaml")
PROFILE = load_profile(REPO / "apps" / "parabank" / "profile.yaml")
CREATE = "/parabank/services_proxy/bank/createAccount"
DEMO = {"account_type": "SAVINGS", "funding_account_id": "13344"}


class Operator:
    """Stands in for the person approving commits at the terminal."""

    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.asked: list[ApprovalRequest] = []

    def __call__(self, request: ApprovalRequest) -> bool:
        self.asked.append(request)
        return self.answer


def run_discovery(
    page: Page,
    gate: PolicyGate,
    tmp_path: Path,
    model: ScriptedModel,
    operator: Operator,
    values: dict[str, str] | None = None,
) -> DiscoveryResult:
    return discover(
        GOAL,
        GOAL.values(values),
        gate=gate,
        profile=PROFILE,
        credentials=Credentials(USERNAME, PASSWORD),
        model=model,
        open_page=lambda: page,
        runs_dir=tmp_path / "runs",
        approve=operator,
        draft_path=tmp_path / "drafts" / "parabank.open_account.json",
        redactor=Redactor(b"k" * 32),
        screenshots=True,
    )


def replay(
    capability: Capability, page: Page, gate: PolicyGate, tmp_path: Path, inputs: dict[str, str]
) -> ReplayResult:
    replayer = Replayer(
        capability,
        gate=gate,
        profile=PROFILE,
        credentials=Credentials(USERNAME, PASSWORD),
        runs_dir=tmp_path / "replays",
        redactor=Redactor(b"k" * 32),
        screenshots=False,
    )
    return replayer.run(
        inputs,
        open_page=lambda: page,
        approval=grant(capability, inputs, gate.tenant.id, "test"),
        allow_draft=True,
    )


@pytest.fixture
def fresh_page(browser: Browser, site: FixtureSite) -> Iterator[Callable[[], Page]]:
    """A page in a browser context of its own, as every real run gets: one context's
    request guard judges every page in it, so two runs must not share one."""
    contexts = []

    def make() -> Page:
        context = browser.new_context(**CONTEXT_OPTIONS)
        site.install(context)
        contexts.append(context)
        return context.new_page()

    yield make
    for context in contexts:
        context.close()


@pytest.fixture
def discovered(
    page: Page, site: FixtureSite, gate: PolicyGate, tmp_path: Path
) -> tuple[DiscoveryResult, ScriptedModel, Operator]:
    model = ScriptedModel(open_account(BASE_URL, "SAVINGS", "13344"))
    operator = Operator()
    result = run_discovery(page, gate, tmp_path, model, operator)
    return result, model, operator


def test_discovery_compiles_a_lint_clean_draft(
    discovered: tuple[DiscoveryResult, ScriptedModel, Operator], site: FixtureSite
) -> None:
    result, _, operator = discovered

    assert result.status == "compiled", result
    assert result.lint_ok, result.lint
    assert not [f for f in result.lint if "[error]" in f]
    assert result.draft is not None and Path(result.draft).is_file()
    # One commit, approved once by a person, for the submit button and nothing else.
    assert [(r.member, r.target) for r in operator.asked] == [
        ("left_click", "button 'Open New Account'")
    ]
    assert site.reached(CREATE) == 1
    assert result.models_served == ["claude-opus-5-5"]
    assert result.usage.cost_usd > 0


def test_the_draft_has_the_shape_of_the_flow(
    discovered: tuple[DiscoveryResult, ScriptedModel, Operator],
) -> None:
    result, _, _ = discovered
    assert result.draft is not None
    capability = hashing.load(Path(result.draft))

    assert [s.action for s in capability.steps] == [
        "navigate",
        "select",
        "select",
        "click",
        "extract",
    ]
    navigate, kind, funding, submit, read = capability.steps
    assert navigate.url == "{base_url}/openaccount.htm" and navigate.reentry_point
    assert [s.id for s in capability.steps if s.reentry_point] == [navigate.id]
    assert kind.value == ParamValue(param="account_type")
    assert funding.value == ParamValue(param="funding_account_id")
    assert isinstance(funding.wait_before, sig.OptionsLoadedCondition)
    # Free-form param: a missing account is the app's answer. Enum param: preflight's job.
    assert funding.on_target_absent is not None
    assert kind.on_target_absent is None
    assert [o.code for o in capability.outcomes] == ["funding_account_id_not_found"]
    assert capability.outcomes[0].detected_by == funding.id
    # Only the submit commits; it is never retried, and it waits on a recorded change.
    assert [s.id for s in capability.steps if s.effect == "commit"] == [submit.id]
    assert submit.retry.max_attempts == 0 and submit.expect
    assert submit.target is not None
    assert submit.target.fingerprint.role == "button"
    assert submit.target.locators[0].by == "role"
    # The output's own value is data: no locator or fingerprint may carry it.
    assert read.target is not None and read.extract is not None
    sample = re.compile(r"\b13566\b")
    assert not sample.search(json.dumps(read.target.model_dump(mode="json")))
    assert read.extract.output == "new_account_id"
    assert [o.source_step for o in capability.outputs] == [read.id]
    assert capability.side_effects.cls == "commit"
    assert capability.side_effects.approval == "per_invocation"
    assert capability.provenance.authored == "discovered"
    assert capability.provenance.models_served == ["claude-opus-5-5"]


def test_the_success_checkpoint_is_not_vacuous(
    discovered: tuple[DiscoveryResult, ScriptedModel, Operator],
) -> None:
    result, _, _ = discovered
    assert result.draft is not None
    capability = hashing.load(Path(result.draft))
    names = {
        (n.role, n.name)
        for n in sig.walk(capability.success.signature)
        if isinstance(n, sig.VisibleCondition)
    }
    assert ("heading", "Account Opened!") in names
    control = capability.success.negative_control
    assert control is not None
    nodes = sig.walk(control)
    assert any(isinstance(n, sig.NotCondition) for n in nodes)
    assert any(isinstance(n, sig.VisibleCondition) and n.name == "Open New Account" for n in nodes)


def test_the_compiled_draft_replays_with_other_inputs(
    discovered: tuple[DiscoveryResult, ScriptedModel, Operator],
    fresh_page: Callable[[], Page],
    site: FixtureSite,
    gate: PolicyGate,
    tmp_path: Path,
) -> None:
    result, _, _ = discovered
    assert result.draft is not None
    capability = hashing.load(Path(result.draft))
    created_before = site.reached(CREATE)

    outcome = replay(
        capability,
        fresh_page(),
        gate,
        tmp_path,
        {"account_type": "CHECKING", "funding_account_id": "12456"},
    )

    assert outcome.status == "succeeded", outcome
    assert outcome.outputs is not None
    new_id = outcome.outputs["new_account_id"]
    assert re.fullmatch(r"\d+", new_id) and new_id in site.accounts
    assert outcome.side_effects.commit_state == "committed"
    assert site.reached(CREATE) == created_before + 1
    assert outcome.drift == []


def test_the_compiled_draft_reports_a_foreign_account_as_a_business_outcome(
    discovered: tuple[DiscoveryResult, ScriptedModel, Operator],
    fresh_page: Callable[[], Page],
    site: FixtureSite,
    gate: PolicyGate,
    tmp_path: Path,
) -> None:
    result, _, _ = discovered
    assert result.draft is not None
    capability = hashing.load(Path(result.draft))
    created_before = site.reached(CREATE)

    outcome = replay(
        capability,
        fresh_page(),
        gate,
        tmp_path,
        {"account_type": "SAVINGS", "funding_account_id": "99999"},
    )

    assert outcome.status == "business_outcome", outcome
    assert outcome.outcome is not None
    assert outcome.outcome.code == "funding_account_id_not_found"
    assert outcome.side_effects.commit_state == "not_attempted"
    assert site.reached(CREATE) == created_before


def test_every_request_is_the_same_shape_and_the_history_only_grows(
    discovered: tuple[DiscoveryResult, ScriptedModel, Operator],
) -> None:
    _, model, _ = discovered
    requests = model.requests
    assert len(requests) >= 8

    first = requests[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["system"] == SYSTEM_PROMPT
    assert first["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert first["output_config"] == {"effort": "high"}
    assert first["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert first["fallbacks"] == "default" and first["betas"] == [FALLBACK_BETA]
    assert first["cache_control"] == {"type": "ephemeral"}
    toolset = first["tools"][0]
    assert toolset["type"] == "browser_toolset_20260801"
    disabled = {name for name, cfg in toolset["configs"].items() if cfg == {"enabled": False}}
    assert {"find", "new_tab", "javascript_exec", "file_upload", "left_click_drag"} <= disabled
    assert {"navigate", "read_page", "form_input", "left_click"}.isdisjoint(disabled)
    custom = {t["name"]: t for t in first["tools"][1:]}
    assert set(custom) == {"record_output", "declare_success", "request_human", "give_up"}
    assert all(t["strict"] for t in custom.values())

    for earlier, later in itertools.pairwise(requests):
        assert later["tools"] == earlier["tools"] and later["system"] == earlier["system"]
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]
        assert len(later["messages"]) == len(earlier["messages"]) + 2


def test_the_run_folder_keeps_no_secrets_or_raw_values(
    discovered: tuple[DiscoveryResult, ScriptedModel, Operator], site: FixtureSite, tmp_path: Path
) -> None:
    result, _, _ = discovered
    folder = tmp_path / "runs" / result.run_id
    names = {p.name for p in folder.iterdir()}
    assert {"trajectory.json", "events.jsonl", "result.json", "usage.json", "draft.json"} <= names
    assert {"review.txt", "screenshots"} <= names

    text = "\n".join(
        p.read_text(errors="ignore")
        for p in folder.rglob("*")
        if p.suffix in (".json", ".jsonl", ".txt")
    )
    assert PASSWORD not in text
    assert site.session_id not in text and "jsessionid" not in text.lower()
    assert SSN not in text
    assert "13344" not in text  # the demo funding account, a financial id
    assert "13566" not in text  # the account discovery opened

    trajectory: dict[str, Any] = json.loads((folder / "trajectory.json").read_text())
    assert trajectory["params"]["funding_account_id"] == "••••4"
    assert trajectory["params"]["account_type"] == "SAVINGS"
    assert [o["name"] for o in trajectory["outputs"]] == ["new_account_id"]


def test_reaching_the_entry_by_a_link_compiles_too(
    page: Page,
    fresh_page: Callable[[], Page],
    site: FixtureSite,
    gate: PolicyGate,
    tmp_path: Path,
) -> None:
    model = ScriptedModel(open_account(BASE_URL, "SAVINGS", "13344", via_link=True))
    result = run_discovery(page, gate, tmp_path, model, Operator())

    assert result.status == "compiled" and result.lint_ok, result
    assert result.draft is not None
    capability = hashing.load(Path(result.draft))
    assert [s.action for s in capability.steps][:2] == ["navigate", "click"]
    assert capability.steps[0].url == "{base_url}/overview.htm"
    assert capability.steps[1].effect == "read"  # a navigation link, not a commit

    outcome = replay(
        capability,
        fresh_page(),
        gate,
        tmp_path,
        {"account_type": "SAVINGS", "funding_account_id": "54321"},
    )
    assert outcome.status == "succeeded", outcome


def test_a_refused_commit_is_never_sent_and_nothing_is_compiled(
    page: Page, site: FixtureSite, gate: PolicyGate, tmp_path: Path
) -> None:
    model = ScriptedModel(open_account(BASE_URL, "SAVINGS", "13344"))
    operator = Operator(answer=False)

    result = run_discovery(page, gate, tmp_path, model, operator)

    assert result.status == "gave_up" and result.draft is None
    assert len(operator.asked) == 1
    assert site.reached(CREATE) == 0
    assert any("did not grant permission" in result_text(r) for r in model.requests)
    folder = tmp_path / "runs" / result.run_id
    trajectory = json.loads((folder / "trajectory.json").read_text())
    refused = [c for c in trajectory["calls"] if not c["ok"]]
    assert [c["member"] for c in refused] == ["left_click"]
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    assert [e["data"]["approval"] for e in events if e["kind"] == "control"] == ["refused"]


def test_a_run_that_breaks_midway_still_reports_what_it_spent(
    page: Page, site: FixtureSite, gate: PolicyGate, tmp_path: Path
) -> None:
    from tests.fixtures.fake_model import Script, browser

    def breaks() -> Script:
        yield browser("navigate", url=f"{BASE_URL}/openaccount.htm")
        yield browser("read_page")
        raise RuntimeError("the model client broke")

    result = run_discovery(page, gate, tmp_path, ScriptedModel(breaks), Operator())

    assert (result.status, result.code, result.turns) == ("failed", "engine_error", 2)
    assert result.usage.cost_usd > 0
    folder = tmp_path / "runs" / result.run_id
    assert json.loads((folder / "usage.json").read_text())["by_model"]
    assert (folder / "trajectory.json").is_file()
