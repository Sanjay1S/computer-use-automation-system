import json
import os
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import anthropic
import typer
import yaml
from playwright.sync_api import Page, sync_playwright
from pydantic import ValidationError

from understudy import __version__
from understudy.artifact import hashing
from understudy.artifact.export import capability_json_schema, tool_definition
from understudy.artifact.lint import lint as run_lint
from understudy.artifact.review import render
from understudy.artifact.schema import Capability
from understudy.chaos import Chaos, parse_fault
from understudy.control.console import DEFAULT_PORT, Console
from understudy.control.handoff import Control
from understudy.control.interventions import Intervention
from understudy.control.plane import ControlPlane
from understudy.discovery.agent import AnthropicModel, ModelTurn
from understudy.discovery.goal import load_goal
from understudy.discovery.run import discover
from understudy.policy.gate import PolicyGate
from understudy.profile import load_profile
from understudy.replay.engine import Replayer
from understudy.replay.preflight import grant
from understudy.session import Credentials, MissingCredentials
from understudy.surface.web import CONTEXT_OPTIONS
from understudy.tenant import load_tenant

app = typer.Typer(no_args_is_help=True, add_completion=False)

CAPABILITIES = Path("capabilities")
APPS = Path("apps")
TENANTS = Path("tenants")

EXIT_CODES = {"succeeded": 0, "failed": 1, "rejected": 3, "business_outcome": 4, "needs_human": 5}


@app.callback()
def main() -> None:
    """Discover UI flows with an LLM, compile them into capabilities, replay them."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command("lint")
def lint_command(
    artifact: Annotated[Path, typer.Argument(help="Path to a capability JSON file.")],
) -> None:
    """Check an artifact against the lint rules."""
    capability = _load(artifact)
    report = run_lint(capability)
    for finding in report.findings:
        typer.echo(str(finding))
    if report.ok:
        typer.echo(f"clean ({len(report.warnings)} warning(s))")
        return
    raise typer.Exit(1)


@app.command("review")
def review_command(
    artifact: Annotated[Path, typer.Argument(help="Path to a capability JSON file.")],
) -> None:
    """Render an artifact as a review summary for a human approver."""
    typer.echo(render(_load(artifact)), nl=False)


@app.command("describe")
def describe_command(
    capability_id: Annotated[str, typer.Argument(help="Capability id, or a path.")],
    as_: Annotated[str, typer.Option("--as", help="Output format: 'tool' or 'json'.")] = "tool",
) -> None:
    """Print the agent-facing contract for a capability."""
    capability = _load(_resolve(capability_id))
    if as_ == "tool":
        typer.echo(json.dumps(tool_definition(capability), indent=2))
    elif as_ == "json":
        typer.echo(hashing.canonical_json(capability), nl=False)
    else:
        typer.echo(f"unknown --as {as_!r}; expected 'tool' or 'json'", err=True)
        raise typer.Exit(2)


@app.command("approve")
def approve_command(
    artifact: Annotated[Path, typer.Argument(help="Path to a capability JSON file.")],
    reviewer: Annotated[str, typer.Option("--reviewer", help="Who reviewed it.")],
    out: Annotated[
        Path | None, typer.Option("--out", help="Where to write it. Default: capabilities/.")
    ] = None,
) -> None:
    """Approve an artifact, binding the approval to its content hash."""
    capability = _load(artifact)
    report = run_lint(capability)
    if not report.ok:
        for finding in report.errors:
            typer.echo(str(finding), err=True)
        typer.echo("not approved: lint errors must be fixed first", err=True)
        raise typer.Exit(1)

    capability.lifecycle.status = "approved"
    capability.lifecycle.reviewed_by = reviewer
    capability.lifecycle.approved_at = datetime.now(UTC)
    capability.lifecycle.approved_hash = hashing.content_hash(capability)

    destination = out or CAPABILITIES / f"{capability.id}@{capability.version}.json"
    hashing.save(capability, destination)
    typer.echo(f"approved {capability.id}@{capability.version}")
    typer.echo(f"  hash {capability.lifecycle.approved_hash}")
    typer.echo(f"  written to {destination}")


@app.command("replay")
def replay_command(
    capability_id: Annotated[str, typer.Argument(help="Capability id, or a path to its JSON.")],
    tenant_id: Annotated[str, typer.Option("--tenant", help="Tenant id under tenants/.")],
    inputs: Annotated[
        list[str] | None, typer.Option("--input", help="An input as name=value. Repeatable.")
    ] = None,
    approve: Annotated[
        bool, typer.Option("--approve", help="Approve the commit for exactly this invocation.")
    ] = False,
    allow_draft: Annotated[
        bool, typer.Option("--allow-draft", help="Run an unapproved artifact (development).")
    ] = False,
    inject: Annotated[
        list[str] | None,
        typer.Option("--inject", help="A fault to inject (sandbox tenants only). Repeatable."),
    ] = None,
    headed: Annotated[bool, typer.Option("--headed", help="Show the browser window.")] = False,
    on_stuck: Annotated[
        str,
        typer.Option(
            "--on-stuck",
            help="fail: report needs_human. escalate: hand the live session to an operator "
            "in the console.",
        ),
    ] = "fail",
    approval_mode: Annotated[
        str,
        typer.Option(
            "--approval",
            help="token: the commit is approved by --approve. interactive: an operator "
            "approves it in the console when the run reaches it.",
        ),
    ] = "token",
    console_port: Annotated[
        int, typer.Option("--console-port", help="Port of the operator console.")
    ] = DEFAULT_PORT,
    handoff_timeout: Annotated[
        float, typer.Option("--handoff-timeout", help="Seconds to wait for an operator.")
    ] = 1_800.0,
    runs_dir: Annotated[Path, typer.Option("--runs-dir", help="Where run folders go.")] = Path(
        "runs"
    ),
) -> None:
    """Replay a capability: no model, deterministic steps, a typed result.

    Prints the result as JSON (outputs raw, for the caller); the copy in the run
    folder is redacted. Exit code: 0 succeeded, 1 failed, 3 rejected, 4 business
    outcome, 5 needs a human.
    """
    if on_stuck not in ("fail", "escalate"):
        typer.echo(f"unknown --on-stuck {on_stuck!r}; expected fail or escalate", err=True)
        raise typer.Exit(2)
    if approval_mode not in ("token", "interactive"):
        typer.echo(f"unknown --approval {approval_mode!r}; expected token or interactive", err=True)
        raise typer.Exit(2)
    if approval_mode == "interactive" and approve:
        typer.echo("--approve and --approval interactive exclude each other", err=True)
        raise typer.Exit(2)
    load_env_file(Path(".env"))
    capability = _load(_resolve(capability_id))
    try:
        tenant = load_tenant(TENANTS / f"{tenant_id}.yaml")
        app_dir = APPS / tenant.vendor_app
        gate = PolicyGate.load(app_dir / "policy.yaml", tenant)
        profile = load_profile(app_dir / "profile.yaml")
        values = _parse_inputs(inputs or [])
        faults = [parse_fault(spec, capability) for spec in inject or []]
        credentials = Credentials.from_env(profile.login)
    except (OSError, ValueError, yaml.YAMLError, MissingCredentials) as error:
        typer.echo(f"cannot replay: {error}", err=True)
        raise typer.Exit(2) from None

    replayer = Replayer(
        capability, gate=gate, profile=profile, credentials=credentials, runs_dir=runs_dir
    )
    approval = grant(capability, values, tenant.id, "cli --approve") if approve else None
    with ExitStack() as stack:
        control = None
        if on_stuck == "escalate" or approval_mode == "interactive":
            control = Control(
                plane=_console(stack, ControlPlane(), runs_dir, console_port),
                escalate=on_stuck == "escalate",
                interactive_approval=approval_mode == "interactive",
                max_wait_s=handoff_timeout,
            )
            if on_stuck == "escalate" and not headed:
                typer.echo(
                    "note: the browser is headless, so an operator can decide in the console "
                    "but cannot act in the page; use --headed to allow a takeover",
                    err=True,
                )

        def open_page() -> Page:
            """Called only once preflight has passed: a rejected run starts no browser."""
            playwright = stack.enter_context(sync_playwright())
            browser = playwright.chromium.launch(headless=not headed, args=gate.chromium_args())
            stack.callback(browser.close)
            return browser.new_context(**CONTEXT_OPTIONS).new_page()

        result = replayer.run(
            values,
            open_page=open_page,
            approval=approval,
            allow_draft=allow_draft,
            chaos=Chaos(faults, gate) if faults else None,
            control=control,
        )
    typer.echo(result.model_dump_json(indent=2, exclude_none=True))
    typer.echo(f"{result.status}: run folder {runs_dir / result.run_id}", err=True)
    raise typer.Exit(EXIT_CODES[result.status])


@app.command("discover")
def discover_command(
    goal_path: Annotated[Path, typer.Argument(help="Path to a goal spec (YAML).")],
    tenant_id: Annotated[str, typer.Option("--tenant", help="Tenant id under tenants/.")],
    params: Annotated[
        list[str] | None,
        typer.Option("--param", help="Override a goal param's demo value: name=value."),
    ] = None,
    headed: Annotated[bool, typer.Option("--headed", help="Show the browser window.")] = False,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Draft path. Default: capabilities/drafts/<id>.json."),
    ] = None,
    effort: Annotated[str, typer.Option("--effort", help="Model effort level.")] = "high",
    on_stuck: Annotated[
        str,
        typer.Option(
            "--on-stuck",
            help="fail: end a stuck run. escalate: hand the live session to an operator, "
            "then let the model continue.",
        ),
    ] = "fail",
    console_port: Annotated[
        int, typer.Option("--console-port", help="Port of the operator console.")
    ] = DEFAULT_PORT,
    handoff_timeout: Annotated[
        float, typer.Option("--handoff-timeout", help="Seconds to wait for an operator.")
    ] = 1_800.0,
    runs_dir: Annotated[Path, typer.Option("--runs-dir", help="Where run folders go.")] = Path(
        "runs"
    ),
) -> None:
    """Discover a flow with the model and compile it into a draft capability.

    Runs only on sandbox tenants. Every commit the model attempts waits for an
    operator's approval in the console (its address is printed). Exit code: 0
    compiled and lint-clean, 1 not compiled or lint errors, 2 usage, 3 rejected.
    """
    load_env_file(Path(".env"))
    try:
        goal = load_goal(goal_path)
        tenant = load_tenant(TENANTS / f"{tenant_id}.yaml")
        app_dir = APPS / tenant.vendor_app
        gate = PolicyGate.load(app_dir / "policy.yaml", tenant)
        profile = load_profile(app_dir / "profile.yaml")
        values = goal.values(_parse_inputs(params or [], what="--param"))
        credentials = Credentials.from_env(profile.login)
    except (OSError, ValueError, yaml.YAMLError, MissingCredentials) as error:
        typer.echo(f"cannot discover: {error}", err=True)
        raise typer.Exit(2) from None
    if effort not in ("low", "medium", "high", "xhigh", "max"):
        typer.echo(f"unknown --effort {effort!r}", err=True)
        raise typer.Exit(2)
    if on_stuck not in ("fail", "escalate"):
        typer.echo(f"unknown --on-stuck {on_stuck!r}; expected fail or escalate", err=True)
        raise typer.Exit(2)

    with ExitStack() as stack:
        control = Control(
            plane=_console(stack, ControlPlane(), runs_dir, console_port),
            escalate=on_stuck == "escalate",
            max_wait_s=handoff_timeout,
        )

        def open_page() -> Page:
            playwright = stack.enter_context(sync_playwright())
            browser = playwright.chromium.launch(headless=not headed, args=gate.chromium_args())
            stack.callback(browser.close)
            return browser.new_context(**CONTEXT_OPTIONS).new_page()

        result = discover(
            goal,
            values,
            gate=gate,
            profile=profile,
            credentials=credentials,
            model=MODEL_FACTORY(),
            open_page=open_page,
            runs_dir=runs_dir,
            draft_path=out,
            effort=effort,
            control=control,
        )
    typer.echo(result.model_dump_json(indent=2, exclude_none=True))
    typer.echo(f"{result.status}: run folder {runs_dir / result.run_id}", err=True)
    if result.status == "rejected":
        raise typer.Exit(3)
    raise typer.Exit(0 if result.status == "compiled" and result.lint_ok else 1)


def _anthropic_model() -> ModelTurn:
    """The SDK resolves credentials itself (ANTHROPIC_API_KEY, or an auth profile)."""
    return AnthropicModel(anthropic.Anthropic())


MODEL_FACTORY: Callable[[], ModelTurn] = _anthropic_model
"""Swapped for a scripted model in tests."""


class _TerminalNotice:
    """The local intervention queue: tells the person at this terminal where to look."""

    def __init__(self, url: str) -> None:
        self.url = url

    def publish(self, intervention: Intervention) -> None:
        where = f" at step {intervention.step_id}" if intervention.step_id else ""
        typer.echo(
            f"\nA person is needed: {intervention.kind} ({intervention.reason}){where}."
            f"\n  Operator console: {self.url}",
            err=True,
        )


def _console(stack: ExitStack, plane: ControlPlane, runs_dir: Path, port: int) -> ControlPlane:
    """Starts the plane's console, running until the command ends."""
    try:
        console = Console(plane, runs_dir, port=port)
    except OSError as error:
        typer.echo(f"cannot start the operator console on port {port}: {error}", err=True)
        raise typer.Exit(2) from None
    stack.enter_context(console)
    plane.queue = _TerminalNotice(console.url)
    typer.echo(f"operator console: {console.url}", err=True)
    return plane


@app.command("schema")
def schema_command() -> None:
    """Print the JSON Schema for the capability artifact format."""
    typer.echo(json.dumps(capability_json_schema(), indent=2))


def _load(path: Path) -> Capability:
    try:
        return hashing.load(path)
    except FileNotFoundError:
        typer.echo(f"no such artifact: {path}", err=True)
        raise typer.Exit(2) from None
    except ValidationError as error:
        typer.echo(f"{path} is not a valid capability:\n{error}", err=True)
        raise typer.Exit(2) from None


def load_env_file(path: Path) -> None:
    """Plain KEY=value lines, as the Makefile reads them. Values already set in the
    environment win, and empty values count as unset."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and value and not os.environ.get(key):
            os.environ[key] = value


def _parse_inputs(pairs: list[str], *, what: str = "--input") -> dict[str, str]:
    values: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise ValueError(f"{what} takes name=value (an argument had no name)")
        if name in values:
            raise ValueError(f"{what} {name!r} given twice")
        values[name] = value
    return values


def _resolve(capability_id: str) -> Path:
    if capability_id.endswith(".json"):
        return Path(capability_id)
    matches = sorted(CAPABILITIES.glob(f"{capability_id}@*.json"))
    if not matches:
        typer.echo(f"no approved artifact for {capability_id!r} in {CAPABILITIES}/", err=True)
        raise typer.Exit(2)
    return matches[-1]
