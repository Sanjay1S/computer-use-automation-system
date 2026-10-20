import json
import os
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from playwright.sync_api import Page, sync_playwright
from pydantic import ValidationError

from understudy import __version__
from understudy.artifact import hashing
from understudy.artifact.export import capability_json_schema, tool_definition
from understudy.artifact.lint import lint as run_lint
from understudy.artifact.review import render
from understudy.artifact.schema import Capability
from understudy.chaos import Chaos, parse_fault
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
    runs_dir: Annotated[Path, typer.Option("--runs-dir", help="Where run folders go.")] = Path(
        "runs"
    ),
) -> None:
    """Replay a capability: no model, deterministic steps, a typed result.

    Prints the result as JSON (outputs raw, for the caller); the copy in the run
    folder is redacted. Exit code: 0 succeeded, 1 failed, 3 rejected, 4 business
    outcome, 5 needs a human.
    """
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
    except (OSError, ValueError, MissingCredentials) as error:
        typer.echo(f"cannot replay: {error}", err=True)
        raise typer.Exit(2) from None

    replayer = Replayer(
        capability, gate=gate, profile=profile, credentials=credentials, runs_dir=runs_dir
    )
    approval = grant(capability, values, tenant.id, "cli --approve") if approve else None
    with ExitStack() as stack:

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
        )
    typer.echo(result.model_dump_json(indent=2, exclude_none=True))
    typer.echo(f"{result.status}: run folder {runs_dir / result.run_id}", err=True)
    raise typer.Exit(EXIT_CODES[result.status])


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


def _parse_inputs(pairs: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise ValueError("--input takes name=value (an argument had no name)")
        if name in values:
            raise ValueError(f"input {name!r} given twice")
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
