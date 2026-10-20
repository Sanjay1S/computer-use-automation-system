import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from understudy import __version__
from understudy.artifact import hashing
from understudy.artifact.export import capability_json_schema, tool_definition
from understudy.artifact.lint import lint as run_lint
from understudy.artifact.review import render
from understudy.artifact.schema import Capability

app = typer.Typer(no_args_is_help=True, add_completion=False)

CAPABILITIES = Path("capabilities")


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


def _resolve(capability_id: str) -> Path:
    if capability_id.endswith(".json"):
        return Path(capability_id)
    matches = sorted(CAPABILITIES.glob(f"{capability_id}@*.json"))
    if not matches:
        typer.echo(f"no approved artifact for {capability_id!r} in {CAPABILITIES}/", err=True)
        raise typer.Exit(2)
    return matches[-1]
