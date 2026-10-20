import typer

from understudy import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Discover UI flows with an LLM, compile them into capabilities, replay them."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)
