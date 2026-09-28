"""`faar-demo --help` must work with the pinned click and typer."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from faar.cli import app

runner = CliRunner()


def _command_names() -> list[str]:
    # Read the names from typer's registry, not from the click group. Building the
    # click group at collection time would turn an incompatible click into a
    # collection error instead of a failing test.
    return sorted(
        command.name or command.callback.__name__.replace("_", "-") for command in app.registered_commands
    )


def test_top_level_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for name in _command_names():
        assert name in result.output


@pytest.mark.parametrize("name", _command_names())
def test_each_command_help_renders(name: str) -> None:
    result = runner.invoke(app, [name, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output
