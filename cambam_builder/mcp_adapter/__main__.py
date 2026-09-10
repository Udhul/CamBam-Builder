"""Guarded command-line launcher for the local MCP server."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


EXPECTED_MCP_VERSION = "2.2.0"


def _require_mcp() -> None:
    """Reject unsupported interpreters/dependency installs before SDK import."""
    if sys.version_info < (3, 10):
        raise SystemExit("cambam-mcp requires Python 3.10 or newer; install the [mcp] extra")
    try:
        installed = version("mcp")
    except PackageNotFoundError:
        raise SystemExit("cambam-mcp requires the optional [mcp] extra (mcp==2.2.0)") from None
    if installed != EXPECTED_MCP_VERSION:
        raise SystemExit(
            f"cambam-mcp requires mcp=={EXPECTED_MCP_VERSION}; found mcp=={installed}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local CamBam MCP server")
    parser.add_argument(
        "--workspace",
        required=True,
        help="absolute existing directory used as the workspace boundary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _require_mcp()
    args = _parser().parse_args(argv)
    workspace = Path(args.workspace)
    if not workspace.is_absolute() or not workspace.is_dir():
        _parser().error("--workspace must be an existing absolute directory")

    # Keep all optional SDK imports below the version/interpreter guard.
    from .server import run_stdio

    try:
        run_stdio(str(workspace))
    except (OSError, ValueError) as exc:
        _parser().error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests
    main()
