"""Shared runtime helpers for launch commands and repo-aware path resolution."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

PACKAGE_NAME = "mcp_data_science_assistant"


@lru_cache(maxsize=1)
def find_repo_root() -> Path | None:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / PACKAGE_NAME).exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def get_workspace_root() -> Path:
    configured_root = os.getenv("MCP_PROJECT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    repo_root = find_repo_root()
    if repo_root is not None:
        return repo_root

    return Path.cwd().resolve()


def resolve_workspace_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (get_workspace_root() / path).resolve()


def _package_is_importable() -> bool:
    """Return True when mcp_data_science_assistant is on sys.path and importable."""
    import importlib.util
    return importlib.util.find_spec(PACKAGE_NAME) is not None


def build_server_command() -> tuple[str, list[str]]:
    """Return the command and args needed to launch the MCP server.

    Prefers the module entrypoint when the package is installed/importable.
    Falls back to a direct script path for source-checkout usage without an
    editable install, so the chat client can still connect.
    """
    if _package_is_importable():
        return sys.executable, ["-m", f"{PACKAGE_NAME}.server"]

    # Package not importable — launch the server script directly by path.
    server_script = Path(__file__).resolve().parent / "server.py"
    return sys.executable, [str(server_script)]