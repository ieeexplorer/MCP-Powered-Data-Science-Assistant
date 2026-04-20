import asyncio

from mcp_data_science_assistant import runtime, server


def test_build_server_command_uses_module_entrypoint_when_installed() -> None:
    """When the package is importable, use the module entrypoint."""
    command, args = runtime.build_server_command()
    assert command
    # Running under pytest with src/ on sys.path: package is importable.
    assert args == ["-m", "mcp_data_science_assistant.server"]


def test_build_server_command_falls_back_to_script_path(monkeypatch) -> None:
    """When the package is not importable, fall back to the direct script path."""
    monkeypatch.setattr(runtime, "_package_is_importable", lambda: False)
    runtime.build_server_command.cache_clear() if hasattr(runtime.build_server_command, "cache_clear") else None
    command, args = runtime.build_server_command()
    assert command
    assert len(args) == 1
    assert args[0].endswith("server.py")


def test_load_csv_returns_expected_dataset_summary() -> None:
    result = asyncio.run(server.load_csv("data/churn_sample.csv"))

    assert result["shape"]["rows"] > 0
    assert "Churn" in result["columns"]
    assert "MonthlyCharges" in result["numeric_summary"]


def test_load_csv_rejects_paths_outside_allowed_directory() -> None:
    result = asyncio.run(server.load_csv("../README.md"))

    assert result["error_type"] == "FileAccessError"