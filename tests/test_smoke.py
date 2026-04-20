import asyncio

from mcp_data_science_assistant import runtime, server


def test_build_server_command_uses_module_entrypoint() -> None:
    command, args = runtime.build_server_command()
    assert command
    assert args == ["-m", "mcp_data_science_assistant.server"]


def test_load_csv_returns_expected_dataset_summary() -> None:
    result = asyncio.run(server.load_csv("data/churn_sample.csv"))

    assert result["shape"]["rows"] > 0
    assert "Churn" in result["columns"]
    assert "MonthlyCharges" in result["numeric_summary"]


def test_load_csv_rejects_paths_outside_allowed_directory() -> None:
    result = asyncio.run(server.load_csv("../README.md"))

    assert result["error_type"] == "FileAccessError"