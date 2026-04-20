# Architecture

## System Overview

```text
User
  │
  ▼
Claude Desktop or Terminal Chat Client
  │
  ▼
MCP Server (FastMCP)
  ├── load_csv
  ├── plot_histogram
  ├── train_random_forest
  ├── scrape_table_from_wikipedia
  ├── fetch_web_json
  └── query_google_gemini
          │
          ├── Local CSV and TSV datasets
          ├── Public web pages and JSON APIs
          └── Gemini API
```

## Package Structure

- `server.py` defines the MCP resources, prompts, tools, error handling, and model pipeline.
- `chat_client.py` provides an optional terminal client that lets Anthropic drive the MCP tools.
- `runtime.py` centralizes workspace-root discovery, relative-path resolution, and the server launch command.
- `__main__.py` provides a package-level module entrypoint.

## Request Flow

1. An MCP host launches the server with `python -m mcp_data_science_assistant.server` or the `mcp-ds-server` console script.
2. The server loads environment variables, resolves its workspace root, and configures allowed data and output directories.
3. The host requests a tool call.
4. The tool validates input, performs the data or network action, and returns structured JSON or an image payload.
5. Failures are normalized into `error` and `error_type` fields so hosts can report them consistently.

## Path Resolution Strategy

Relative paths resolve in this order:

1. `MCP_PROJECT_ROOT`, when explicitly configured.
2. The repository root, when the package is running from a source checkout.
3. The current working directory, as a fallback for installed environments.

That approach removes the previous assumption that the server must always be launched from a `src/.../server.py` file path.

## Safety Boundaries

- Dataset access is restricted to `ALLOWED_DATA_DIR`.
- Directory traversal outside that directory is rejected.
- Dataset reads are limited to `.csv` and `.tsv` files.
- Large files are blocked using `MAX_CSV_SIZE_MB`.
- Wikipedia scraping is intentionally scoped to `wikipedia.org` in this demo.
- Logging goes to stderr so MCP stdio transports stay clean.

## Modeling Choices

- Random Forest provides a reasonable baseline for classification and regression without forcing extensive feature engineering.
- Numeric and categorical preprocessing run through a `ColumnTransformer` with imputation and one-hot encoding.
- Feature names are simplified after one-hot encoding so the returned importances are readable in an MCP host UI.
