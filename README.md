# MCP-Powered Data Science Assistant

MCP-Powered Data Science Assistant is a Python MCP server that turns an LLM into a practical data-analysis copilot. It can inspect tabular datasets, generate charts, train baseline Random Forest models, scrape public Wikipedia tables, and call Gemini for reasoning-heavy follow-up tasks.

The repository is organized as an installable Python package with module-based entrypoints, which makes it easier to run locally, wire into Claude Desktop, and extend without relying on fragile file-path startup commands.

## Highlights

- Installable `src`-layout package with CLI entrypoints and module-based startup.
- Safe local file access restricted to a configurable data directory.
- Structured tool errors that MCP hosts can surface cleanly.
- Optional terminal chat client for local demos.
- Lightweight smoke tests covering import and path-resolution regressions.

## Repository Layout

```text
.
├── data/                          Sample datasets
├── docs/                          Architecture and design notes
├── examples/                      Demo prompts for MCP hosts
├── outputs/generated_plots/       Saved chart output
├── src/mcp_data_science_assistant/
│   ├── __main__.py                `python -m mcp_data_science_assistant`
│   ├── chat_client.py             Optional terminal client
│   ├── runtime.py                 Shared launch and path helpers
│   └── server.py                  MCP server tools
├── tests/                         Smoke tests
├── .env.example                   Runtime configuration template
├── .editorconfig                  Basic editor defaults
├── pyproject.toml                 Package metadata and tooling config
└── requirements.txt               Pinned runtime dependencies
```

## Core Tools

1. `load_csv(file_path)`
   Returns schema, missing values, duplicates, preview rows, and numeric summary stats.
2. `plot_histogram(file_path, column, bins=20)`
   Generates a PNG histogram and returns it as an MCP image.
3. `train_random_forest(file_path, target, features=None)`
   Trains a baseline Random Forest model and returns evaluation metrics plus readable feature importances.
4. `scrape_table_from_wikipedia(url, table_index=0)`
   Extracts structured rows from a public Wikipedia table.
5. `fetch_web_json(url)`
   Fetches and previews JSON from a public endpoint.
6. `query_google_gemini(prompt, model=None)`
   Calls Gemini through the official Google GenAI SDK.

## Quick Start

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install the package

Recommended for local development and MCP host integration:

```bash
pip install -e ".[dev]"
```

If you want a pinned runtime environment first, install the locked dependencies before the editable package:

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and set the keys you need.

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Enables the Gemini helper tool | unset |
| `GEMINI_MODEL` | Default Gemini model | `gemini-2.5-flash` |
| `ANTHROPIC_API_KEY` | Enables the terminal chat client | unset |
| `ANTHROPIC_MODEL` | Default Anthropic model for the chat client | `claude-sonnet-4-20250514` |
| `MCP_SERVER_NAME` | Display name exposed to MCP hosts | `MCP-Powered Data Science Assistant` |
| `MCP_PROJECT_ROOT` | Explicit base path for relative dataset and output directories | unset |
| `ALLOWED_DATA_DIR` | Directory allowed for CSV and TSV reads | `data` |
| `OUTPUT_DIR` | Directory where generated plots are written | `outputs/generated_plots` |
| `MAX_CSV_SIZE_MB` | Maximum dataset size in megabytes | `50` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `LOG_FILE` | Optional rotating log file path | unset |

`MCP_PROJECT_ROOT` matters when an MCP host launches the package from outside the repository directory. When it is set, relative paths such as `data/` and `outputs/generated_plots/` resolve from that root instead of the host's current working directory.

### 4. Run the MCP server

After the editable install, either command works:

```bash
mcp-ds-server
```

```bash
python -m mcp_data_science_assistant.server
```

### 5. Optional: run the terminal chat client

```bash
mcp-ds-chat
```

Example prompt:

```text
Analyze data/churn_sample.csv and tell me which feature matters most for churn.
```

### 6. Connect Claude Desktop

Use [claude_desktop_config.example.json](claude_desktop_config.example.json) as a template. The important part is that Claude Desktop should launch the package with your virtual-environment Python interpreter and pass `MCP_PROJECT_ROOT` so the server can still find `data/` and `outputs/` reliably.

## Example Workflow

1. Call `load_csv("data/churn_sample.csv")`.
2. Inspect missing values, duplicates, and candidate features.
3. Call `train_random_forest(file_path="data/churn_sample.csv", target="Churn")`.
4. Review `top_feature_importances` such as `Contract = Month-to-month`.
5. Ask the host LLM to summarize the results in business language.

## Development

```bash
pytest
ruff check .
```

Architecture notes live in [docs/architecture.md](docs/architecture.md), and example prompts live in [examples/demo_queries.md](examples/demo_queries.md).

## Security and Runtime Notes

- Tool-based file reads are restricted to `ALLOWED_DATA_DIR`.
- Directory traversal outside the allowed directory is rejected.
- Only `.csv` and `.tsv` files are accepted by dataset tools.
- Oversized datasets are rejected using `MAX_CSV_SIZE_MB`.
- Wikipedia scraping is restricted to `wikipedia.org` URLs in this demo.
- Logs stay on stderr for MCP compatibility and can optionally rotate to a file.

## Future Work

- Add richer explainability such as SHAP or partial dependence plots.
- Expand automated coverage beyond smoke tests into integration tests.
- Add deployment examples for additional MCP hosts and transports.
