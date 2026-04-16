# MCP-Powered Data Science Assistant

A production-ready MCP server that turns an LLM into a conversational data analyst.

This version adds stronger security boundaries, structured error handling, configurable logging, cleaner ML output, and a more resilient terminal client while keeping the original tool set intact.

## What changed

- File access is restricted to a configured data directory.
- CSV and TSV loading now enforces a maximum file size.
- Tool failures return structured error payloads instead of raw exceptions.
- Logging still goes to stderr for MCP compatibility and can also rotate to a file.
- Feature importance labels are easier to read after one-hot encoding.
- The terminal client retries Anthropic rate-limit and transient connection failures.

## Main tools

1. `load_csv(file_path)`
   Returns row and column counts, schema, missing values, duplicates, preview rows, and numeric summary stats.
2. `plot_histogram(file_path, column, bins=20)`
   Creates a histogram for a numeric column and returns it as an MCP image.
3. `train_random_forest(file_path, target, features=None)`
   Trains a baseline Random Forest pipeline with preprocessing, train/test split, metrics, and readable feature importances.
4. `scrape_table_from_wikipedia(url, table_index=0)`
   Extracts structured rows from a public Wikipedia table.
5. `fetch_web_json(url)`
   Fetches and previews JSON from a public endpoint.
6. `query_google_gemini(prompt, model=None)`
   Calls Gemini through the official Google GenAI SDK.

## Quick start

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and set your API keys and any optional server settings.

Important defaults:

| Variable | Description | Default |
|---|---|---|
| `MCP_SERVER_NAME` | Display name exposed to MCP hosts | `MCP-Powered Data Science Assistant` |
| `ALLOWED_DATA_DIR` | Directory allowed for CSV/TSV reads | `data` |
| `OUTPUT_DIR` | Directory for generated plots | `outputs/generated_plots` |
| `MAX_CSV_SIZE_MB` | Maximum dataset size in megabytes | `50` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `LOG_FILE` | Optional rotating log file path | empty |

### 3. Place datasets under the allowed directory

By default, datasets must live under `data/`. For example:

```text
data/churn_sample.csv
```

### 4. Run the MCP server

```bash
python src/mcp_data_science_assistant/server.py
```

### 5. Connect Claude Desktop

Use [claude_desktop_config.example.json](claude_desktop_config.example.json) as a template and point it at your local `server.py` path, then restart Claude Desktop.

### 6. Optional: run the terminal chat client

```bash
python src/mcp_data_science_assistant/chat_client.py
```

Example prompt:

```text
Analyze data/churn_sample.csv and tell me which feature matters most for churn.
```

## Security model

- Tool-based file reads are limited to `ALLOWED_DATA_DIR`.
- Directory traversal outside that directory is rejected.
- Only `.csv` and `.tsv` files are accepted.
- Oversized datasets are rejected using `MAX_CSV_SIZE_MB`.
- Wikipedia scraping stays limited to `wikipedia.org` URLs in this demo.

## Error handling

Tool failures return a simple structured response:

```json
{
  "error": "human-readable message",
  "error_type": "SpecificErrorClass"
}
```

That gives MCP hosts and the included terminal client a stable contract for reporting failures without crashing the full interaction.

## Example workflow

1. Call `load_csv("data/churn_sample.csv")`.
2. Inspect missing values and candidate features.
3. Call `train_random_forest(file_path="data/churn_sample.csv", target="Churn")`.
4. Read `top_feature_importances` such as `Contract = Month-to-month`.
5. Summarize the findings in plain English.

## Development notes

- `requirements.txt` is pinned for reproducible installs.
- `pyproject.toml` keeps runtime dependencies loose and adds optional dev extras.
- The matplotlib `Agg` backend is configured before importing pyplot for headless environments.

## Future work

- Add a small pytest suite for core helpers and security checks.
- Add richer model explainability such as SHAP or partial dependence plots.
- Add more transports, auth, and integration testing as the project grows.
