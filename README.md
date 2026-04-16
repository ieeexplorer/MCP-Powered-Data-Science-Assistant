# MCP-Powered Data Science Assistant

A portfolio-ready project that turns an LLM into a **conversational data analyst** using an **MCP server**.

Instead of building isolated notebooks or one-off scripts, this project packages real data tasks as MCP tools:

- load a dataset
- inspect quality issues
- generate a chart
- train a baseline ML model
- scrape a public table
- call Gemini for extra reasoning

That means a host like **Claude Desktop** can use your tools step by step and answer questions such as:

> “Analyze the churn dataset and tell me which feature matters most.”

---

## Why this project is strong for your profile

This project combines:
- **MCP server engineering**
- **Python data science**
- **baseline machine learning**
- **API/web data ingestion**
- **LLM tooling**
- **clean, modular architecture**

It shows that you can move from:
- single scripts  
to
- a reusable AI-tooling system

---

## Architecture

```text
User
  │
  ▼
Claude Desktop or Custom Terminal Client
  │
  ▼
MCP Server
  ├── load_csv(file_path)
  ├── plot_histogram(file_path, column, bins)
  ├── train_random_forest(file_path, target, features)
  ├── scrape_table_from_wikipedia(url, table_index)
  ├── fetch_web_json(url)
  └── query_google_gemini(prompt)
```

---

## Project structure

```text
mcp-powered-data-science-assistant/
├── src/
│   └── mcp_data_science_assistant/
│       ├── __init__.py
│       ├── server.py
│       └── chat_client.py
├── data/
│   └── churn_sample.csv
├── docs/
│   └── architecture.md
├── examples/
│   └── demo_queries.md
├── outputs/
├── .env.example
├── .gitignore
├── claude_desktop_config.example.json
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Main tools

### 1) `load_csv(file_path)`
Returns:
- row/column count
- column names
- data types
- missing values
- duplicate count
- preview rows
- numeric summary

### 2) `plot_histogram(file_path, column, bins=20)`
Creates a histogram for a numeric column and returns an image to the MCP host.

### 3) `train_random_forest(file_path, target, features=None)`
Builds a baseline Random Forest pipeline with:
- missing-value handling
- one-hot encoding for categorical fields
- train/test split
- feature importances
- model metrics

It auto-detects:
- **classification** for label-like targets
- **regression** for continuous numeric targets

### 4) `scrape_table_from_wikipedia(url, table_index=0)`
Uses `pandas.read_html()` to extract a public table from Wikipedia and return structured records.

### 5) `fetch_web_json(url)`
Fetches JSON from a public API endpoint and returns a clean preview.

### 6) `query_google_gemini(prompt, model=None)`
Uses the **official Google GenAI SDK** if `GEMINI_API_KEY` is present.

---

## Quick start

## 1. Create a virtual environment

### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Add environment variables

Create a `.env` file from `.env.example`.

Example:

```env
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash
ANTHROPIC_API_KEY=your-anthropic-api-key
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

---

## 3. Run the MCP server

```bash
python src/mcp_data_science_assistant/server.py
```

---

## 4. Connect Claude Desktop

Open your Claude Desktop MCP config and add something like:

```json
{
  "mcpServers": {
    "mcp-data-science-assistant": {
      "command": "python",
      "args": [
        "C:/path/to/mcp-powered-data-science-assistant/src/mcp_data_science_assistant/server.py"
      ]
    }
  }
}
```

Then fully restart Claude Desktop.

---

## 5. Optional: run the custom terminal chat client

```bash
python src/mcp_data_science_assistant/chat_client.py
```

Then try:

```text
Analyze data/churn_sample.csv and tell me which feature matters most for churn.
```

---

## Example workflow

### User
Analyze `data/churn_sample.csv` and tell me which feature matters most.

### LLM host
1. calls `load_csv("data/churn_sample.csv")`
2. calls `train_random_forest(file_path="data/churn_sample.csv", target="Churn")`
3. reads `top_feature_importances`
4. answers in plain English

---

## Simple explanation of the ML part

Think of the model as a smart pattern finder.

It learns from old examples:
- customers who stayed
- customers who left

Then it checks which columns helped the most in separating those two groups.

Typical important features in churn-style datasets are things like:
- tenure
- monthly charges
- contract type
- total charges

---

## Good interview talking points

### What problem does it solve?
It lets an LLM perform real data tasks through controlled tools instead of hallucinating from raw chat alone.

### Why MCP?
Because it makes your tools reusable across multiple hosts instead of locking them into one notebook or one UI.

### Why Random Forest first?
Because it is strong as a baseline, easy to explain, and gives feature importance out of the box.

### Why this is better than a notebook-only project?
Because it separates:
- conversation layer
- tool execution layer
- data/ML logic

That is closer to production design.

---

## Best CV bullets

- Built an **MCP-based data science assistant** in Python that exposed dataset inspection, plotting, ML training, web scraping, and Gemini reasoning as reusable tools for LLM hosts.
- Developed a **FastMCP / MCP Python SDK server** that enabled Claude Desktop or a custom chat client to execute structured analytics workflows over CSV data.
- Implemented a **Random Forest ML pipeline** with preprocessing, missing-value handling, one-hot encoding, model evaluation, and feature importance reporting for conversational analysis.
- Added **web-data capabilities** using API ingestion and Wikipedia table extraction, extending the assistant beyond local files into public structured data sources.
- Designed the project using a **modular architecture** separating MCP tools, ML logic, host integration, and environment-based secret management.

---

## Tips and tricks

| Situation | Best move |
|---|---|
| You want the host to “understand the data first” | Always call `load_csv()` before training |
| Column contains words like Yes/No | Treat it as classification |
| Target is continuous numeric | Use regression |
| Chart request fails | Check if the column is numeric |
| Bad model performance | Start by checking missing values, leakage, and class imbalance |
| Wikipedia extraction breaks | Try a different `table_index` |
| Gemini tool fails | Check `GEMINI_API_KEY` in `.env` |

---

## Easy demo prompts

- Analyze `data/churn_sample.csv` and summarize the dataset health.
- Train a model on `data/churn_sample.csv` with `Churn` as the target.
- Plot a histogram of `MonthlyCharges`.
- Scrape the first table from a Wikipedia page and summarize the rows.
- Ask Gemini to explain model metrics like accuracy, precision, recall, and F1 in plain English.

---

## Future upgrades

- SHAP explanations
- correlation heatmaps
- SQL database connectors
- anomaly detection tools
- forecast tools
- Streamlit dashboard
- remote MCP transport over Streamable HTTP
- authentication and access control
- test suite and CI pipeline

---

## One-sentence pitch

**MCP-Powered Data Science Assistant** is a Python project that turns an LLM into a safe, tool-using data analyst by exposing structured analytics and ML capabilities through the Model Context Protocol.
