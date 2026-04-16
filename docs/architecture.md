# Architecture

```text
User
  │
  ▼
Claude Desktop or Custom Terminal Client
  │
  ▼
MCP Server (FastMCP / official MCP Python SDK)
  ├── load_csv
  ├── plot_histogram
  ├── train_random_forest
  ├── scrape_table_from_wikipedia
  ├── fetch_web_json
  └── query_google_gemini
          │
          ├── Local files / CSV datasets
          ├── Public web pages / APIs
          └── Gemini API
```

## Design choices

- **MCP server first**: the logic is packaged as tools, not as one-off scripts.
- **Interview-friendly ML**: Random Forest gives a practical, explainable baseline.
- **LLM-host agnostic**: you can plug this into Claude Desktop or the included terminal client.
- **Safe defaults**: Wikipedia scraping is restricted to wikipedia.org in the demo project.
- **STDIO-safe logging**: the server logs to stderr rather than stdout.
