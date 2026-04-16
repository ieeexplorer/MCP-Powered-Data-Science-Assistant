from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP, Image
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

mcp = FastMCP(
    name="MCP-Powered Data Science Assistant",
    instructions=(
        "Use the available tools to inspect datasets, create charts, train baseline machine "
        "learning models, scrape public Wikipedia tables, and call Gemini for reasoning-heavy tasks."
    ),
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "generated_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_path(file_path: str) -> Path:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _load_dataframe(file_path: str) -> pd.DataFrame:
    path = _resolve_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("Only .csv and .tsv files are supported by this demo server.")
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=separator)


def _infer_task(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 10:
        return "regression"
    return "classification"


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
        else:
            last_step = transformer
        if hasattr(last_step, "get_feature_names_out"):
            generated = last_step.get_feature_names_out(columns)
            names.extend([str(item) for item in generated])
        else:
            names.extend([str(column) for column in columns])
    return names


@mcp.resource("guide://capabilities")
def capabilities_guide() -> str:
    """Human-readable overview of the server's tools."""
    return """
MCP-Powered Data Science Assistant

Main tools:
1. load_csv(file_path)
   - Returns shape, columns, dtypes, missing values, duplicates, and summary statistics.

2. plot_histogram(file_path, column, bins)
   - Returns a PNG chart for a numeric column.

3. train_random_forest(file_path, target, features=None, test_size=0.2, random_state=42)
   - Trains a baseline Random Forest model and returns metrics and feature importances.

4. scrape_table_from_wikipedia(url, table_index=0, max_rows=20)
   - Extracts a Wikipedia table into JSON-like records.

5. fetch_web_json(url)
   - Fetches and previews JSON from a public API.

6. query_google_gemini(prompt, model=None)
   - Uses the official Google GenAI SDK when GEMINI_API_KEY is available.

Best practice:
- Start with load_csv
- Inspect the target column
- Train a model
- Then ask the LLM to explain the results in plain English
""".strip()


@mcp.prompt(title="Dataset analysis starter")
def dataset_analysis_prompt(file_path: str, target: str) -> str:
    """Starter prompt template for a host LLM."""
    return f"""
Analyze the dataset at {file_path}.

Suggested tool sequence:
1. Call load_csv(file_path="{file_path}")
2. Understand missing values and candidate features
3. Call train_random_forest(file_path="{file_path}", target="{target}")
4. Summarize the top feature importances in simple language
5. Highlight 2 practical business actions based on the model output
""".strip()


@mcp.tool()
async def load_csv(file_path: str, ctx: Context | None = None) -> dict[str, Any]:
    """Load a CSV/TSV file and return a dataset health summary."""
    if ctx:
        await ctx.info(f"Loading dataset: {file_path}")
    df = _load_dataframe(file_path)

    summary = {
        "resolved_path": str(_resolve_path(file_path)),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {column: int(value) for column, value in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "preview": df.head(5).fillna("").to_dict(orient="records"),
        "numeric_summary": df.describe(include="number").round(3).fillna("").to_dict(),
    }
    return summary


@mcp.tool()
async def plot_histogram(
    file_path: str,
    column: str,
    bins: int = 20,
    ctx: Context | None = None,
) -> Image:
    """Create a histogram for a numeric column and return it as an MCP image."""
    if ctx:
        await ctx.info(f"Creating histogram for {column} from {file_path}")
    df = _load_dataframe(file_path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")

    fig, ax = plt.subplots(figsize=(8, 5))
    df[column].dropna().plot(kind="hist", bins=bins, ax=ax)
    ax.set_title(f"Histogram: {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()

    path = OUTPUT_DIR / f"{column}_histogram.png"
    fig.savefig(path, format="png", dpi=150)
    image_bytes = io.BytesIO()
    fig.savefig(image_bytes, format="png", dpi=150)
    plt.close(fig)

    if ctx:
        await ctx.info(f"Histogram saved to {path}")
    return Image(data=image_bytes.getvalue(), format="png")


@mcp.tool()
async def train_random_forest(
    file_path: str,
    target: str,
    features: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Train a baseline Random Forest and return evaluation metrics plus feature importances."""
    if ctx:
        await ctx.info(f"Training random forest for target '{target}' from {file_path}")
    df = _load_dataframe(file_path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    working_df = df.copy()
    if features:
        missing_features = [feature for feature in features if feature not in working_df.columns]
        if missing_features:
            raise ValueError(f"Features not found: {missing_features}")
        columns_to_keep = list(dict.fromkeys(features + [target]))
        working_df = working_df[columns_to_keep]

    working_df = working_df.dropna(subset=[target])
    X = working_df.drop(columns=[target])
    y = working_df[target]

    if X.empty:
        raise ValueError("No feature columns remain after preprocessing.")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    task = _infer_task(y)
    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
        )
        stratify = y if y.nunique() > 1 else None
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
        )
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    fitted_preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    fitted_model = pipeline.named_steps["model"]
    transformed_feature_names = _feature_names(fitted_preprocessor)

    importances = getattr(fitted_model, "feature_importances_", [])
    importance_pairs = [
        {"feature": name, "importance": round(float(score), 6)}
        for name, score in sorted(
            zip(transformed_feature_names, importances),
            key=lambda item: item[1],
            reverse=True,
        )
    ][:10]

    if task == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision_macro": round(float(precision_score(y_test, predictions, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_test, predictions, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4),
        }
    else:
        metrics = {
            "r2": round(float(r2_score(y_test, predictions)), 4),
            "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
            "rmse": round(float(root_mean_squared_error(y_test, predictions)), 4),
        }

    return {
        "resolved_path": str(_resolve_path(file_path)),
        "task": task,
        "target": target,
        "rows_used": int(working_df.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "features_used": list(X.columns),
        "metrics": metrics,
        "top_feature_importances": importance_pairs,
    }


@mcp.tool()
async def scrape_table_from_wikipedia(
    url: str,
    table_index: int = 0,
    max_rows: int = 20,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Scrape a public Wikipedia table and return structured rows."""
    if "wikipedia.org" not in url:
        raise ValueError("For safety, this demo only supports wikipedia.org URLs.")

    if ctx:
        await ctx.info(f"Scraping Wikipedia table {table_index} from {url}")

    tables = pd.read_html(url)
    if table_index < 0 or table_index >= len(tables):
        raise ValueError(f"table_index must be between 0 and {len(tables) - 1}")

    table = tables[table_index]
    table.columns = [str(column) for column in table.columns]
    preview = table.head(max_rows).fillna("").to_dict(orient="records")

    return {
        "url": url,
        "table_index": table_index,
        "available_tables": len(tables),
        "columns": list(table.columns),
        "row_count": int(table.shape[0]),
        "preview": preview,
    }


@mcp.tool()
async def fetch_web_json(
    url: str,
    timeout_seconds: int = 20,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Fetch JSON from a public API endpoint."""
    if ctx:
        await ctx.info(f"Fetching JSON from {url}")

    response = requests.get(url, timeout=timeout_seconds, headers={"User-Agent": "mcp-ds-assistant/0.1"})
    response.raise_for_status()
    data = response.json()

    if isinstance(data, list):
        preview = data[:5]
        item_count = len(data)
    elif isinstance(data, dict):
        preview = dict(list(data.items())[:10])
        item_count = len(data)
    else:
        preview = data
        item_count = 1

    return {
        "url": url,
        "status_code": response.status_code,
        "item_count": item_count,
        "preview": preview,
    }


@mcp.tool()
async def query_google_gemini(
    prompt: str,
    model: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Call Gemini using the official Google GenAI SDK."""
    if ctx:
        await ctx.info("Calling Gemini")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file first.")

    chosen_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=chosen_model,
        contents=prompt,
    )

    return {
        "model": chosen_model,
        "text": response.text,
    }


def main() -> None:
    """Run the MCP server over STDIO by default."""
    mcp.run()


if __name__ == "__main__":
    main()
