"""MCP server exposing dataset analysis, plotting, and helper tools."""

from __future__ import annotations

import io
import logging
import logging.handlers
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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

try:
    from mcp_data_science_assistant.runtime import get_workspace_root, resolve_workspace_path
except ModuleNotFoundError:
    from runtime import get_workspace_root, resolve_workspace_path

load_dotenv()

P = ParamSpec("P")
T = TypeVar("T")

PROJECT_ROOT = get_workspace_root()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "")
ALLOWED_DATA_DIR = resolve_workspace_path(os.getenv("ALLOWED_DATA_DIR", "data"))
OUTPUT_DIR = resolve_workspace_path(os.getenv("OUTPUT_DIR", "outputs/generated_plots"))
MAX_CSV_SIZE_MB = int(os.getenv("MAX_CSV_SIZE_MB", "50"))
MAX_CSV_SIZE_BYTES = MAX_CSV_SIZE_MB * 1024 * 1024

ALLOWED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("mcp-data-science-assistant")
logger.setLevel(LOG_LEVEL)
logger.handlers.clear()
logger.propagate = False

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(stderr_handler)

if LOG_FILE:
    resolved_log_file = resolve_workspace_path(LOG_FILE)
    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        resolved_log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(file_handler)


class MCPServerError(Exception):
    """Base exception for MCP tool failures."""


class FileAccessError(MCPServerError):
    """Raised when a file is outside the allowed directory or exceeds size limits."""


class DataProcessingError(MCPServerError):
    """Raised when loading or preparing a dataset fails."""


class ModelTrainingError(MCPServerError):
    """Raised when the model pipeline fails to train or evaluate."""


class ErrorResponse(TypedDict):
    error: str
    error_type: str


class LoadCSVOutput(TypedDict):
    resolved_path: str
    shape: dict[str, int]
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    duplicate_rows: int
    preview: list[dict[str, Any]]
    numeric_summary: dict[str, dict[str, float]]


class FeatureImportance(TypedDict):
    feature: str
    importance: float


class TrainRFOutput(TypedDict):
    resolved_path: str
    task: str
    target: str
    rows_used: int
    train_rows: int
    test_rows: int
    features_used: list[str]
    metrics: dict[str, float]
    top_feature_importances: list[FeatureImportance]


def _error_response(error: Exception | str, error_type: str) -> ErrorResponse:
    message = str(error) if isinstance(error, Exception) else error
    return {"error": message, "error_type": error_type}


def handle_tool_errors(
    func: Callable[P, Awaitable[T]],
) -> Callable[P, Awaitable[T | ErrorResponse]]:
    """Return structured tool errors instead of bubbling raw exceptions to the host."""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | ErrorResponse:
        try:
            return await func(*args, **kwargs)
        except MCPServerError as error:
            logger.error("Tool error in %s: %s", func.__name__, error)
            return _error_response(error, error.__class__.__name__)
        except Exception as error:
            logger.exception("Unexpected error in %s", func.__name__)
            return _error_response(f"Unexpected error: {error}", "InternalError")

    return wrapper


mcp = FastMCP(
    name=os.getenv("MCP_SERVER_NAME", "MCP-Powered Data Science Assistant"),
    instructions=(
        "Use the available tools to inspect datasets, create charts, train baseline machine "
        "learning models, scrape public Wikipedia tables, and call Gemini for reasoning-heavy tasks. "
        "Dataset file access is restricted to the configured data directory."
    ),
)


def _resolve_and_validate_path(file_path: str) -> Path:
    path = resolve_workspace_path(file_path)

    try:
        path.relative_to(ALLOWED_DATA_DIR)
    except ValueError as error:
        raise FileAccessError(
            f"Access denied: '{path}' is outside allowed data directory '{ALLOWED_DATA_DIR}'."
        ) from error

    if not path.exists():
        raise FileAccessError(f"File not found: {path}")
    if path.stat().st_size > MAX_CSV_SIZE_BYTES:
        raise FileAccessError(
            f"File size ({path.stat().st_size / 1024 / 1024:.1f} MB) exceeds limit ({MAX_CSV_SIZE_MB} MB)."
        )
    return path


def _load_dataframe(file_path: str) -> pd.DataFrame:
    path = _resolve_and_validate_path(file_path)
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise DataProcessingError("Only .csv and .tsv files are supported by this demo server.")
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        return pd.read_csv(path, sep=separator)
    except Exception as error:
        raise DataProcessingError(f"Failed to read file: {error}") from error


def _infer_task(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 10:
        return "regression"
    return "classification"


def _simplify_feature_names(names: list[str]) -> list[str]:
    simplified: list[str] = []
    for name in names:
        if name.startswith("num__"):
            simplified.append(name.removeprefix("num__"))
            continue
        if name.startswith("cat__"):
            remainder = name.removeprefix("cat__")
            if "_" in remainder:
                column, encoded_value = remainder.split("_", 1)
                simplified.append(f"{column} = {encoded_value}")
            else:
                simplified.append(remainder)
            continue
        if "__" in name:
            simplified.append(name.split("__", 1)[1])
            continue
        simplified.append(name)
    return simplified


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
    return _simplify_feature_names(names)


@mcp.resource("guide://capabilities")
def capabilities_guide() -> str:
    """Human-readable overview of the server's tools."""
    return """
MCP-Powered Data Science Assistant

Main tools:
1. load_csv(file_path)
   - Returns shape, columns, dtypes, missing values, duplicates, and summary statistics.
   - Only reads files from the configured data directory.

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

Error handling:
- Tool failures return a structured payload with error and error_type fields.
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
@handle_tool_errors
async def load_csv(file_path: str, ctx: Context | None = None) -> LoadCSVOutput:
    """Load a CSV/TSV file from the allowed data directory and return a dataset health summary."""
    if ctx:
        await ctx.info(f"Loading dataset: {file_path}")
    logger.info("Tool called: load_csv('%s')", file_path)
    df = _load_dataframe(file_path)

    summary: LoadCSVOutput = {
        "resolved_path": str(_resolve_and_validate_path(file_path)),
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
@handle_tool_errors
async def plot_histogram(
    file_path: str,
    column: str,
    bins: int = 20,
    ctx: Context | None = None,
) -> Image:
    """Create a histogram for a numeric column and return it as an MCP image."""
    if ctx:
        await ctx.info(f"Creating histogram for {column} from {file_path}")
    logger.info("Tool called: plot_histogram('%s', '%s', bins=%s)", file_path, column, bins)
    df = _load_dataframe(file_path)

    if column not in df.columns:
        raise DataProcessingError(f"Column '{column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise DataProcessingError(f"Column '{column}' is not numeric.")

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
    logger.info("Histogram saved to %s", path)
    return Image(data=image_bytes.getvalue(), format="png")


@mcp.tool()
@handle_tool_errors
async def train_random_forest(
    file_path: str,
    target: str,
    features: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    ctx: Context | None = None,
) -> TrainRFOutput:
    """Train a baseline Random Forest and return evaluation metrics plus feature importances."""
    if ctx:
        await ctx.info(f"Training random forest for target '{target}' from {file_path}")
    logger.info(
        "Tool called: train_random_forest('%s', target='%s', features=%s)",
        file_path,
        target,
        features,
    )
    df = _load_dataframe(file_path)

    if target not in df.columns:
        raise DataProcessingError(f"Target column '{target}' not found.")

    working_df = df.copy()
    if features:
        missing_features = [feature for feature in features if feature not in working_df.columns]
        if missing_features:
            raise DataProcessingError(f"Features not found: {missing_features}")
        columns_to_keep = list(dict.fromkeys(features + [target]))
        working_df = working_df[columns_to_keep]

    working_df = working_df.dropna(subset=[target])
    X = working_df.drop(columns=[target])
    y = working_df[target]

    if X.empty:
        raise DataProcessingError("No feature columns remain after preprocessing.")

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
    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
    except Exception as error:
        raise ModelTrainingError(f"Failed to train model: {error}") from error

    fitted_preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    fitted_model = pipeline.named_steps["model"]
    transformed_feature_names = _feature_names(fitted_preprocessor)

    importances = getattr(fitted_model, "feature_importances_", [])
    importance_pairs: list[FeatureImportance] = [
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

    output: TrainRFOutput = {
        "resolved_path": str(_resolve_and_validate_path(file_path)),
        "task": task,
        "target": target,
        "rows_used": int(working_df.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "features_used": list(X.columns),
        "metrics": metrics,
        "top_feature_importances": importance_pairs,
    }
    logger.info("Model training completed. Task: %s, Metrics: %s", task, metrics)
    return output


@mcp.tool()
@handle_tool_errors
async def scrape_table_from_wikipedia(
    url: str,
    table_index: int = 0,
    max_rows: int = 20,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Scrape a public Wikipedia table and return structured rows."""
    if "wikipedia.org" not in url:
        raise MCPServerError("For safety, this demo only supports wikipedia.org URLs.")

    if ctx:
        await ctx.info(f"Scraping Wikipedia table {table_index} from {url}")
    logger.info("Tool called: scrape_table_from_wikipedia('%s', table_index=%s)", url, table_index)

    tables = pd.read_html(url)
    if table_index < 0 or table_index >= len(tables):
        raise MCPServerError(f"table_index must be between 0 and {len(tables) - 1}")

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
@handle_tool_errors
async def fetch_web_json(
    url: str,
    timeout_seconds: int = 20,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Fetch JSON from a public API endpoint."""
    if ctx:
        await ctx.info(f"Fetching JSON from {url}")
    logger.info("Tool called: fetch_web_json('%s')", url)

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
@handle_tool_errors
async def query_google_gemini(
    prompt: str,
    model: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Call Gemini using the official Google GenAI SDK."""
    if ctx:
        await ctx.info("Calling Gemini")
    logger.info("Tool called: query_google_gemini")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise MCPServerError("GEMINI_API_KEY is missing. Add it to your .env file first.")

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
    logger.info("Starting MCP Data Science Assistant server...")
    mcp.run()


if __name__ == "__main__":
    main()
