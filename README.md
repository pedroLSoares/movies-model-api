# Movies predictor — ML API

API to train and serve regression models that predict movie **ratings** from a dataset. It includes **local versioning** of datasets and models, **MLflow** integration (Tracking and Model Registry), and structured **logging** for latency and errors.

## Stack

- Python 3.13+
- [Poetry](https://python-poetry.org/) for dependency management
- [DuckDB](https://duckdb.org/) — Parquet aggregation and filtering during training load (`app/services/data_pipeline.py`)

```bash
cd ml_challenge
poetry install
```

## Run the API

From the project root:

```bash
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs (OpenAPI): **http://127.0.0.1:8000/docs**

### Docker (`docker-compose.yml`)

The image **does not** bundle `app/artifacts` (it is listed in `.dockerignore` so Parquet is not baked into the image). Mount your host folder into the container at the path used by `ARTIFACTS_DIR`: with `WORKDIR /app` in the Dockerfile, the package lives under `/app/app`, so artifacts must appear at **`/app/app/artifacts`** (e.g. `./app/artifacts:/app/app/artifacts`). Ensure Parquet files exist on the host (e.g. `app/artifacts/datasets/1.0.0/…`) before calling `POST /train` in Docker.

```bash
docker compose up --build
```

If training exits with **code 137** in the container, it is usually **out-of-memory (OOM)**. Increasing **`shm_size` in Compose does not replace process RAM** — it only changes `/dev/shm`. In **Docker Desktop**, raise **Resources → Memory** (e.g. 6–8 GB) if needed. The pipeline aggregates `ratings` in batches and reads `credits`/`keywords` via **streaming filtered by `id`** (and DuckDB where applicable) to avoid loading huge files entirely into RAM.

Useful environment variables:

| Variable | Description |
|----------|-------------|
| `LOG_LEVEL` | Log level (`INFO`, `DEBUG`, …). |
| `MLFLOW_*` | See [MLflow](#mlflow) below. |

---

## Artifacts and layout

Everything lives under **`app/artifacts/`** (next to the Python package `app`).

```
app/artifacts/
├── datasets/<dataset_version>/     # versioned snapshots (Parquet)
│   ├── manifest.json               # snapshot metadata (optional/manual)
│   ├── ratings.parquet
│   ├── links.parquet
│   ├── movies_metadata.parquet
│   ├── credits.parquet
│   └── keywords.parquet
└── models/<model_version>/         # one trained model per semver folder
    ├── model.joblib                # sklearn estimator (joblib)
    └── metadata.json               # hyperparameters, metrics, MLflow fields, etc.
```

By default MLflow writes outside this tree to **`mlruns/`** at the project root.

---

## Dataset versioning

- Each **version** is a folder **`app/artifacts/datasets/<version>/`** (e.g. `1.0.0`, `1.0.1`).
- The training pipeline (`app/services/data_pipeline.py`) expects these **Parquet** files in that folder:
  - **`ratings.parquet`**
  - **`links.parquet`**
  - **`movies_metadata.parquet`**
  - **`credits.parquet`**
  - **`keywords.parquet`**
- Join logic: mean `rating` per `movieId` → `links` → `movies` + `credits` + `keywords` (TMDB `id`) → join on `tmdbId`.
- **From CSV**: use the manual script `scripts/csv_to_parquet.py` (e.g. `poetry run python scripts/csv_to_parquet.py app/artifacts/datasets/1.0.0`) to generate `.parquet` files in the same folder.
- **New version**: copy or generate a new set of files under another semver folder (e.g. `1.0.1`) for reproducibility and comparison across runs.

If **no** dataset exists under `app/artifacts/datasets/`, training fails until at least one version contains the required Parquet files.

---

## Model versioning

- Each successful training run creates a new folder **`app/artifacts/models/<version>/`** with automatic semver (`1.0.0`, `1.0.1`, … — **patch** increment when names follow `M.m.p`; see `registry.next_model_version()` for numeric ordering, not lexicographic).
- Main files:
  - **`model.joblib`**: serialized model.
  - **`metadata.json`**: artifact version, `dataset_version` used, `estimator_id`, applied hyperparameters, validation metrics, feature names and dimension, MLflow fields when present.

**Inference** reads `metadata.json` for metadata (e.g. `dataset_version`, `feature_dim`); the **`POST /predict`** body follows the fixed contract `PREDICT_FEATURE_NAMES` in `data_pipeline.py` (`runtime`, `budget`, `release_month`, `director`, `cast`).

---

## Estimators (models)

Available algorithms are registered in **`app/services/estimators.py`** (`ESTIMATORS`). The default is **`hist_gradient_boosting`**.

- To train with another id: set **`estimator`** in `POST /train`.
- Valid hyperparameters are those accepted by `get_params(deep=False)` on the corresponding sklearn estimator; unknown keys are ignored.

---

## Exploratory analysis

Exploratory work (see also `data_analysis.ipynb`) informed how we shaped the training data and which errors we optimized for.

**Vote count and stability.** Much of the raw data is concentrated on movies with **fewer than 50 votes**. Using those rows makes the **mean rating very unstable** (high variance with little evidence). We therefore **prefer rows with more votes**: the pipeline keeps titles with **`vote_count > 50`** so averages are less volatile and more reliable for modeling.

**Target and relationships.** We did **not** observe strong linear or clear relationships between individual inputs and the target: ratings **cluster heavily around ~6** on average, which limits separability. **Balancing** or stratifying the target (e.g. equal mass per bin) would have required **dropping many rows**; that was **not** pursued because the **50-vote filter had already reduced** the usable sample size.

**Error metrics.** We relied mainly on **R²** and **MAE** to see how far predictions were from the true rating. The **practical goal** was to keep errors **around or below 0.5** (on the same scale as star ratings), i.e. predictions typically within half a point of the observed value.

---

## Model and feature engineering

The default estimator in the refined flow (notebook / model **v1.1.0**) is **`HistGradientBoostingRegressor`**, chosen for robustness and dense data after preprocessing.

### Feature rationale

To avoid **data leakage**, only **pre-release** information is used:

| Feature | Type | Description / treatment |
|---------|------|-------------------------|
| `runtime` | Numeric | Normalized with `StandardScaler`. |
| `log_budget` | Numeric | Log scale to reduce variance for blockbusters. |
| `release_month` | Categorical | `OneHotEncoder` for seasonality. |
| `director` | Ordinal / categorical | **Target encoding**: prestige score from history. |
| `cast_str` | Text | `CountVectorizer`: binary bag of words (top 100 actors). |

### Training and pipeline

- **Cleaning:** JSON-like strings via `ast.literal_eval` and name normalization (e.g. `Tom_Hanks`).
- **Target encoding:** for high-cardinality directors — historical mean score; new directors use the global mean.
- **Tuning:** hyperparameters aimed at RMSE stability and MAE reduction.

### Performance metrics (v1.1.0)

Results after feature refinement and hyperparameter tuning:

| Metric | Value |
|--------|--------|
| **MAE** (mean absolute error) | **0.56** (~0.5 stars on average) |
| **RMSE** | **0.71** |
| **R²** | **0.3048** |
| **Accuracy (1.0 margin)** | **84.64%** (predictions with error under 1 point) |

> **Note:** The HTTP service accepts hyperparameters via `POST /train` (`estimator` + `hyperparams`). The minimal pipeline in `app/services/data_pipeline.py` may differ from the notebook; align the model’s `metadata.json` with the `feature_names` order exported at training time.

---

## Endpoints

Example base URL: `http://127.0.0.1:8000` (adjust host/port as needed).

### `GET /list`

Lists versioned **models** and **datasets** (metadata from `metadata.json` / `manifest.json`).

**curl**

```bash
curl -s http://127.0.0.1:8000/list
```

**Response (200)** — example:

```json
{
  "models": [
    {
      "version": "1.0.0",
      "created_at": "2025-03-25T12:00:00+00:00",
      "dataset_version": "1.0.0",
      "estimator_id": "hist_gradient_boosting",
      "mlflow_run_id": "a1b2c3d4e5f6789..."
    }
  ],
  "datasets": [
    {
      "version": "1.0.0",
      "created_at": "2025-03-23T13:50:00+00:00"
    }
  ]
}
```

`models[]`: `version`, `created_at`, `dataset_version`, `estimator_id`, `mlflow_run_id`.  
`datasets[]`: `version`, `created_at`.

---

### `GET /metrics/{version}`

Returns **training** metrics and metadata for model **`version`** (`app/artifacts/models/<version>/`).

**curl**

```bash
curl -s http://127.0.0.1:8000/metrics/1.0.0
```

**Response (200)** — example:

```json
{
  "model_version": "1.0.0",
  "estimator": "hist_gradient_boosting",
  "training_metrics": {
    "val_mae": 0.75,
    "val_rmse": 0.97,
    "val_r2": 0.12,
    "n_train": 8000.0,
    "n_test": 2000.0
  },
  "dataset_used": "1.0.0",
  "mlflow_run": "a1b2c3d4e5f6789..."
}
```

**404** if the version does not exist or `metadata.json` is missing.

---

### `POST /train`

Body (`application/json`):

| Field | Type | Description |
|-------|------|-------------|
| `dataset_version` | string, optional | Dataset version under `app/artifacts/datasets/`. If omitted, uses the **latest** semver. |
| `estimator` | string | Estimator id (e.g. `hist_gradient_boosting`). |
| `hyperparams` | object | Sklearn hyperparameters filtered by the estimator. |

**curl**

```bash
curl -s -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_version": "1.0.0",
    "estimator": "hist_gradient_boosting",
    "hyperparams": {}
  }'
```

**Response (200)** — example:

```json
{
  "model_version": "1.0.1",
  "dataset_version": "1.0.0",
  "message": "Training finished and artifacts saved.",
  "metrics": {
    "val_mae": 0.75,
    "val_rmse": 0.97,
    "val_r2": 0.12,
    "n_train": 7238.0,
    "n_test": 1810.0
  },
  "mlflow": {
    "mlflow_run_id": "a1b2c3d4e5f6789...",
    "mlflow_model_uri": "runs:/a1b2c3d4e5f6789.../model",
    "mlflow_experiment_name": "ml-movies-challenge",
    "mlflow_tracking_uri": "/path/to/project/mlruns"
  }
}
```

`mlflow` may be `null` if tracking is disabled. **400** on validation errors (missing dataset, unknown estimator, etc.).

---

### `POST /predict`

| Field | Type | Description |
|-------|------|-------------|
| `features` | object | Fixed keys: `runtime`, `budget`, `release_month`, `director`, `cast` (list of names). The server derives `log_budget` and `cast_str`. |
| `model_version` | string, optional | Model to use; if omitted, uses the **latest** semver. |

**curl**

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "runtime": 120.0,
      "budget": 100000000,
      "release_month": 7,
      "director": "Christopher Nolan",
      "cast": ["Tom Hanks", "Julia Roberts"]
    },
    "model_version": "1.0.0"
  }'
```

**Response (200)** — example:

```json
{
  "model_version": "1.0.0",
  "prediction": 3.65,
  "metrics": {
    "latency_ms": 12.3456,
    "dataset_version": "1.0.0",
    "estimator_id": "hist_gradient_boosting",
    "feature_dim": 5,
    "mlflow_run_id": "a1b2c3d4e5f6789...",
    "absolute_error": null,
    "squared_error": null
  }
}
```

`absolute_error` / `squared_error` are only set if the request includes `actual_rating` (not implemented in the current handler — they stay `null`).

**400** if `features` is missing or prediction fails.

---

## MLflow

Integration lives in **`app/services/mlflow_integration.py`**. On each successful **training** run:

- A **run** is created in the configured experiment.
- **Tags** (e.g. `dataset_version`, `artifact_version`, `estimator_id`) and **parameters** (`hp_*`, etc.) are logged.
- Validation **metrics** are logged (same keys as in `metadata.json`).
- The model is stored with **`mlflow.sklearn.log_model`** under the run artifact `model`.
- **`mlflow.register_model`** may be called for the Model Registry when configured.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `<project>/mlruns` | Tracking backend URI (local file store or server, e.g. `http://127.0.0.1:5000`). |
| `MLFLOW_EXPERIMENT_NAME` | `ml-movies-challenge` | MLflow experiment name. |
| `MLFLOW_REGISTERED_MODEL_NAME` | *(empty)* | If set, registers the model in the **Model Registry** after training (requires a compatible backend; may fail with a pure file store without affecting local artifacts). |
| `MLFLOW_ENABLED` | `1` | `0` / `false` disables MLflow logging (local training still works). |

Typical fields also written to **`metadata.json`**: `mlflow_run_id`, `mlflow_model_uri`, `mlflow_experiment_name`, `mlflow_tracking_uri`, and optionally `mlflow_registered_model_name` / `mlflow_registered_model_version`.

**Local UI (optional):**

```bash
mlflow ui --backend-store-uri ./mlruns
```

(Use the same URI as `MLFLOW_TRACKING_URI` if not using the default.)

---

## Logging (observability)

- **`ml_api.access`**: HTTP middleware — method, path, `status`, `duration_ms`, `request_id` (**`X-Request-ID`** header).
- **`ml_api.api`**: business events (`train_ok` / `train_failed`, `predict_ok` / `predict_failed`, `metrics_served`, etc.).

Global level: **`LOG_LEVEL`** (e.g. `INFO`).

---

## Training vs prediction features

Training uses `log_budget` and `cast_str` derived from `budget` and `cast`/`crew` loaded from Parquet (`FEATURE_COLUMNS` in `data_pipeline.py`). **`POST /predict`** uses only `PREDICT_FEATURE_NAMES`:

`runtime`, `budget`, `release_month`, `director`, `cast`

Preprocessing (`ColumnTransformer` + `HistGradientBoostingRegressor` in `estimators.py`) applies `StandardScaler`, `OneHotEncoder`, `TargetEncoder`, and `CountVectorizer` to internal columns (`log_budget`, `cast_str`, etc.).

---

## Planned improvements

- **New Relic** — add the agent to monitor response times and application metrics beyond current HTTP logging (`duration_ms` in middleware).
- **Asynchronous training** — decouple long-running `POST /train` work via messaging (task queue / broker), with progress state and completion notifications.
- **Remote training data** — fetch dataset snapshots from an external service (e.g. **AWS S3**) instead of relying only on local files under `app/artifacts/datasets/`.

---

## `pandas` dependency

The project pins **`pandas` 2.x** because current **MLflow** releases still require `pandas < 3`. When upgrading, align versions in `pyproject.toml`.
