import logging
import time
from contextlib import asynccontextmanager
from typing import Any
from app.services.data_analysis import run_drift_validation

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles

from app.http_logging import RequestLoggingMiddleware, configure_logging
from app.dto import (
    DatasetInfo,
    ListResponse,
    ModelInfo,
    PredictionMetrics,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
)
from app.services import ml_core, registry

API_LOG = logging.getLogger("ml_api.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    yield


app = FastAPI(
    title="ML API",
    description="API with versioned models and datasets; training runs are logged to MLflow (Tracking). ",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/reports", StaticFiles(directory="app/artifacts/models"), name="reports")

@app.post("/monitor/drift/{version}")
async def monitor_drift(version: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_drift_validation, version)
    return {"message": "Drift validation started in background"}


@app.get("/metrics/{version}")
def get_model_metrics(version: str):
    """Return training metrics and performance for a specific model version."""
    try:
        meta = registry.read_model_metadata(version)
        if not meta:
            API_LOG.warning("metrics_not_found version=%s", version)
            raise HTTPException(
                status_code=404, detail="Model version not found."
            )
        API_LOG.info("metrics_served version=%s", version)
        return {
            "model_version": version,
            "estimator": meta.get("estimator_id"),
            "training_metrics": meta.get("metrics"),
            "dataset_used": meta.get("dataset_version"),
            "mlflow_run": meta.get("mlflow_run_id"),
        }
    except HTTPException:
        raise
    except Exception as e:
        API_LOG.exception("metrics_error version=%s err=%s", version, e)
        raise HTTPException(
            status_code=404, detail="Model version not found."
        ) from e


@app.get("/list", response_model=ListResponse)
def list_artifacts() -> ListResponse:
    models = [ModelInfo(**m) for m in registry.list_model_versions()]
    datasets = [DatasetInfo(**d) for d in registry.list_dataset_versions()]
    return ListResponse(models=models, datasets=datasets)


@app.post("/train", response_model=TrainResponse)
def train_endpoint(body: TrainRequest) -> TrainResponse:
    t0 = time.perf_counter()
    API_LOG.info(
        "train_started dataset_version=%s estimator=%s",
        body.dataset_version or "(latest)",
        body.estimator,
    )
    try:
        model_version, dataset_version, metrics, mlflow_info = ml_core.train(
            body.dataset_version,
            body.hyperparams,
            estimator_id=body.estimator,
        )
    except ValueError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        API_LOG.warning(
            "train_failed duration_ms=%.3f estimator=%s detail=%s",
            elapsed_ms,
            body.estimator,
            e,
        )
        raise HTTPException(status_code=400, detail=str(e)) from e

    elapsed_ms = (time.perf_counter() - t0) * 1000
    API_LOG.info(
        "train_finished duration_ms=%.3f model_version=%s dataset_version=%s estimator=%s val_mae=%s",
        elapsed_ms,
        model_version,
        dataset_version,
        body.estimator,
        (metrics or {}).get("val_mae"),
    )
    return TrainResponse(
        model_version=model_version,
        dataset_version=dataset_version,
        message="Training finished and artifacts saved.",
        metrics=metrics,
        mlflow=mlflow_info or None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(body: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    try:
        prediction, v = ml_core.predict(
            body.features, body.model_version
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
    except ValueError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        API_LOG.warning(
            "predict_failed duration_ms=%.3f model_version=%s detail=%s",
            elapsed_ms,
            body.model_version,
            e,
        )
        raise HTTPException(status_code=400, detail=str(e)) from e

    pred_val = prediction.item() if hasattr(prediction, "item") else prediction

    meta = registry.read_model_metadata(v)
    fd = meta.get("feature_dim")
    metrics_kw: dict[str, Any] = {
        "latency_ms": round(infer_ms, 4),
        "dataset_version": meta.get("dataset_version"),
        "estimator_id": meta.get("estimator_id"),
        "feature_dim": int(fd) if fd is not None else None,
        "mlflow_run_id": meta.get("mlflow_run_id"),
    }

    API_LOG.info(
        "predict_ok model_version=%s latency_ms=%.4f",
        v, infer_ms
    )

    return PredictResponse(
        model_version=v,
        prediction=pred_val,
        metrics=PredictionMetrics(**metrics_kw),
    )


app.add_middleware(RequestLoggingMiddleware)
