from __future__ import annotations

import logging
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from app.config import (
    MLFLOW_ENABLED,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)
from app.services.estimators import EstimatorSpec

logger = logging.getLogger(__name__)


def configure() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_training_run(
    *,
    model: Any,
    spec: EstimatorSpec,
    dataset_version: str,
    artifact_version: str,
    metrics: dict[str, float],
    applied_hyperparams: dict[str, Any],
) -> dict[str, Any]:
    """
    Log an MLflow run tied to the local version in `app/artifacts/models/<artifact_version>/`.

    Returns fields to merge into `metadata.json` (e.g. mlflow_run_id, mlflow_model_uri).
    """
    if not MLFLOW_ENABLED:
        return {}

    extra: dict[str, Any] = {}
    try:
        configure()
        with mlflow.start_run(run_name=f"artifact-{artifact_version}") as run:
            mlflow.set_tag("dataset_version", dataset_version)
            mlflow.set_tag("artifact_version", artifact_version)
            mlflow.set_tag("estimator_id", spec.id)
            mlflow.set_tag("estimator_class", spec.sklearn_name)

            mlflow.log_param("dataset_version", dataset_version)
            mlflow.log_param("estimator_id", spec.id)
            for key, val in applied_hyperparams.items():
                mlflow.log_param(f"hp_{key}", val)

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(model, name="model")

            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            extra["mlflow_run_id"] = run_id
            extra["mlflow_experiment_name"] = MLFLOW_EXPERIMENT_NAME
            extra["mlflow_model_uri"] = model_uri
            extra["mlflow_tracking_uri"] = MLFLOW_TRACKING_URI

            if MLFLOW_REGISTERED_MODEL_NAME:
                try:
                    mv = mlflow.register_model(
                        model_uri, MLFLOW_REGISTERED_MODEL_NAME
                    )
                    extra["mlflow_registered_model_name"] = MLFLOW_REGISTERED_MODEL_NAME
                    extra["mlflow_registered_model_version"] = int(mv.version)
                except Exception as reg_err:
                    logger.warning(
                        "MLflow Model Registry unavailable or registration failed: %s",
                        reg_err,
                    )
    except Exception as e:
        logger.exception("MLflow Tracking failed (local training artifacts are unchanged): %s", e)
        return {}

    return extra
