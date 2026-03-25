from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """Training parameters; extend as the model evolves."""

    dataset_version: str | None = Field(
        default=None,
        description="Dataset version to use; if omitted, uses the latest.",
    )
    estimator: str = Field(
        default="hist_gradient_boosting",
        description="Estimator id (see `app.services.estimators.ESTIMATORS`).",
    )
    hyperparams: dict[str, Any] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    model_version: str
    dataset_version: str
    message: str
    metrics: dict[str, float] | None = Field(
        default=None,
        description="Validation metrics (e.g. MAE/RMSE) after training.",
    )
    mlflow: dict[str, Any] | None = Field(
        default=None,
        description="MLflow metadata (run_id, model_uri, etc.) when tracking is enabled.",
    )
