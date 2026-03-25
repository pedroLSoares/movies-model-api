from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    features: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Values keyed by column name (e.g. runtime, budget, release_month, director, "
            "cast). The server derives `log_budget` from `budget` and `cast_str` from "
            "`cast` (list of names), matching training."
        ),
    )
    model_version: str | None = Field(
        default=None,
        description="Model version to use; if omitted, uses the latest registered version.",
    )

    @model_validator(mode="after")
    def require_features(self) -> Self:
        if self.features is None:
            raise ValueError("Provide `features`.")
        return self


class PredictionMetrics(BaseModel):
    """Per-request metrics for comparing versions (e.g. 1.0.0 vs 1.0.1) in A/B tests."""

    latency_ms: float = Field(
        ...,
        description="Server-side inference time (load model + predict), in ms.",
    )
    dataset_version: str | None = Field(
        default=None,
        description="Dataset snapshot the model was trained on.",
    )
    estimator_id: str | None = None
    feature_dim: int | None = None
    mlflow_run_id: str | None = None
    absolute_error: float | None = Field(
        default=None,
        description="Only if `actual_rating` was sent: |y_hat - y|.",
    )
    squared_error: float | None = Field(
        default=None,
        description="Only if `actual_rating` was sent: (y_hat - y)².",
    )


class PredictResponse(BaseModel):
    model_version: str
    prediction: Any
    metrics: PredictionMetrics
