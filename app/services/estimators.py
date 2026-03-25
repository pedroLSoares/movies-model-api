from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder


def _cast_text_to_1d(X):
    a = np.asarray(X)
    if a.ndim == 2 and a.shape[1] == 1:
        return a.ravel()
    return a


NUMERIC_FEATURES: list[str] = ["runtime", "log_budget"]
NOMINAL_CAT_COLUMNS: list[str] = ["release_month"]
ORDINAL_FEATURES: list[str] = ["director"]
TEXT_COLUMN: str = "cast_str"


def _make_full_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        n_jobs=1,
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat_simple",
                OneHotEncoder(handle_unknown="ignore"),
                NOMINAL_CAT_COLUMNS,
            ),
            ("cat", TargetEncoder(smooth=50.0), ORDINAL_FEATURES),
            (
                "cast",
                Pipeline(
                    [
                        (
                            "to1d",
                            FunctionTransformer(_cast_text_to_1d, validate=False),
                        ),
                        (
                            "vect",
                            CountVectorizer(max_features=100, binary=True),
                        ),
                    ]
                ),
                [TEXT_COLUMN],
            ),
        ],
        sparse_threshold=0,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "hgb",
                HistGradientBoostingRegressor(
                    random_state=42,
                    max_iter=500,
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class EstimatorSpec:
    id: str
    sklearn_name: str
    fixed_params: dict[str, Any]
    estimator_class: type[Any] | None = None
    factory: Callable[[], Any] | None = None

    def build(self, hyperparams: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """
        Build the estimator with `fixed_params` overridden by valid hyperparameters.

        Returns the unfitted estimator and the subset of `hyperparams` that was applied.
        """
        if self.factory is not None:
            proto = self.factory()
            valid_keys = frozenset(proto.get_params(deep=True).keys())
            filtered = {k: v for k, v in hyperparams.items() if k in valid_keys}
            merged = {**self.fixed_params, **filtered}
            est = self.factory()
            est.set_params(**merged)
            return est, filtered
        if self.estimator_class is None:
            raise TypeError("EstimatorSpec requires either `estimator_class` or `factory`.")
        proto = self.estimator_class()
        shallow_keys = frozenset(proto.get_params(deep=False).keys())
        filtered = {k: v for k, v in hyperparams.items() if k in shallow_keys}
        merged = {**self.fixed_params, **filtered}
        estimator = self.estimator_class(**merged)
        return estimator, filtered


ESTIMATORS: dict[str, EstimatorSpec] = {
    "hist_gradient_boosting": EstimatorSpec(
        id="hist_gradient_boosting",
        sklearn_name="Pipeline(ColumnTransformer+HistGradientBoostingRegressor)",
        fixed_params={
            "hgb__learning_rate": 0.01,
            "hgb__max_leaf_nodes": 50,
            "hgb__min_samples_leaf": 20,
            "hgb__max_iter": 100,
        },
        factory=_make_full_pipeline,
    ),
}


def resolve(estimator_id: str) -> EstimatorSpec:
    if estimator_id not in ESTIMATORS:
        known = ", ".join(sorted(ESTIMATORS))
        raise ValueError(
            f"Unknown estimator: {estimator_id!r}. Options: {known}"
        )
    return ESTIMATORS[estimator_id]


def list_ids() -> list[str]:
    return sorted(ESTIMATORS.keys())
