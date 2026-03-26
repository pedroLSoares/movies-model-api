from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.services.data_analysis import save_reference_data, log_inference
from app.services import estimators, registry
from app.services.data_pipeline import (
    PREDICT_FEATURE_NAMES,
    load_training_arrays,
    predict_payload_to_model_frame,
)
from app.services.mlflow_integration import log_training_run

ML_LOG = logging.getLogger("ml_api.ml_core")


def train(
    dataset_version: str | None,
    hyperparams: dict[str, Any],
    estimator_id: str = "hist_gradient_boosting",
) -> tuple[str, str, dict[str, float], dict[str, Any]]:
    registry._ensure_dirs()
    spec = estimators.resolve(estimator_id)

    ds_ver = dataset_version or registry.latest_dataset_version()
    if ds_ver is None:
        raise ValueError("No versioned dataset found. Run /train first.")


    ds_path = registry.dataset_dir(ds_ver)
    if not ds_path.is_dir():
        raise ValueError(f"Dataset version not found: {ds_ver}")

    ML_LOG.info(
        "training_started dataset_version=%s estimator_id=%s path=%s",
        ds_ver,
        estimator_id,
        ds_path,
    )

    X, y, _ = load_training_arrays(ds_path)
    n_total = len(X)
    ML_LOG.info(
        "training_data_loaded total_rows=%s n_features=%s",
        n_total,
        len(PREDICT_FEATURE_NAMES),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    ML_LOG.info(
        "training_split n_train=%s n_test=%s (total_rows=%s)",
        len(X_train),
        len(X_test),
        n_total,
    )

    model, applied_hp = spec.build(hyperparams)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "val_mae": float(mean_absolute_error(y_test, pred)),
        "val_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "val_r2": float(r2_score(y_test, pred)),
        "n_train": float(len(X_train)),
        "n_test": float(len(X_test)),
    }

    model_version = registry.next_model_version()
    out_dir = registry.model_dir(model_version)
    out_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, out_dir / "model.joblib")
    meta: dict[str, Any] = {
        "version": model_version,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": ds_ver,
        "estimator_id": spec.id,
        "estimator_class": spec.sklearn_name,
        "hyperparams": applied_hp,
        "feature_names": list(PREDICT_FEATURE_NAMES),
        "feature_dim": len(PREDICT_FEATURE_NAMES),
        "metrics": metrics,
    }
    mlflow_extra = log_training_run(
        model=model,
        spec=spec,
        dataset_version=ds_ver,
        artifact_version=model_version,
        metrics=metrics,
        applied_hyperparams=applied_hp,
    )
    meta.update(mlflow_extra)
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    ML_LOG.info(
        "training_finished model_version=%s dataset_version=%s artifact_dir=%s",
        model_version,
        ds_ver,
        out_dir,
    )

    save_reference_data(X_train, out_dir)
    return model_version, ds_ver, metrics, mlflow_extra


def load_model(version: str | None) -> tuple[Any, str]:
    registry._ensure_dirs()
    v = version or registry.latest_model_version()
    if v is None:
        raise ValueError("No versioned model found. Run /train first.")
    path = registry.model_dir(v) / "model.joblib"
    if not path.is_file():
        raise ValueError(f"Model artifact not found: {path}")
    return joblib.load(path), v


def predict(
    features: dict[str, Any] | None,
    model_version: str | None,
) -> tuple[Any, str]:
    model, v = load_model(model_version)
    meta_path = registry.model_dir(v) / "metadata.json"
    if not meta_path.is_file():
        raise ValueError("metadata.json is missing for this model.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ds_ver = meta.get("dataset_version")
    if not ds_ver:
        raise ValueError("dataset_version is not set in model metadata.")
    if features is None:
        raise ValueError("Provide `features` (dict keyed by column name).")
    input_data = predict_payload_to_model_frame(features)

    pred = model.predict(input_data)[0]

    log_inference(features, v)
    return pred, v
