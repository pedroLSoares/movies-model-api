
import json
import pandas as pd
import numpy as np
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset
from app.services.data_pipeline import _log_budget_from_budget_raw, _cast_list_to_cast_str


def save_reference_data(X_train, model_version_path):
    X_train.to_parquet(model_version_path / "reference.parquet")



def log_inference(features: dict, version: str):
    log_path = Path(f"app/artifacts/models/{version}/inference_logs.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(features) + "\n")


def run_drift_validation(model_version: str):
    model_path = Path(f"app/artifacts/models/{model_version}")
    log_path = model_path / "inference_logs.jsonl"
    ref_path = model_path / "reference.parquet"

    if not log_path.exists():
        return {"error": "No inference logs found for this version."}
    if not ref_path.exists():
        return {"error": "Reference data not found. Drift analysis impossible."}

    ref_df = pd.read_parquet(ref_path)
    raw_logs = pd.read_json(log_path, lines=True)

    curr_df = pd.DataFrame()
    curr_df["runtime"] = raw_logs["runtime"].astype(float)
    curr_df["log_budget"] = raw_logs["budget"].apply(_log_budget_from_budget_raw)
    curr_df["release_month"] = raw_logs["release_month"].astype(int)
    curr_df["director"] = raw_logs["director"].astype(str)
    curr_df["cast_str"] = raw_logs["cast"].apply(_cast_list_to_cast_str)

    data_drift_tests = Report([
        DataDriftPreset(method="psi")
            ],
        include_tests="True")
    
    evalResult = data_drift_tests.run(reference_data=ref_df, current_data=curr_df)
    
    result = evalResult.json()

    evalResult.save_html(f"app/artifacts/models/{model_version}/drift_results.html")

    
    return {
        "model_version": model_version,
        "drift_detected": result,
    }