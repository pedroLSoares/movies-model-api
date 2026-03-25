import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = _APP_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ml-movies-challenge")
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "").strip()
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "1").lower() in ("1", "true", "yes")
