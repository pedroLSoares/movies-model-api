from pydantic import BaseModel


class ModelInfo(BaseModel):
    version: str
    created_at: str | None = None
    dataset_version: str | None = None
    estimator_id: str | None = None
    mlflow_run_id: str | None = None


class DatasetInfo(BaseModel):
    version: str
    created_at: str | None = None


class ListResponse(BaseModel):
    models: list[ModelInfo]
    datasets: list[DatasetInfo]
