from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import DATASETS_DIR, MODELS_DIR


def _semver_tuple(name: str) -> tuple[int, int, int] | None:
    parts = name.split(".")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return int(parts[0]), int(parts[1]), int(parts[2])
    return None


def _max_semver_dirname(base: Path) -> str | None:
    """Highest ``M.m.p`` folder name under ``base`` (not lexicographic order)."""
    best: tuple[tuple[int, int, int], str] | None = None
    for d in base.iterdir():
        if not d.is_dir():
            continue
        t = _semver_tuple(d.name)
        if t is None:
            continue
        if best is None or t > best[0]:
            best = (t, d.name)
    return best[1] if best else None


def next_model_version() -> str:
    """Next patch version after the greatest existing ``M.m.p`` under ``MODELS_DIR``."""
    _ensure_dirs()
    triples: list[tuple[int, int, int]] = []
    for d in MODELS_DIR.iterdir():
        if not d.is_dir():
            continue
        t = _semver_tuple(d.name)
        if t is not None:
            triples.append(t)
    if not triples:
        return "1.0.0"
    major, minor, patch = max(triples)
    return f"{major}.{minor}.{patch + 1}"


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_model_versions() -> list[dict[str, Any]]:
    _ensure_dirs()
    out: list[dict[str, Any]] = []
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir():
            meta = _read_json(d / "metadata.json")
            out.append(
                {
                    "version": d.name,
                    "created_at": meta.get("created_at"),
                    "dataset_version": meta.get("dataset_version"),
                    "estimator_id": meta.get("estimator_id"),
                    "mlflow_run_id": meta.get("mlflow_run_id"),
                }
            )
    return out


def list_dataset_versions() -> list[dict[str, Any]]:
    _ensure_dirs()
    out: list[dict[str, Any]] = []
    for d in sorted(DATASETS_DIR.iterdir()):
        if d.is_dir():
            meta = _read_json(d / "manifest.json")
            out.append(
                {
                    "version": d.name,
                    "created_at": meta.get("created_at")
                }
            )
    return out


def latest_dataset_version() -> str | None:
    _ensure_dirs()
    return _max_semver_dirname(DATASETS_DIR)


def latest_model_version() -> str | None:
    _ensure_dirs()
    return _max_semver_dirname(MODELS_DIR)


def register_dataset_version(version: str, source_path: Path | None) -> Path:
    _ensure_dirs()
    target = DATASETS_DIR / version
    target.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "source_path": str(source_path.resolve()) if source_path else None,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if source_path and source_path.is_file():
        shutil.copy2(source_path, target / source_path.name)
    return target


def model_dir(version: str) -> Path:
    return MODELS_DIR / version


def read_model_metadata(version: str) -> dict[str, Any]:
    return _read_json(model_dir(version) / "metadata.json")


def dataset_dir(version: str) -> Path:
    return DATASETS_DIR / version
