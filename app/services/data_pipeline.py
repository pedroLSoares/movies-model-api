from __future__ import annotations

import ast
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FEATURE_COLUMNS: list[str] = [
    "runtime",
    "log_budget",
    "release_month",
    "director",
    "cast_str",
]

PREDICT_FEATURE_NAMES: list[str] = [
    "runtime",
    "budget",
    "release_month",
    "director",
    "cast",
]

REQUIRED_DATASET_FILES: tuple[str, ...] = (
    "ratings.parquet",
    "links.parquet",
    "movies_metadata.parquet",
    "credits.parquet",
    "keywords.parquet",
)

_LINKS_COLS = ("movieId", "tmdbId")
_MOVIES_COLS = (
    "id",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "budget",
    "revenue",
    "release_date",
)
_CREDITS_COLS = ("id", "cast", "crew")
_KEYWORDS_COLS = ("id",)


def _read_parquet_subset(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow", columns=list(columns))
    except Exception:
        return pd.read_parquet(path, engine="pyarrow")


def _read_parquet_rows_matching_ids(
    path: Path,
    columns: tuple[str, ...],
    id_column: str,
    allowed_ids: set[int],
) -> pd.DataFrame:
    if not allowed_ids:
        return pd.DataFrame(columns=list(columns))
    chunks: list[pd.DataFrame] = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=list(columns), batch_size=65_536):
        pdf = batch.to_pandas()
        raw_id = pd.to_numeric(pdf[id_column], errors="coerce")
        take = raw_id.notna() & raw_id.astype("Int64").isin(allowed_ids)
        sub = pdf.loc[take].copy()
        if len(sub) > 0:
            sub[id_column] = raw_id.loc[take].astype("int64")
            chunks.append(sub)
        del pdf, raw_id, take, sub
    if not chunks:
        return pd.DataFrame(columns=list(columns))
    out = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return out.drop_duplicates(subset=[id_column], keep="first")


def ensure_dataset_parquet_files(dataset_root: Path) -> None:
    missing = [
        name for name in REQUIRED_DATASET_FILES if not (dataset_root / name).is_file()
    ]
    if missing:
        raise ValueError(
            f"Incomplete dataset at '{dataset_root}': missing {missing}. "
            "Required files: ratings.parquet, links.parquet, movies_metadata.parquet, "
            "credits.parquet, keywords.parquet. "
            "Convert CSVs with: poetry run python scripts/csv_to_parquet.py <dataset_dir>"
        )


def _director_from_crew(crew_raw: object) -> str:
    if pd.isna(crew_raw) or crew_raw is None:
        return "Unknown"
    try:
        crew = ast.literal_eval(str(crew_raw))
    except (ValueError, SyntaxError, TypeError):
        return "Unknown"
    if not isinstance(crew, list):
        return "Unknown"
    for item in crew:
        if isinstance(item, dict) and item.get("job") == "Director":
            return str(item.get("name", "Unknown"))
    return "Unknown"


def _log_budget_from_budget_raw(budget: Any) -> float:
    """Same as training: `np.log1p` on non-negative `budget`."""
    b = pd.to_numeric(budget, errors="coerce")
    if pd.isna(b):
        b = 0.0
    else:
        b = float(b)
    b = max(0.0, b)
    return float(np.log1p(b))


def _cast_list_to_cast_str(cast_val: Any) -> str:
    """
    Matches training (`_cast_to_str` on list of dicts with `name`): spaces in names become `_`;
    tokens joined with spaces.
    Accepts a list of strings, list of `{"name": "..."}`, or a plain string.
    """
    if cast_val is None or (isinstance(cast_val, float) and pd.isna(cast_val)):
        return ""
    if isinstance(cast_val, str):
        return cast_val.strip()
    if isinstance(cast_val, list):
        parts: list[str] = []
        for item in cast_val[:100]:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip().replace(" ", "_"))
            elif isinstance(item, dict) and item.get("name"):
                parts.append(str(item["name"]).strip().replace(" ", "_"))
        return " ".join(parts)
    raise ValueError(
        "cast must be a list of name strings, dicts with a 'name' key, or a string."
    )


def predict_payload_to_model_frame(
    features: dict[str, Any]
) -> pd.DataFrame:
    log_b = _log_budget_from_budget_raw(features["budget"])
    cast_s = _cast_list_to_cast_str(features["cast"])

    return pd.DataFrame(
        [
            {
                "runtime": float(features["runtime"]),
                "log_budget": log_b,
                "release_month": int(features["release_month"]),
                "director": str(features["director"]),
                "cast_str": cast_s,
            }
        ]
    )


def _cast_to_str(cast_raw: object) -> str:
    if pd.isna(cast_raw) or cast_raw is None:
        return ""
    try:
        cast = ast.literal_eval(str(cast_raw))
    except (ValueError, SyntaxError, TypeError):
        return ""
    if not isinstance(cast, list):
        return ""
    parts: list[str] = []
    for item in cast[:100]:
        if isinstance(item, dict) and item.get("name"):
            parts.append(str(item["name"]).replace(" ", "_"))
    return " ".join(parts)


def load_training_arrays(
    dataset_root: Path,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    ensure_dataset_parquet_files(dataset_root)

    df_ratings = _read_parquet_subset(
        dataset_root / "ratings.parquet", ("movieId", "rating")
    )
    df_ratings_agregado = (
        df_ratings.groupby("movieId", sort=False)["rating"].mean().reset_index()
    )

    df_links = _read_parquet_subset(dataset_root / "links.parquet", _LINKS_COLS)
    df_movies = _read_parquet_subset(
        dataset_root / "movies_metadata.parquet", _MOVIES_COLS
    )

    df_movies["id"] = pd.to_numeric(df_movies["id"], errors="coerce")
    df_movies = df_movies.dropna(subset=["id"]).astype({"id": "int64"})
    df_movies = df_movies.drop_duplicates(subset=["id"], keep="first")

    df_links["tmdbId"] = pd.to_numeric(df_links["tmdbId"], errors="coerce")
    df_links = df_links.dropna(subset=["tmdbId"]).astype({"tmdbId": "int64"})
    df_links = df_links.drop_duplicates(subset=["movieId"], keep="first")

    df_inter = df_ratings_agregado.merge(
        df_links, on="movieId", how="inner"
    )

    tmdb_needed = set(df_inter["tmdbId"].astype("int64").unique().tolist())
    df_movies = df_movies[df_movies["id"].isin(tmdb_needed)].copy()
    gc.collect()

    df_base = df_inter.merge(
        df_movies, left_on="tmdbId", right_on="id", how="inner"
    )

    allowed_movie_ids = set(df_base["id"].astype("int64").unique().tolist())

    df_credits = _read_parquet_rows_matching_ids(
        dataset_root / "credits.parquet",
        _CREDITS_COLS,
        "id",
        allowed_movie_ids,
    )
    df_keywords = _read_parquet_rows_matching_ids(
        dataset_root / "keywords.parquet",
        _KEYWORDS_COLS,
        "id",
        allowed_movie_ids,
    )

    df = df_base.merge(df_credits, on="id", how="inner")
    df = df.merge(df_keywords, on="id", how="inner")

    cols_to_fix = [
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
        "budget",
        "revenue",
    ]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["log_budget"] = np.log1p(df["budget"].fillna(0).clip(lower=0))

    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["release_month"] = df["release_date"].dt.month.fillna(0).astype(int)

    df["director"] = df["crew"].apply(_director_from_crew)
    df["cast_str"] = df["cast"].apply(_cast_to_str)

    df = df.dropna(subset=["rating"])
    df = df[df["vote_count"] > 50].copy()

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing feature column after merge: {col}")

    df = df.dropna(subset=FEATURE_COLUMNS)

    X = df[FEATURE_COLUMNS].copy()
    y = df["rating"].copy()
    return X, y, list(FEATURE_COLUMNS)
