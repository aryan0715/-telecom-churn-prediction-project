from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


UNKNOWN_LEVEL = "__unknown__"


def fit_preprocessor(
    data: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> dict[str, Any]:
    numeric_columns = numeric_columns or NUMERIC_COLUMNS
    categorical_columns = categorical_columns or CATEGORICAL_COLUMNS

    numeric_stats: dict[str, dict[str, float]] = {}
    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        numeric_stats[column] = {"mean": mean, "std": std}

    category_levels: dict[str, list[str]] = {}
    for column in categorical_columns:
        values = data[column].fillna(UNKNOWN_LEVEL).astype(str)
        levels = sorted(level for level in values.unique().tolist() if level != UNKNOWN_LEVEL)
        levels.append(UNKNOWN_LEVEL)
        category_levels[column] = levels

    feature_names = list(numeric_columns)
    for column in categorical_columns:
        feature_names.extend([f"{column}={level}" for level in category_levels[column]])

    return {
        "numeric_columns": list(numeric_columns),
        "categorical_columns": list(categorical_columns),
        "numeric_stats": numeric_stats,
        "category_levels": category_levels,
        "feature_names": feature_names,
        "unknown_level": UNKNOWN_LEVEL,
    }


def transform_features(data: pd.DataFrame, preprocessor: dict[str, Any]) -> np.ndarray:
    frames: list[np.ndarray] = []

    for column in preprocessor["numeric_columns"]:
        stats = preprocessor["numeric_stats"][column]
        values = pd.to_numeric(data.get(column), errors="coerce")
        values = values.fillna(stats["mean"]).to_numpy(dtype=float)
        frames.append(((values - stats["mean"]) / stats["std"]).reshape(-1, 1))

    unknown_level = preprocessor.get("unknown_level", UNKNOWN_LEVEL)
    for column in preprocessor["categorical_columns"]:
        levels = preprocessor["category_levels"][column]
        values = data.get(column)
        if values is None:
            values = pd.Series([unknown_level] * len(data), index=data.index)
        values = values.fillna(unknown_level).astype(str)
        values = values.where(values.isin(levels), unknown_level)
        encoded = np.zeros((len(data), len(levels)), dtype=float)
        level_to_index = {level: index for index, level in enumerate(levels)}
        for row_index, value in enumerate(values):
            encoded[row_index, level_to_index[value]] = 1.0
        frames.append(encoded)

    return np.hstack(frames).astype(float)

