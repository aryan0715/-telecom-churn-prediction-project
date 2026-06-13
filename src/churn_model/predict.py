from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CUSTOMER, MODEL_PATH, NUMERIC_COLUMNS
from .model import predict_probabilities
from .preprocessing import transform_features


def load_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}. Run `python train.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _records_to_frame(records: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(records, dict):
        records = [records]
    if not records:
        raise ValueError("At least one customer record is required.")

    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(DEFAULT_CUSTOMER)
        row.update(record)
        if "total_charges" not in record and {"monthly_charges", "tenure_months"} <= set(record):
            row["total_charges"] = float(record["monthly_charges"]) * float(record["tenure_months"])
        for column in NUMERIC_COLUMNS:
            row[column] = float(row[column])
        rows.append(row)
    return pd.DataFrame(rows)


def _risk_band(probability: float, threshold: float) -> str:
    if probability >= max(0.70, threshold + 0.15):
        return "high"
    if probability >= threshold:
        return "elevated"
    if probability >= 0.30:
        return "watch"
    return "low"


def _risk_signals(row: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    if row.get("contract") == "Month-to-month":
        signals.append("Month-to-month contract")
    if float(row.get("tenure_months", 0)) <= 12:
        signals.append("Short tenure")
    if row.get("internet_service") == "Fiber optic":
        signals.append("Fiber optic service")
    if row.get("online_security") == "No":
        signals.append("No online security")
    if row.get("tech_support") == "No":
        signals.append("No tech support")
    if row.get("payment_method") == "Electronic check":
        signals.append("Electronic check payment")
    if float(row.get("monthly_charges", 0)) >= 85:
        signals.append("High monthly charges")
    return signals[:4]


def predict_records(
    records: dict[str, Any] | list[dict[str, Any]],
    *,
    artifact: dict[str, Any] | None = None,
    artifact_path: Path = MODEL_PATH,
) -> list[dict[str, Any]]:
    artifact = artifact or load_artifact(artifact_path)
    data = _records_to_frame(records)
    features = transform_features(data, artifact["preprocessor"])
    probabilities = predict_probabilities(
        features,
        np.array(artifact["weights"], dtype=float),
        float(artifact["intercept"]),
    )
    threshold = float(artifact["threshold"])

    predictions: list[dict[str, Any]] = []
    for index, probability in enumerate(probabilities):
        row = data.iloc[index].to_dict()
        prediction = int(probability >= threshold)
        predictions.append(
            {
                "customer_id": row.get("customer_id", f"request-{index + 1}"),
                "churn_probability": round(float(probability), 4),
                "prediction": "churn" if prediction == 1 else "no_churn",
                "prediction_label": "Churn" if prediction == 1 else "No churn",
                "threshold": round(threshold, 4),
                "risk_band": _risk_band(float(probability), threshold),
                "risk_signals": _risk_signals(row),
            }
        )
    return predictions

