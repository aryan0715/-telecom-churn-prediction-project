from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .config import (
    ARTIFACT_DIR,
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    ID_COLUMN,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_COLUMNS,
    RANDOM_SEED,
    REPORT_DIR,
    REPORT_PATH,
    TARGET_COLUMN,
)
from .data_generation import save_dataset
from .metrics import binary_classification_metrics, choose_threshold
from .model import fit_logistic_regression, predict_probabilities
from .preprocessing import fit_preprocessor, transform_features


def _stratified_split(
    data: pd.DataFrame,
    target_column: str,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_indexes: list[int] = []
    validation_indexes: list[int] = []
    test_indexes: list[int] = []

    for _, group in data.groupby(target_column):
        indexes = group.index.to_numpy().copy()
        rng.shuffle(indexes)
        test_count = int(round(len(indexes) * test_size))
        validation_count = int(round(len(indexes) * validation_size))
        test_indexes.extend(indexes[:test_count].tolist())
        validation_indexes.extend(indexes[test_count : test_count + validation_count].tolist())
        train_indexes.extend(indexes[test_count + validation_count :].tolist())

    rng.shuffle(train_indexes)
    rng.shuffle(validation_indexes)
    rng.shuffle(test_indexes)
    return (
        data.loc[train_indexes].reset_index(drop=True),
        data.loc[validation_indexes].reset_index(drop=True),
        data.loc[test_indexes].reset_index(drop=True),
    )


def _load_or_create_dataset(path: Path, rows: int, seed: int, force: bool) -> pd.DataFrame:
    if force or not path.exists():
        return save_dataset(path=path, n_samples=rows, seed=seed)
    return pd.read_csv(path)


def _serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(metrics))


def _baseline_metrics(target: np.ndarray) -> dict[str, float]:
    positive_rate = float(target.mean())
    majority_accuracy = float(max(positive_rate, 1 - positive_rate))
    return {
        "majority_class_accuracy": majority_accuracy,
        "random_guess_auc": 0.5,
        "test_positive_rate": positive_rate,
    }


def _write_report(
    path: Path,
    *,
    artifact: dict[str, Any],
    split_metrics: dict[str, Any],
    baseline: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    test = split_metrics["test"]
    validation = split_metrics["validation"]
    lines = [
        "# Telecom Churn Model Report",
        "",
        f"Generated: {artifact['created_at']}",
        f"Model: {artifact['model_type']}",
        f"Rows: {artifact['data']['rows']:,}",
        f"Selected threshold: {artifact['threshold']:.3f}",
        "",
        "## Held-Out Test Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {test['accuracy']:.3f} |",
        f"| Precision | {test['precision']:.3f} |",
        f"| Recall | {test['recall']:.3f} |",
        f"| F1 | {test['f1']:.3f} |",
        f"| AUC | {test['auc']:.3f} |",
        "",
        "## Baseline",
        "",
        f"- Majority-class accuracy on test: {baseline['majority_class_accuracy']:.3f}",
        f"- Random-guess AUC: {baseline['random_guess_auc']:.3f}",
        f"- Test churn rate: {baseline['test_positive_rate']:.3f}",
        "",
        "## Validation Threshold Selection",
        "",
        f"The threshold was selected on the validation split to maximize F1 while targeting recall >= 0.75.",
        f"Validation F1: {validation['f1']:.3f}; validation recall: {validation['recall']:.3f}.",
        "",
        "## Data Notes",
        "",
        "This repository started without an included telecom dataset. The training data is a deterministic",
        "synthetic dataset with realistic subscription, billing, tenure, support, and product-add-on signals.",
        "Replace `data/telecom_churn.csv` with a real dataset that uses the same schema, then rerun training.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train_pipeline(
    *,
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    report_path: Path = REPORT_PATH,
    rows: int = 7043,
    seed: int = RANDOM_SEED,
    force_data: bool = False,
) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = _load_or_create_dataset(dataset_path, rows=rows, seed=seed, force=force_data)
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="raise").astype(int)

    train_data, validation_data, test_data = _stratified_split(data, TARGET_COLUMN, seed=seed)
    preprocessor = fit_preprocessor(train_data, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)

    x_train = transform_features(train_data, preprocessor)
    y_train = train_data[TARGET_COLUMN].to_numpy(dtype=int)
    x_validation = transform_features(validation_data, preprocessor)
    y_validation = validation_data[TARGET_COLUMN].to_numpy(dtype=int)
    x_test = transform_features(test_data, preprocessor)
    y_test = test_data[TARGET_COLUMN].to_numpy(dtype=int)

    fitted = fit_logistic_regression(x_train, y_train, seed=seed)
    weights = fitted["weights"]
    intercept = float(fitted["intercept"])

    validation_probabilities = predict_probabilities(x_validation, weights, intercept)
    threshold, validation_threshold_metrics = choose_threshold(y_validation, validation_probabilities)

    split_metrics = {
        "train": binary_classification_metrics(
            y_train,
            predict_probabilities(x_train, weights, intercept),
            threshold,
        ),
        "validation": validation_threshold_metrics,
        "test": binary_classification_metrics(
            y_test,
            predict_probabilities(x_test, weights, intercept),
            threshold,
        ),
    }
    split_metrics = _serialize_metrics(split_metrics)
    baseline = _baseline_metrics(y_test)

    artifact: dict[str, Any] = {
        "project": "telecom_customer_churn",
        "version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "logistic_regression_numpy",
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "preprocessor": preprocessor,
        "feature_names": preprocessor["feature_names"],
        "weights": np.asarray(weights, dtype=float).tolist(),
        "intercept": intercept,
        "threshold": threshold,
        "metrics": split_metrics,
        "baseline": baseline,
        "data": {
            "path": str(dataset_path),
            "rows": int(len(data)),
            "train_rows": int(len(train_data)),
            "validation_rows": int(len(validation_data)),
            "test_rows": int(len(test_data)),
            "positive_rate": float(data[TARGET_COLUMN].mean()),
            "source": "deterministic synthetic telecom churn dataset",
        },
        "training": {
            "seed": seed,
            "loss_history": fitted["loss_history"],
            "threshold_selection": "max validation F1 with recall >= 0.75 when feasible",
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    metrics_path.write_text(
        json.dumps({"metrics": split_metrics, "baseline": baseline, "threshold": threshold}, indent=2),
        encoding="utf-8",
    )
    _write_report(report_path, artifact=artifact, split_metrics=split_metrics, baseline=baseline)

    return artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the telecom churn model.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--rows", type=int, default=7043)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--force-data", action="store_true", help="Regenerate the synthetic dataset.")
    args = parser.parse_args(argv)

    artifact = train_pipeline(
        dataset_path=args.dataset,
        model_path=args.model,
        metrics_path=args.metrics,
        report_path=args.report,
        rows=args.rows,
        seed=args.seed,
        force_data=args.force_data,
    )
    test = artifact["metrics"]["test"]
    baseline = artifact["baseline"]
    print("Training complete")
    print(f"Model artifact: {args.model}")
    print(f"Report: {args.report}")
    print(
        "Test metrics: "
        f"accuracy={test['accuracy']:.3f}, precision={test['precision']:.3f}, "
        f"recall={test['recall']:.3f}, f1={test['f1']:.3f}, auc={test['auc']:.3f}"
    )
    print(f"Baseline majority accuracy: {baseline['majority_class_accuracy']:.3f}")


if __name__ == "__main__":
    main()
