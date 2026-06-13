from __future__ import annotations

import numpy as np


def confusion_counts(target: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    target = target.astype(int)
    predicted = predicted.astype(int)
    return {
        "tp": int(((target == 1) & (predicted == 1)).sum()),
        "tn": int(((target == 0) & (predicted == 0)).sum()),
        "fp": int(((target == 0) & (predicted == 1)).sum()),
        "fn": int(((target == 1) & (predicted == 0)).sum()),
    }


def roc_auc_score(target: np.ndarray, probabilities: np.ndarray) -> float:
    target = target.astype(int)
    probabilities = probabilities.astype(float)
    positive_count = int(target.sum())
    negative_count = int(len(target) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return 0.5

    order = np.argsort(probabilities)
    sorted_probabilities = probabilities[order]
    ranks = np.empty(len(probabilities), dtype=float)

    start = 0
    while start < len(probabilities):
        end = start + 1
        while end < len(probabilities) and sorted_probabilities[end] == sorted_probabilities[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    rank_sum_positive = float(ranks[target == 1].sum())
    auc = (rank_sum_positive - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )
    return float(auc)


def binary_classification_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int | dict[str, int]]:
    predicted = (probabilities >= threshold).astype(int)
    counts = confusion_counts(target, predicted)
    total = len(target)

    precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
    recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    accuracy = (counts["tp"] + counts["tn"]) / max(total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": roc_auc_score(target, probabilities),
        "threshold": float(threshold),
        "support": int(total),
        "positive_rate": float(target.mean()),
        "confusion_matrix": counts,
    }


def choose_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_recall: float = 0.75,
) -> tuple[float, dict[str, float | int | dict[str, int]]]:
    thresholds = np.linspace(0.10, 0.90, 161)
    scored = [
        (threshold, binary_classification_metrics(target, probabilities, float(threshold)))
        for threshold in thresholds
    ]
    feasible = [(threshold, metrics) for threshold, metrics in scored if metrics["recall"] >= min_recall]
    candidates = feasible or scored
    best_threshold, best_metrics = max(
        candidates,
        key=lambda item: (item[1]["f1"], item[1]["accuracy"], item[1]["precision"]),
    )
    return float(best_threshold), best_metrics


def rmse(target: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - predicted) ** 2)))


def r2_score(target: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator == 0:
        return 0.0
    numerator = float(np.sum((target - predicted) ** 2))
    return float(1 - numerator / denominator)

