from __future__ import annotations

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic_regression(
    features: np.ndarray,
    target: np.ndarray,
    *,
    learning_rate: float = 0.08,
    epochs: int = 3500,
    l2: float = 0.002,
    class_weight: str = "balanced",
    seed: int = 42,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_rows, n_features = features.shape
    target = target.astype(float)

    positive_rate = float(np.clip(target.mean(), 1e-4, 1 - 1e-4))
    weights = rng.normal(0, 0.01, n_features)
    intercept = float(np.log(positive_rate / (1 - positive_rate)))

    if class_weight == "balanced":
        positive_weight = n_rows / (2 * max(target.sum(), 1.0))
        negative_weight = n_rows / (2 * max(n_rows - target.sum(), 1.0))
        sample_weight = np.where(target == 1.0, positive_weight, negative_weight)
    else:
        sample_weight = np.ones(n_rows)

    normalizer = float(sample_weight.sum())
    loss_history: list[float] = []

    for epoch in range(epochs):
        probabilities = sigmoid(features @ weights + intercept)
        error = (probabilities - target) * sample_weight
        gradient_weights = (features.T @ error) / normalizer + l2 * weights
        gradient_intercept = float(error.sum() / normalizer)

        weights -= learning_rate * gradient_weights
        intercept -= learning_rate * gradient_intercept

        if epoch % 250 == 0 or epoch == epochs - 1:
            probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
            cross_entropy = -np.average(
                target * np.log(probabilities) + (1 - target) * np.log(1 - probabilities),
                weights=sample_weight,
            )
            regularization = 0.5 * l2 * float(np.dot(weights, weights))
            loss_history.append(float(cross_entropy + regularization))

    return {
        "weights": weights,
        "intercept": intercept,
        "loss_history": loss_history,
    }


def predict_probabilities(features: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    return sigmoid(features @ weights + intercept)

