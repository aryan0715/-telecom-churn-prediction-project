from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn_model.metrics import binary_classification_metrics, roc_auc_score


class MetricsTest(unittest.TestCase):
    def test_auc_for_perfect_ranking(self) -> None:
        target = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(roc_auc_score(target, probabilities), 1.0)

    def test_binary_metrics(self) -> None:
        target = np.array([0, 1, 1, 0])
        probabilities = np.array([0.2, 0.8, 0.4, 0.7])
        metrics = binary_classification_metrics(target, probabilities, threshold=0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()

