from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn_model.config import CATEGORICAL_COLUMNS, DEFAULT_CUSTOMER, NUMERIC_COLUMNS
from churn_model.preprocessing import fit_preprocessor, transform_features


class PredictionContractTest(unittest.TestCase):
    def test_preprocessor_handles_missing_and_unknown_values(self) -> None:
        train = pd.DataFrame(
            [
                {**DEFAULT_CUSTOMER, "contract": "Month-to-month"},
                {**DEFAULT_CUSTOMER, "contract": "Two year", "internet_service": "DSL"},
            ]
        )
        preprocessor = fit_preprocessor(train, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)
        request = pd.DataFrame([{**DEFAULT_CUSTOMER, "contract": "New plan"}])
        features = transform_features(request, preprocessor)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[1], len(preprocessor["feature_names"]))


if __name__ == "__main__":
    unittest.main()

