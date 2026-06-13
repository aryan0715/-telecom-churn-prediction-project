from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
REPORT_DIR = ROOT_DIR / "reports"
DEMO_DIR = ROOT_DIR / "demo"

DATASET_PATH = DATA_DIR / "telecom_churn.csv"
MODEL_PATH = ARTIFACT_DIR / "churn_model.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
REPORT_PATH = REPORT_DIR / "model_report.md"

RANDOM_SEED = 42
TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"

NUMERIC_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

EXPECTED_COLUMNS = [ID_COLUMN, *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS, TARGET_COLUMN]

DEFAULT_CUSTOMER = {
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 89.9,
    "total_charges": 1078.8,
}

