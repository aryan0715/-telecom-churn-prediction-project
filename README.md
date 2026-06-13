# Predict Customer Churn for a Telecom Company

An interview-ready churn prediction project with clean training code, held-out metrics, a deployable local API, and a browser demo.

## What It Builds

- A deterministic telecom churn dataset when no external dataset is present.
- A NumPy logistic-regression classifier with preprocessing, threshold tuning, and saved artifacts.
- Metrics for accuracy, precision, recall, F1, and AUC on train, validation, and held-out test splits.
- A standard-library HTTP API for single or batch predictions.
- A polished local demo at `http://127.0.0.1:8000`.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python train.py --force-data
python app.py
```

Open `http://127.0.0.1:8000`.

If your machine uses a different Python launcher, replace `python` with that command.

## Project Structure

```text
.
├── app.py                         # API/demo entrypoint
├── train.py                       # Training entrypoint
├── data/telecom_churn.csv         # Generated training data
├── artifacts/churn_model.json     # Saved model and preprocessing artifact
├── artifacts/metrics.json         # Metrics snapshot
├── demo/index.html                # Browser demo
├── reports/model_report.md        # Interview-friendly model report
├── src/churn_model/               # Package code
└── tests/                         # Unit tests
```

## API

Run:

```powershell
python app.py --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `GET /metrics`
- `GET /schema`
- `POST /predict`

Example request:

```powershell
$body = @{
  tenure_months = 6
  monthly_charges = 99.95
  total_charges = 599.70
  contract = "Month-to-month"
  internet_service = "Fiber optic"
  online_security = "No"
  tech_support = "No"
  payment_method = "Electronic check"
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -Body $body -ContentType "application/json"
```

Missing fields are filled from sensible defaults in `src/churn_model/config.py`.

## Model Success Criteria

The training script evaluates the model on a held-out test split and compares it with simple baselines:

- Accuracy, precision, recall, F1, and AUC for churn classification.
- Majority-class accuracy as a classification baseline.
- Random-guess AUC of `0.500`.

The decision threshold is selected on the validation split to maximize F1 while targeting recall of at least `0.75` when feasible. This favors finding likely churners while still balancing false positives.

## Replacing the Synthetic Dataset

This workspace started empty, so the project creates `data/telecom_churn.csv`. To use a real telecom dataset, keep the same column names, replace that CSV, then run:

```powershell
python train.py
```

The target column must be `churn` with `0` for no churn and `1` for churn.

## Tests

```powershell
python -m unittest discover -s tests
```

