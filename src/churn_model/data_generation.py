from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_PATH, RANDOM_SEED


YES_NO = np.array(["No", "Yes"])


def _choice(rng: np.random.Generator, values: list[str], probs: list[float], n: int) -> np.ndarray:
    return rng.choice(np.array(values), size=n, p=np.array(probs, dtype=float))


def _addon(
    rng: np.random.Generator,
    internet_service: np.ndarray,
    yes_probability: float,
) -> np.ndarray:
    addon = np.full(internet_service.shape[0], "No internet service", dtype=object)
    has_internet = internet_service != "No"
    addon[has_internet] = np.where(
        rng.random(has_internet.sum()) < yes_probability,
        "Yes",
        "No",
    )
    return addon


def generate_synthetic_churn_data(n_samples: int = 7043, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a realistic telecom churn dataset with known business signal.

    The project workspace does not include a proprietary telecom dataset, so this
    generator creates a deterministic substitute with the same feature patterns
    seen in common subscription churn problems.
    """

    rng = np.random.default_rng(seed)
    customer_id = np.array([f"C{i:05d}" for i in range(1, n_samples + 1)])

    gender = _choice(rng, ["Female", "Male"], [0.50, 0.50], n_samples)
    senior_citizen = _choice(rng, ["No", "Yes"], [0.84, 0.16], n_samples)
    partner = _choice(rng, ["No", "Yes"], [0.52, 0.48], n_samples)
    dependents = np.where(
        partner == "Yes",
        _choice(rng, ["No", "Yes"], [0.53, 0.47], n_samples),
        _choice(rng, ["No", "Yes"], [0.79, 0.21], n_samples),
    )

    contract = _choice(rng, ["Month-to-month", "One year", "Two year"], [0.55, 0.22, 0.23], n_samples)
    tenure_months = np.empty(n_samples, dtype=int)
    month_to_month = contract == "Month-to-month"
    one_year = contract == "One year"
    two_year = contract == "Two year"
    tenure_months[month_to_month] = np.ceil(rng.beta(1.2, 2.5, month_to_month.sum()) * 72).astype(int)
    tenure_months[one_year] = np.ceil(rng.beta(2.0, 1.8, one_year.sum()) * 72).astype(int)
    tenure_months[two_year] = np.ceil(rng.beta(2.6, 1.3, two_year.sum()) * 72).astype(int)
    tenure_months = np.clip(tenure_months, 1, 72)

    phone_service = _choice(rng, ["No", "Yes"], [0.10, 0.90], n_samples)
    multiple_lines = np.full(n_samples, "No phone service", dtype=object)
    has_phone = phone_service == "Yes"
    multiple_lines[has_phone] = np.where(rng.random(has_phone.sum()) < 0.46, "Yes", "No")

    internet_service = _choice(rng, ["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22], n_samples)
    online_security = _addon(rng, internet_service, yes_probability=0.42)
    online_backup = _addon(rng, internet_service, yes_probability=0.44)
    device_protection = _addon(rng, internet_service, yes_probability=0.43)
    tech_support = _addon(rng, internet_service, yes_probability=0.40)
    streaming_tv = _addon(rng, internet_service, yes_probability=0.49)
    streaming_movies = _addon(rng, internet_service, yes_probability=0.50)

    paperless_billing = _choice(rng, ["No", "Yes"], [0.41, 0.59], n_samples)
    payment_method = _choice(
        rng,
        ["Bank transfer", "Credit card", "Electronic check", "Mailed check"],
        [0.22, 0.22, 0.34, 0.22],
        n_samples,
    )

    monthly_charges = np.full(n_samples, 18.0)
    monthly_charges += np.where(phone_service == "Yes", 7.5, 0.0)
    monthly_charges += np.where(multiple_lines == "Yes", 6.5, 0.0)
    monthly_charges += np.where(internet_service == "DSL", 29.0, 0.0)
    monthly_charges += np.where(internet_service == "Fiber optic", 48.0, 0.0)
    monthly_charges += np.where(online_security == "Yes", 5.5, 0.0)
    monthly_charges += np.where(online_backup == "Yes", 5.0, 0.0)
    monthly_charges += np.where(device_protection == "Yes", 5.0, 0.0)
    monthly_charges += np.where(tech_support == "Yes", 5.5, 0.0)
    monthly_charges += np.where(streaming_tv == "Yes", 9.5, 0.0)
    monthly_charges += np.where(streaming_movies == "Yes", 9.5, 0.0)
    monthly_charges += rng.normal(0, 3.2, n_samples)
    monthly_charges = np.clip(monthly_charges, 18.0, 124.0).round(2)

    total_charges = monthly_charges * tenure_months + rng.normal(0, 45, n_samples)
    total_charges = np.clip(total_charges, monthly_charges, None).round(2)

    logit = np.full(n_samples, -0.75)
    logit += np.where(contract == "Month-to-month", 1.45, 0.0)
    logit += np.where(contract == "One year", -0.55, 0.0)
    logit += np.where(contract == "Two year", -1.15, 0.0)
    logit += np.where(internet_service == "Fiber optic", 0.72, 0.0)
    logit += np.where(internet_service == "No", -0.78, 0.0)
    logit += np.where(online_security == "No", 0.48, 0.0)
    logit += np.where(tech_support == "No", 0.46, 0.0)
    logit += np.where(payment_method == "Electronic check", 0.55, 0.0)
    logit += np.where(paperless_billing == "Yes", 0.24, 0.0)
    logit += np.where(senior_citizen == "Yes", 0.26, 0.0)
    logit += np.where(partner == "Yes", -0.26, 0.0)
    logit += np.where(dependents == "Yes", -0.38, 0.0)
    logit += np.where(tenure_months <= 6, 0.42, 0.0)
    logit += -0.040 * tenure_months
    logit += 0.018 * (monthly_charges - 65)
    logit += rng.normal(0, 0.35, n_samples)

    churn_probability = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n_samples) < churn_probability).astype(int)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "churn": churn,
        }
    )


def save_dataset(path: Path = DATASET_PATH, n_samples: int = 7043, seed: int = RANDOM_SEED) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_churn_data(n_samples=n_samples, seed=seed)
    data.to_csv(path, index=False)
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic telecom churn dataset.")
    parser.add_argument("--rows", type=int, default=7043)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    args = parser.parse_args(argv)

    data = save_dataset(args.output, n_samples=args.rows, seed=args.seed)
    churn_rate = data["churn"].mean()
    print(f"Wrote {len(data):,} rows to {args.output}")
    print(f"Churn rate: {churn_rate:.1%}")


if __name__ == "__main__":
    main()

