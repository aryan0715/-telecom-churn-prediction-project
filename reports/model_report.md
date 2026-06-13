# Telecom Churn Model Report

Generated: 2026-06-13T05:33:11.300085+00:00
Model: logistic_regression_numpy
Rows: 7,043
Selected threshold: 0.435

## Held-Out Test Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.823 |
| Precision | 0.742 |
| Recall | 0.856 |
| F1 | 0.795 |
| AUC | 0.897 |

## Baseline

- Majority-class accuracy on test: 0.599
- Random-guess AUC: 0.500
- Test churn rate: 0.401

## Validation Threshold Selection

The threshold was selected on the validation split to maximize F1 while targeting recall >= 0.75.
Validation F1: 0.786; validation recall: 0.856.

## Data Notes

This repository started without an included telecom dataset. The training data is a deterministic
synthetic dataset with realistic subscription, billing, tenure, support, and product-add-on signals.
Replace `data/telecom_churn.csv` with a real dataset that uses the same schema, then rerun training.
