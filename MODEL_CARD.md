# Model Card — Car Leasing Price Prediction

## Overview

- **Task:** Regression (predict listing price for used cars)
- **Primary model:** XGBoost regressor inside a preprocessing pipeline
- **Serving entry points:** [`api/main.py`](./api/main.py), [`streamlit_app/app.py`](./streamlit_app/app.py)
- **Training/export script:** [`serving/training.py`](./serving/training.py)

## Intended use

- Manual pricing support for portfolio/demo scenarios
- Single-record inference from user-entered listing attributes

Not intended for:

- fully automated valuation decisions without human review
- legal/compliance-critical use without additional governance

## Data and preprocessing

- Source artifact used for training: `.cache/vehicles_cleaned.parquet`
- Base filtering:
  - drop `region`
  - drop missing target (`price`)
- Split: random train/test with fixed seed
- Leakage guard:
  - outlier cutoffs are fit on train split only
- Feature engineering:
  - `age`, `mileage_per_year`, `model_frequency`
  - rare-category grouping
- Final serving drop columns:
  - `state`, `size`, `year`, `location_cluster`

## Inputs and fallback behavior

Expected raw fields:

- numeric: `year`, `odometer`, `lat`, `long`
- categorical: `manufacturer`, `condition`, `cylinders`, `fuel`, `title_status`, `transmission`, `drive`, `size`, `type`, `paint_color`, `state`, `model`

Serving behavior:

- `lat` and `long` are optional at inference time
- if missing, defaults from training schema are used
- unknown categories are handled by preprocessing (`OneHotEncoder(handle_unknown="ignore")`)

## Validation and safety checks

API-level validation (Pydantic):

- bounded numeric ranges (`year`, `odometer`, `lat`, `long`)
- non-empty categorical strings

Predictor-level validation:

- odometer must be non-negative
- numeric values must be finite
- warnings are returned for low year or `title_status=lien`

## Performance reporting

Primary metrics tracked:

- MAE, RMSE, R²
- relative MAE (`MAE / median_price`)
- error tails (`p90`, `p95`, `p99`)
- segment-level error tables

To refresh metrics:

1. retrain/export with `python -m serving.training`
2. rerun notebook diagnostics in [`05_hyperparameter_tuning.ipynb`](./05_hyperparameter_tuning.ipynb)

## Known limitations

- Predictions can be unstable for unrealistic synthetic combinations.
- Manual-input deployment does not include live market data ingestion.
- Category and market drift can degrade performance over time.
- Optional location inputs reduce location-specific precision when omitted.

## Monitoring / maintenance recommendations

- Track prediction distribution and request volume.
- Track monthly error on newly labeled data.
- Add retraining trigger when MAE degradation exceeds threshold.
- Version model artifacts and keep changelog of parameter/feature changes.

