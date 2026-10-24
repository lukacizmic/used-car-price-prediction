# Car Leasing Price Prediction (Portfolio Project)

End-to-end used-car price prediction project with:

- data cleaning + EDA notebooks
- feature engineering + ablation
- Optuna hyperparameter tuning
- deployable inference service (FastAPI + Streamlit)

## Project objective

Estimate a fair used-car listing price from structured listing attributes
(`year`, `odometer`, `manufacturer`, condition and other listing metadata).

This repository focuses on a production-style **inference app** and reproducible
training/export flow suitable for portfolio demonstration.

## Repository structure

- Notebooks:
  - [`01_cleaning.ipynb`](./01_cleaning.ipynb)
  - [`02_eda.ipynb`](./02_eda.ipynb)
  - [`03_feature_selection_baseline_eval.ipynb`](./03_feature_selection_baseline_eval.ipynb)
  - [`04_feature_engineering.ipynb`](./04_feature_engineering.ipynb)
  - [`05_hyperparameter_tuning.ipynb`](./05_hyperparameter_tuning.ipynb)
- Core pipeline logic: [`car_pipeline.py`](./car_pipeline.py)
- Serving/training scripts: [`serving/`](./serving/)
- One-command local launcher: [`deploy_local.py`](./deploy_local.py)
- API app: [`api/main.py`](./api/main.py)
- UI app: [`streamlit_app/app.py`](./streamlit_app/app.py)
- CI tests: [`tests/`](./tests/)
- Model card: [`MODEL_CARD.md`](./MODEL_CARD.md)

## Model notes

- Base model family: **XGBoost regressor**
- Feature engineering includes:
  - `age`, `mileage_per_year`, `model_frequency`
  - rare-category grouping
- Selected dropped columns for final serving model:
  - `state`, `size`, `year`, `location_cluster`
- Training-time outlier handling is fit on train split to avoid leakage.

## Quickstart (local)

### One-command run (recommended)

```bash
python deploy_local.py
```

This single command will:

1. install serving dependencies
2. build serving artifacts
3. start FastAPI
4. start Streamlit (already pointed at the API)

If port `8000` or `8501` is busy, the script automatically picks the next free port.

Useful options:

```bash
python deploy_local.py --skip-install
python deploy_local.py --skip-train
python deploy_local.py --sample-rows 50000
```

### 1) Install dependencies

```bash
pip install -r requirements-serving.txt
```

### 2) Build serving artifacts (model + schema + metadata)

```bash
python -m serving.training
```

Generated artifacts:

- `.cache/engineered/serving_model.joblib`
- `.cache/engineered/serving_schema.json`
- `.cache/engineered/serving_metadata.json`

Optional smoke run:

```bash
python -m serving.training --sample-rows 50000 --output-dir .cache/engineered_smoke
```

### 3) Run API

```bash
uvicorn api.main:app --reload
```

Endpoints:

- `GET /health`
- `GET /schema`
- `GET /model-info`
- `POST /predict`

### 4) Run Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

Optional API override:

```bash
set CAR_API_URL=http://127.0.0.1:8000
```

## Testing and CI

Install test dependency and run:

```bash
pip install -r requirements-dev.txt
pytest -q
```

GitHub Actions workflow is defined in:
[`/.github/workflows/ci.yml`](./.github/workflows/ci.yml)

## Data source and deployment assumptions

This deployed version is **manual-input inference**:

- user fills listing features in the Streamlit form
- latitude/longitude are optional in the UI; if omitted, training-default values are used
- API scores one record with the trained pipeline
- no live external feed is required

If you want to extend this to live ingestion, add a scheduled data pipeline and
periodic retraining process.

## Limitations

- predictions can be unstable for unrealistic/manual synthetic combinations
- unseen categories are handled gracefully but may reduce accuracy
- metrics are dataset-specific and should be revalidated for new geographies/time windows
