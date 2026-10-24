import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from car_pipeline import FeatureEngineeringTransformer, OutlierDetector
from serving.config import (
    BEST_PARAMS_PATH,
    DATA_PATH,
    DEFAULT_XGB_PARAMS,
    DROP_COLUMNS,
    METADATA_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    RANDOM_STATE,
    RAW_FEATURE_COLUMNS,
    SCHEMA_ARTIFACT_PATH,
    TEST_SIZE,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)


def load_training_data(data_path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    data = pd.read_parquet(data_path).copy()
    data = data.drop(columns=["region"]).dropna(subset=["price"])
    if sample_rows is not None and sample_rows < len(data):
        data = data.sample(sample_rows, random_state=RANDOM_STATE).copy()
    return data


def split_and_filter(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = data.drop(columns=["price"])
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    detector = OutlierDetector().fit(X_train, y_train)
    X_train, y_train = detector.transform(X_train, y_train)
    X_test, y_test = detector.transform(X_test, y_test, filter_price=True, filter_odometer=False)
    return X_train, X_test, y_train, y_test


def load_best_params() -> dict[str, Any]:
    params = DEFAULT_XGB_PARAMS.copy()
    if BEST_PARAMS_PATH.exists():
        artifact = joblib.load(BEST_PARAMS_PATH)
        params.update(artifact.get("best_params", {}))
    return params


def build_pipeline(xgb_params: dict[str, Any]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_transformer, make_column_selector(dtype_exclude=np.number)),
        ]
    )

    model = XGBRegressor(
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
        objective="reg:squarederror",
        **xgb_params,
    )
    return Pipeline(
        [
            ("features", FeatureEngineeringTransformer(n_clusters=50, drop_columns=DROP_COLUMNS)),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _category_options(series: pd.Series, top_k: int | None = None) -> list[str]:
    counts = series.fillna("Unknown").astype(str).value_counts()
    values = list(counts.index if top_k is None else counts.head(top_k).index)
    if "Unknown" not in values:
        values.insert(0, "Unknown")
    return values


def build_serving_schema(train_X: pd.DataFrame) -> dict[str, Any]:
    numeric_schema = {}
    for col in NUMERIC_COLUMNS:
        numeric_col = train_X[col].dropna()
        median_value = float(numeric_col.median()) if not numeric_col.empty else 0.0
        numeric_schema[col] = {
            "min": float(numeric_col.min()) if not numeric_col.empty else 0.0,
            "max": float(numeric_col.max()) if not numeric_col.empty else 1.0,
            "default": median_value,
        }

    categorical_schema = {
        "manufacturer": _category_options(train_X["manufacturer"]),
        "condition": _category_options(train_X["condition"]),
        "cylinders": _category_options(train_X["cylinders"]),
        "fuel": _category_options(train_X["fuel"]),
        "title_status": _category_options(train_X["title_status"]),
        "transmission": _category_options(train_X["transmission"]),
        "drive": _category_options(train_X["drive"]),
        "size": _category_options(train_X["size"]),
        "type": _category_options(train_X["type"]),
        "paint_color": _category_options(train_X["paint_color"]),
        "state": _category_options(train_X["state"]),
        # model has very high cardinality; keep dropdown practical with top values.
        "model": ["__other__"] + _category_options(train_X["model"], top_k=300),
    }

    return {
        "raw_feature_columns": list(RAW_FEATURE_COLUMNS),
        "numeric": numeric_schema,
        "categorical": categorical_schema,
        "drop_columns": list(DROP_COLUMNS),
    }


def _evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipe.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    return {
        "test_mse": mse,
        "test_rmse": float(np.sqrt(mse)),
        "test_mae": mae,
        "test_r2": float(r2_score(y_test, y_pred)),
    }


def fit_and_export(
    output_dir: Path, data_path: Path = DATA_PATH, sample_rows: int | None = None
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_training_data(data_path, sample_rows=sample_rows)
    X_train, X_test, y_train, y_test = split_and_filter(data)

    params = load_best_params()
    pipe = build_pipeline(params)
    pipe.fit(X_train, y_train)

    schema = build_serving_schema(X_train)
    metrics = _evaluate(pipe, X_test, y_test)
    metadata = {
        "model_name": "xgboost_serving_pipeline",
        "xgb_params": params,
        "drop_columns": list(DROP_COLUMNS),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
    }

    joblib.dump(pipe, output_dir / MODEL_ARTIFACT_PATH.name)
    (output_dir / SCHEMA_ARTIFACT_PATH.name).write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (output_dir / METADATA_ARTIFACT_PATH.name).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export serving artifacts.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=MODEL_ARTIFACT_PATH.parent)
    parser.add_argument("--sample-rows", type=int, default=None)
    args = parser.parse_args()

    metadata = fit_and_export(
        output_dir=args.output_dir, data_path=args.data_path, sample_rows=args.sample_rows
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

