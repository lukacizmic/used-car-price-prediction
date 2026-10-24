from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache"
ENGINEERED_CACHE_DIR = CACHE_DIR / "engineered"

DATA_PATH = CACHE_DIR / "vehicles_cleaned.parquet"
BEST_PARAMS_PATH = ENGINEERED_CACHE_DIR / "xgb_optuna_selected_final.joblib"

MODEL_ARTIFACT_PATH = ENGINEERED_CACHE_DIR / "serving_model.joblib"
SCHEMA_ARTIFACT_PATH = ENGINEERED_CACHE_DIR / "serving_schema.json"
METADATA_ARTIFACT_PATH = ENGINEERED_CACHE_DIR / "serving_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

DROP_COLUMNS = ("state", "size", "year", "location_cluster")

NUMERIC_COLUMNS = ("year", "odometer", "lat", "long")
CATEGORICAL_COLUMNS = (
    "manufacturer",
    "condition",
    "cylinders",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "size",
    "type",
    "paint_color",
    "state",
    "model",
)
RAW_FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

DEFAULT_XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

