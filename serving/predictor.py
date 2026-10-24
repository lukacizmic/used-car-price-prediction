import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from serving.config import (
    CATEGORICAL_COLUMNS,
    METADATA_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    NUMERIC_COLUMNS,
    SCHEMA_ARTIFACT_PATH,
)


class ServingPredictor:
    def __init__(
        self,
        model_path: Path = MODEL_ARTIFACT_PATH,
        schema_path: Path = SCHEMA_ARTIFACT_PATH,
        metadata_path: Path = METADATA_ARTIFACT_PATH,
    ) -> None:
        self.model_path = model_path
        self.schema_path = schema_path
        self.metadata_path = metadata_path
        self.model = None
        self.schema: dict[str, Any] | None = None
        self.metadata: dict[str, Any] | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Run `python -m serving.training` first."
            )
        if not self.schema_path.exists():
            raise FileNotFoundError(
                f"Schema artifact not found at {self.schema_path}. "
                "Run `python -m serving.training` first."
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata artifact not found at {self.metadata_path}. "
                "Run `python -m serving.training` first."
            )

        self.model = joblib.load(self.model_path)
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _cast_numeric(self, col: str, value: Any) -> float:
        if value is None or value == "":
            return float(self.schema["numeric"][col]["default"])
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{col} must be a finite number")
        return numeric_value

    def _cast_categorical(self, col: str, value: Any) -> str:
        if value is None or value == "":
            return "Unknown"
        value = str(value).strip()
        if value == "":
            return "Unknown"
        if col == "model" and value == "__other__":
            return "Unknown"
        return value

    def normalize_record(self, record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        normalized = {}
        warnings = []

        for col in NUMERIC_COLUMNS:
            numeric_value = self._cast_numeric(col, record.get(col))
            if col == "odometer" and numeric_value < 0:
                raise ValueError("odometer must be >= 0")
            normalized[col] = numeric_value

        for col in CATEGORICAL_COLUMNS:
            normalized[col] = self._cast_categorical(col, record.get(col))

        if normalized["year"] < 2000:
            warnings.append(
                "Input year is below 2000. The model was trained mostly on newer vehicles."
            )
        if normalized["title_status"].lower() == "lien":
            warnings.append(
                "title_status=lien was filtered in training and may produce less reliable estimates."
            )

        return normalized, warnings

    def predict_one(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized, warnings = self.normalize_record(record)
        frame = pd.DataFrame([normalized])
        prediction = float(self.model.predict(frame)[0])
        return {
            "predicted_price": round(prediction, 2),
            "warnings": warnings,
            "model_metrics": self.metadata.get("metrics", {}),
        }
