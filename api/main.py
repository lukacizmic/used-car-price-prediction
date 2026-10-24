from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from serving.predictor import ServingPredictor

app = FastAPI(title="Car Price Predictor API", version="1.0.0")

_predictor: ServingPredictor | None = None
_load_error: str | None = None
try:
    _predictor = ServingPredictor()
except Exception as exc:  # explicit API surface with detailed startup failure
    _load_error = str(exc)


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    year: float = Field(..., ge=1900, le=2100)
    odometer: float = Field(..., ge=0, le=2_000_000)
    lat: float | None = Field(default=None, ge=-90, le=90)
    long: float | None = Field(default=None, ge=-180, le=180)
    manufacturer: str = Field(..., min_length=1, max_length=64)
    condition: str = Field(..., min_length=1, max_length=64)
    cylinders: str = Field(..., min_length=1, max_length=64)
    fuel: str = Field(..., min_length=1, max_length=32)
    title_status: str = Field(..., min_length=1, max_length=32)
    transmission: str = Field(..., min_length=1, max_length=32)
    drive: str = Field(..., min_length=1, max_length=32)
    size: str = Field(..., min_length=1, max_length=32)
    type: str = Field(..., min_length=1, max_length=64)
    paint_color: str = Field(..., min_length=1, max_length=32)
    state: str = Field(..., min_length=1, max_length=32)
    model: str = Field(..., min_length=1, max_length=128)

    @field_validator(
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
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("must not be empty")
        return value


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _predictor is not None else "error",
        "error": _load_error,
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    if _predictor is None:
        raise HTTPException(status_code=503, detail=_load_error or "Predictor not initialized")
    return {
        "schema": _predictor.schema,
        "metadata": _predictor.metadata,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if _predictor is None:
        raise HTTPException(status_code=503, detail=_load_error or "Predictor not initialized")
    return _predictor.metadata


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    if _predictor is None:
        raise HTTPException(status_code=503, detail=_load_error or "Predictor not initialized")
    try:
        result = _predictor.predict_one(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return result
