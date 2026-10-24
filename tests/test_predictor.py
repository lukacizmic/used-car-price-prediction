import pytest

from serving.predictor import ServingPredictor


class DummyModel:
    def predict(self, frame):
        return [12345.678]


def make_predictor_stub() -> ServingPredictor:
    predictor = ServingPredictor.__new__(ServingPredictor)
    predictor.model = DummyModel()
    predictor.schema = {
        "numeric": {
            "year": {"default": 2015},
            "odometer": {"default": 65000},
            "lat": {"default": 38.627},
            "long": {"default": -90.199},
        }
    }
    predictor.metadata = {"metrics": {"test_mae": 3800.0, "test_r2": 0.81}}
    return predictor


def test_normalize_uses_defaults_for_missing_numeric_values():
    predictor = make_predictor_stub()
    normalized, warnings = predictor.normalize_record(
        {
            "year": None,
            "odometer": None,
            "lat": None,
            "long": None,
            "manufacturer": "ford",
            "condition": "good",
            "cylinders": "4 cylinders",
            "fuel": "gas",
            "title_status": "clean",
            "transmission": "automatic",
            "drive": "fwd",
            "size": "mid-size",
            "type": "sedan",
            "paint_color": "white",
            "state": "mo",
            "model": "__other__",
        }
    )
    assert normalized["year"] == 2015.0
    assert normalized["odometer"] == 65000.0
    assert normalized["lat"] == 38.627
    assert normalized["long"] == -90.199
    assert normalized["model"] == "Unknown"
    assert warnings == []


def test_normalize_raises_for_negative_odometer():
    predictor = make_predictor_stub()
    with pytest.raises(ValueError, match="odometer must be >= 0"):
        predictor.normalize_record(
            {
                "year": 2018,
                "odometer": -1,
                "lat": 38.6,
                "long": -90.2,
                "manufacturer": "toyota",
                "condition": "good",
                "cylinders": "4 cylinders",
                "fuel": "gas",
                "title_status": "clean",
                "transmission": "automatic",
                "drive": "fwd",
                "size": "mid-size",
                "type": "sedan",
                "paint_color": "black",
                "state": "mo",
                "model": "camry",
            }
        )


def test_normalize_rejects_non_finite_numeric_values():
    predictor = make_predictor_stub()
    with pytest.raises(ValueError, match="lat must be a finite number"):
        predictor.normalize_record(
            {
                "year": 2018,
                "odometer": 1000,
                "lat": float("nan"),
                "long": -90.2,
                "manufacturer": "toyota",
                "condition": "good",
                "cylinders": "4 cylinders",
                "fuel": "gas",
                "title_status": "clean",
                "transmission": "automatic",
                "drive": "fwd",
                "size": "mid-size",
                "type": "sedan",
                "paint_color": "black",
                "state": "mo",
                "model": "camry",
            }
        )


def test_predict_one_returns_rounded_price_and_warnings():
    predictor = make_predictor_stub()
    result = predictor.predict_one(
        {
            "year": 1999,
            "odometer": 120000,
            "lat": None,
            "long": None,
            "manufacturer": "honda",
            "condition": "fair",
            "cylinders": "4 cylinders",
            "fuel": "gas",
            "title_status": "lien",
            "transmission": "automatic",
            "drive": "fwd",
            "size": "compact",
            "type": "coupe",
            "paint_color": "silver",
            "state": "il",
            "model": "civic",
        }
    )
    assert result["predicted_price"] == 12345.68
    assert result["model_metrics"]["test_mae"] == 3800.0
    assert any("below 2000" in w for w in result["warnings"])
    assert any("title_status=lien" in w for w in result["warnings"])
