from fastapi.testclient import TestClient

import api.main as api_main


class FakePredictor:
    def __init__(self):
        self.schema = {
            "numeric": {
                "year": {"default": 2015},
                "odometer": {"default": 65000},
                "lat": {"default": 38.627},
                "long": {"default": -90.199},
            },
            "categorical": {"manufacturer": ["Unknown", "ford"]},
        }
        self.metadata = {"metrics": {"test_mae": 3800.0, "test_r2": 0.81}}

    def predict_one(self, payload):
        if payload["odometer"] < 0:
            raise ValueError("odometer must be >= 0")
        return {
            "predicted_price": 12000.0,
            "warnings": [],
            "model_metrics": self.metadata["metrics"],
        }


def valid_payload():
    return {
        "year": 2018,
        "odometer": 72000,
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
        "model": "fusion",
    }


def test_health_ok_when_predictor_present(monkeypatch):
    monkeypatch.setattr(api_main, "_predictor", FakePredictor())
    monkeypatch.setattr(api_main, "_load_error", None)
    client = TestClient(api_main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_schema_returns_payload(monkeypatch):
    monkeypatch.setattr(api_main, "_predictor", FakePredictor())
    client = TestClient(api_main.app)
    response = client.get("/schema")
    assert response.status_code == 200
    body = response.json()
    assert "schema" in body
    assert "metadata" in body


def test_model_info_returns_metadata(monkeypatch):
    monkeypatch.setattr(api_main, "_predictor", FakePredictor())
    client = TestClient(api_main.app)
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["metrics"]["test_mae"] == 3800.0


def test_predict_success(monkeypatch):
    monkeypatch.setattr(api_main, "_predictor", FakePredictor())
    client = TestClient(api_main.app)
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_price"] == 12000.0
    assert body["model_metrics"]["test_r2"] == 0.81


def test_predict_validation_error_maps_to_422(monkeypatch):
    monkeypatch.setattr(api_main, "_predictor", FakePredictor())
    client = TestClient(api_main.app)
    payload = valid_payload()
    payload["odometer"] = -10
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "odometer"


def test_predict_rejects_invalid_latitude(monkeypatch):
    monkeypatch.setattr(api_main, "_predictor", FakePredictor())
    client = TestClient(api_main.app)
    payload = valid_payload()
    payload["lat"] = 120
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "lat"


def test_predict_rejects_blank_manufacturer(monkeypatch):
    monkeypatch.setattr(api_main, "_predictor", FakePredictor())
    client = TestClient(api_main.app)
    payload = valid_payload()
    payload["manufacturer"] = "   "
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "manufacturer"
