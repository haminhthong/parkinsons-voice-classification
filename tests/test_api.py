from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api import app
from app.settings import MAX_UPLOAD_BYTES

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "scope": "research-only"}


def test_predict_endpoint_accepts_csv(data_path):
    with data_path.open("rb") as stream:
        response = client.post("/predict", files={"file": ("sample.csv", stream, "text/csv")})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["records"]) == 195
    assert len(payload["subjects"]) == 32
    assert "không dùng để chẩn đoán" in payload["warning"]
    assert payload["probability_aggregation"] in {"mean", "median", "max"}
    assert 0 < payload["decision_threshold"] < 1


def test_empty_csv_returns_422():
    response = client.post(
        "/predict",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 422


def test_large_upload_returns_413():
    content = b"x" * (MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/predict",
        files={"file": ("large.csv", content, "text/csv")},
    )
    assert response.status_code == 413


def test_internal_error_does_not_expose_path(monkeypatch, data_path):
    monkeypatch.setattr(
        "app.api.predict_records",
        Mock(side_effect=RuntimeError(r"C:\secret\artifact\internal_file.py")),
    )

    with data_path.open("rb") as stream:
        response = client.post("/predict", files={"file": ("sample.csv", stream, "text/csv")})

    assert response.status_code == 500
    assert "C:\\secret" not in response.text
