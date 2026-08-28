from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "scope": "research-only"}


def test_predict_endpoint_accepts_csv(data_path):
    with data_path.open("rb") as stream:
        response = client.post(
            "/predict", files={"file": ("sample.csv", stream, "text/csv")}
        )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["records"]) == 195
    assert len(payload["subjects"]) == 32
    assert "không dùng để chẩn đoán" in payload["warning"]

