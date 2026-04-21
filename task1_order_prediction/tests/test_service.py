import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient
from service import app, predictor

predictor.load()
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_known_customer():
    customer_id = predictor.known_customers()[0]
    response = client.post("/predict", json={"customer_id": customer_id, "top_n": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == customer_id
    assert len(body["recommendations"]) <= 5
    for rec in body["recommendations"]:
        assert 0.0 <= rec["reorder_probability"] <= 1.0
        assert isinstance(rec["product_id"], int)


def test_predict_respects_top_n():
    customer_id = predictor.known_customers()[0]
    response = client.post("/predict", json={"customer_id": customer_id, "top_n": 2})
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 2


def test_predict_recommendations_sorted_descending():
    customer_id = predictor.known_customers()[0]
    response = client.post("/predict", json={"customer_id": customer_id, "top_n": 5})
    probs = [r["reorder_probability"] for r in response.json()["recommendations"]]
    assert probs == sorted(probs, reverse=True)


def test_predict_unknown_customer_returns_404():
    response = client.post("/predict", json={"customer_id": 999999})
    assert response.status_code == 404


def test_customers_endpoint():
    response = client.get("/customers")
    assert response.status_code == 200
    ids = response.json()["customer_ids"]
    assert len(ids) > 0


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "RandomForestClassifier"
    assert "feature_order" in body
