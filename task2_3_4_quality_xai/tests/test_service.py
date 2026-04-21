import io
import pickle
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch

import sys
sys.path.insert(0, "../src")

from service import app

client = TestClient(app)


class _DummyClassifier:
    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


def _upload_dummy_pkl(version: str = "v-test") -> None:
    buf = io.BytesIO(pickle.dumps(_DummyClassifier()))
    response = client.post(
        "/upload-model",
        data={"version": version, "accuracy": 0.91, "f1_score": 0.89, "notes": "unit test"},
        files={"model_file": ("model.pkl", buf, "application/octet-stream")},
    )
    assert response.status_code == 200, response.text


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_invalid_extension():
    fake_file = io.BytesIO(b"not a model")
    response = client.post(
        "/upload-model",
        data={"version": "v1.0", "accuracy": 0.92, "notes": "test"},
        files={"model_file": ("model.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_valid_pkl_model():
    model_bytes = io.BytesIO(pickle.dumps(_DummyClassifier()))
    response = client.post(
        "/upload-model",
        data={"version": "v1.0-test", "accuracy": 0.91, "f1_score": 0.89, "notes": "unit test"},
        files={"model_file": ("model.pkl", model_bytes, "application/octet-stream")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1.0-test"
    assert "activated_at" in data


def test_list_models_after_upload():
    response = client.get("/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_grade_requires_image():
    response = client.post("/grade", data={"product_id": 1})
    assert response.status_code == 422


def test_grade_returns_valid_structure():
    _upload_dummy_pkl(version="grade-test")

    from PIL import Image
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    with patch("service.logger.log", return_value=True):
        response = client.post(
            "/grade",
            data={"product_id": 1},
            files={"image": ("test.jpg", buf, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["grade"] in ("A", "B", "C")
    assert "color_score" in data
    assert "size_score" in data
    assert "ripeness_score" in data


def test_grade_a_thresholds():
    from quality_grader import _scores_to_grade
    assert _scores_to_grade(75, 80, 70) == "A"
    assert _scores_to_grade(100, 100, 100) == "A"


def test_grade_b_thresholds():
    from quality_grader import _scores_to_grade
    assert _scores_to_grade(65, 70, 60) == "B"
    assert _scores_to_grade(70, 75, 65) == "B"


def test_grade_c_thresholds():
    from quality_grader import _scores_to_grade
    assert _scores_to_grade(50, 60, 50) == "C"
    assert _scores_to_grade(0, 0, 0) == "C"


def test_get_interactions_calls_desd():
    with patch("service.logger.fetch_logs", return_value=[]) as mock_fetch:
        response = client.get("/interactions?service_type=quality")
        assert response.status_code == 200
        mock_fetch.assert_called_once_with(
            service_type="quality",
            start_date=None,
            end_date=None,
            overrides_only=False,
        )
