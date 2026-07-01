"""
API endpoint tests for the Medical VQA System.
"""

import io
import pytest


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_has_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_check_has_models_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data

    def test_health_check_has_device(self, client):
        response = client.get("/health")
        data = response.json()
        assert "device" in data
        assert data["device"] in ["cuda", "mps", "cpu"]


class TestModelStatusEndpoint:
    """Tests for the /models/status endpoint."""

    def test_model_status_returns_200(self, client):
        response = client.get("/models/status")
        assert response.status_code == 200

    def test_model_status_has_availability(self, client):
        response = client.get("/models/status")
        data = response.json()
        assert "availability" in data

    def test_model_status_has_validation(self, client):
        response = client.get("/models/status")
        data = response.json()
        assert "validation" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint validation."""

    def test_predict_requires_image(self, client):
        """Prediction without image should fail."""
        response = client.post(
            "/predict",
            data={"question": "What is visible?"},
        )
        assert response.status_code == 422  # Validation error

    def test_predict_requires_question(self, client, sample_image_bytes):
        """Prediction without question should fail."""
        response = client.post(
            "/predict",
            files={"image": ("test.png", sample_image_bytes, "image/png")},
        )
        assert response.status_code == 422

    def test_predict_rejects_empty_file(self, client):
        """Prediction with empty file should fail."""
        response = client.post(
            "/predict",
            data={"question": "What is visible?"},
            files={"image": ("test.png", b"", "image/png")},
        )
        assert response.status_code in [400, 500]

    def test_predict_rejects_oversized_file(self, client):
        """Prediction with oversized file should fail with 413."""
        # Create a ~60MB file (over 50MB limit)
        large_data = b"x" * (51 * 1024 * 1024)
        response = client.post(
            "/predict",
            data={"question": "What is visible?"},
            files={"image": ("big.png", large_data, "image/png")},
        )
        assert response.status_code == 413


class TestPredictJsonEndpoint:
    """Tests for the /predict/json endpoint."""

    def test_predict_json_requires_base64(self, client):
        """JSON prediction without base64 should fail."""
        response = client.post(
            "/predict/json",
            json={
                "question": "What is visible?",
            },
        )
        assert response.status_code == 400


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "info" in data
        assert data["info"]["title"] == "Medical Visual Question Answering System"

    def test_docs_endpoint(self, client):
        response = client.get("/docs")
        assert response.status_code == 200


class TestTimingHeader:
    """Tests for the X-Process-Time header."""

    def test_timing_header_present(self, client):
        response = client.get("/health")
        assert "x-process-time" in response.headers
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0
