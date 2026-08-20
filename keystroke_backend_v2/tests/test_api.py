import pytest
from fastapi.testclient import TestClient
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "ok"]
    assert "model_loaded" in data


def test_model_info_endpoint():
    response = client.get("/model-info")
    # Returns 200 if loaded or 503 if model not trained yet
    assert response.status_code in [200, 503]
