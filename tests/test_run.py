import os
import pytest
from unittest.mock import patch

# Mock environment BEFORE any imports from src
os.environ["REBECCA_API_TOKEN"] = "test-valid-token"

from fastapi.testclient import TestClient
from src.api import app, reload_core_adapter

pytestmark = pytest.mark.smoke

# Force reload with test token
reload_core_adapter()

client = TestClient(app)
AUTH_HEADERS = {"Authorization": "Bearer test-valid-token"}


def test_run_pipeline_basic():
    """Smoke test for /run endpoint with minimal payload"""
    payload = {
        "input_data": "test query",
        "trace_id": "test-trace-123"
    }
    response = client.post("/run", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "result" in data
    assert "trace_id" in data
    assert data["trace_id"] == "test-trace-123"


def test_run_pipeline_missing_input_data():
    """Test /run endpoint with missing input_data"""
    payload = {"trace_id": "test-trace-456"}
    response = client.post("/run", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "result" in data


def test_run_pipeline_empty_payload():
    """Test /run endpoint with empty payload"""
    response = client.post("/run", json={}, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "trace_id" in data


def test_run_pipeline_no_auth():
    """Test /run endpoint rejects requests without valid token"""
    payload = {"input_data": "test"}
    response = client.post("/run", json=payload)
    assert response.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])