import pytest
from fastapi.testclient import TestClient
from src.api import app

pytestmark = pytest.mark.smoke

client = TestClient(app)

def test_health_check():
    """Smoke test for /health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"

def test_health_check_structure():
    """Verify health response contains expected fields"""
    response = client.get("/health")
    data = response.json()
    assert isinstance(data, dict)
    # Add more field checks based on actual health_check() implementation

if __name__ == "__main__":
    pytest.main([__file__, "-v"])