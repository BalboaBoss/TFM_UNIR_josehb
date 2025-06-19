from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_classification():
    payload = {"subject": "Login issue", "body": "Cannot access the platform."}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200
    assert "category" in response.json()
    assert "confidence" in response.json()