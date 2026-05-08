from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_analyze():
    response = client.post("/analyze", json={
        "transcript": "Jessica will make the C++ server, Bob will do UI and UX"
    })

    assert response.status_code == 200

    data = response.json()
    assert "decisions" in data
    assert "tasks" in data