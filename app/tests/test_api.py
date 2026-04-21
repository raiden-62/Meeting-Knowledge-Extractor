from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_analyze():
    response = client.post("/analyze", json={
        "transcript": "We decided to use FastAPI. Alex will implement it."
    })

    assert response.status_code == 200

    data = response.json()
    assert "decisions" in data
    assert "tasks" in data