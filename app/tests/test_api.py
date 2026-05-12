from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_analyze():
    response = client.post("/analyze", json={
        "transcript": "Jessica will make the C++ server, Bob will do UI and UX"
    })

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["decisions"], list)
    assert isinstance(data["people"], dict)


def test_transcript_too_long():
    transcript = "A" * (20_000 + 1)

    response = client.post(
        "/analyze",
        json={
            "transcript": transcript
        }
    )

    assert response.status_code == 422


def test_missing_transcript():
    response = client.post(
        "/analyze",
        json={}
    )

    assert response.status_code == 422

def test_mcp_schema():
    response = client.get("/mcp/tool")

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "extract_meeting_knowledge"