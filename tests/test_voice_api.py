from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_model_status_endpoint():
    response = client.get("/api/v1/model/status")

    assert response.status_code == 200

    data = response.json()

    assert "model" in data
    assert "device" in data
    assert "loaded" in data


def test_clone_rejects_empty_text():
    response = client.post(
        "/api/v1/voice/clone",
        files={
            "reference_audio": (
                "test.wav",
                b"invalid-audio",
                "audio/wav",
            )
        },
        data={
            "text": "   "
        },
    )

    assert response.status_code == 400
    assert "Text cannot be empty" in response.json()["detail"]


def test_clone_requires_reference_audio():
    response = client.post(
        "/api/v1/voice/clone",
        data={
            "text": "hello"
        },
    )

    assert response.status_code == 422
