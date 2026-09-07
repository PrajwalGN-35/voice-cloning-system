from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import settings
from backend.app.security.rate_limiter import rate_limiter

client = TestClient(
    app,
    headers={
        "X-API-Key": settings.api_key,
    },
)


def setup_function():
    rate_limiter.reset()


def test_health_endpoint():
    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_model_status_endpoint():
    response = client.get("/api/v1/model/status")

    assert response.status_code == 200


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


def test_clone_requires_reference_audio():
    response = client.post(
        "/api/v1/voice/clone",
        data={
            "text": "hello"
        },
    )

    assert response.status_code == 422
