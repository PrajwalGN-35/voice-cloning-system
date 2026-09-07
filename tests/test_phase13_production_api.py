from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import settings
from backend.app.services.model_service import model_service


client = TestClient(
    app,
    headers={
        "X-API-Key": settings.api_key,
    },
)


def test_root_uses_configured_application_metadata():
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert data["service"] == settings.app_name
    assert data["version"] == settings.app_version
    assert data["status"] == "running"


def test_health_endpoint_returns_service_health():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == settings.app_name
    assert data["version"] == settings.app_version


def test_readiness_endpoint_matches_model_state():
    original_state = model_service.is_loaded

    try:
        model_service.is_loaded = lambda: True

        response = client.get("/api/v1/ready")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["status"] == "ready"
        assert data["model_loaded"] is True

        model_service.is_loaded = lambda: False

        response = client.get("/api/v1/ready")

        assert response.status_code == 503
        data = response.json()

        assert data["success"] is False
        assert data["status"] == "not_ready"
        assert data["model_loaded"] is False

    finally:
        model_service.is_loaded = original_state


def test_cors_does_not_allow_arbitrary_origin():
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers.get("access-control-allow-origin") != (
        "https://attacker.example"
    )


def test_audio_upload_authentication_is_configurable():
    original_auth_state = settings.api_auth_enabled

    try:
        settings.api_auth_enabled = True

        response = client.post(
            "/api/v1/audio/upload",
            files={
                "file": (
                    "test.wav",
                    b"invalid-audio",
                    "audio/wav",
                )
            },
        )

        assert response.status_code != 401

    finally:
        settings.api_auth_enabled = original_auth_state


def test_analyze_route_authentication_is_configurable():
    original_auth_state = settings.api_auth_enabled

    try:
        settings.api_auth_enabled = True

        response = client.post(
            "/api/v1/voice/analyze",
            files={
                "file": (
                    "test.wav",
                    b"invalid-audio",
                    "audio/wav",
                )
            },
        )

        assert response.status_code != 401

    finally:
        settings.api_auth_enabled = original_auth_state


def test_clone_route_authentication_is_configurable():
    original_auth_state = settings.api_auth_enabled

    try:
        settings.api_auth_enabled = True

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
                "text": "test",
            },
        )

        assert response.status_code != 401

    finally:
        settings.api_auth_enabled = original_auth_state
