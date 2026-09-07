from backend.app.config import settings
from pathlib import Path
import os

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(
    app,
    headers={
        "X-API-Key": settings.api_key,
    },
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAKE_AUDIO = PROJECT_ROOT / "tests" / "fixtures" / "fake_voice.flac"

REAL_AUDIO_ENV = os.getenv("VOICE_REAL_TEST_AUDIO")


def test_analyze_real_audio():
    """Run the detector on a known local REAL audio file when configured."""

    if not REAL_AUDIO_ENV:
        pytest.skip(
            "VOICE_REAL_TEST_AUDIO is not configured; "
            "skipping local REAL-audio integration test."
        )

    real_audio = Path(REAL_AUDIO_ENV)

    assert real_audio.exists(), f"Missing real test audio: {real_audio}"

    with real_audio.open("rb") as audio:
        response = client.post(
            "/api/v1/voice/analyze",
            files={
                "file": (
                    real_audio.name,
                    audio,
                    "audio/wav",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["prediction"] in {"REAL", "FAKE"}

    assert 0 <= data["original_probability"] <= 1
    assert 0 <= data["fake_probability"] <= 1

    assert abs(
        (data["original_probability"] + data["fake_probability"]) - 1
    ) < 0.001


def test_analyze_fake_audio():
    """Known Member 5 FAKE fixture should be blocked by risk engine."""

    assert FAKE_AUDIO.exists(), f"Missing fake fixture: {FAKE_AUDIO}"

    with FAKE_AUDIO.open("rb") as audio:
        response = client.post(
            "/api/v1/voice/analyze",
            files={
                "file": (
                    FAKE_AUDIO.name,
                    audio,
                    "audio/flac",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["prediction"] == "FAKE"

    assert 0 <= data["original_probability"] <= 1
    assert 0 <= data["fake_probability"] <= 1

    assert data["risk_level"] == "HIGH"
    assert data["action"] == "BLOCK"

    assert data["verification_required"] is True
    assert data["verification_method"] == "OTP + SECONDARY_CHECK"


def test_analyze_response_contains_required_fields():
    """API contract should expose all frontend-required fields."""

    assert FAKE_AUDIO.exists(), f"Missing fake fixture: {FAKE_AUDIO}"

    required_fields = {
        "success",
        "prediction",
        "original_probability",
        "fake_probability",
        "original_percentage",
        "fake_percentage",
        "confidence",
        "risk_level",
        "action",
        "message",
        "verification_required",
        "verification_method",
    }

    with FAKE_AUDIO.open("rb") as audio:
        response = client.post(
            "/api/v1/voice/analyze",
            files={
                "file": (
                    FAKE_AUDIO.name,
                    audio,
                    "audio/flac",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert required_fields.issubset(data.keys())


def test_analyze_requires_audio_file():
    """Endpoint should reject requests without an uploaded audio file."""

    response = client.post("/api/v1/voice/analyze")

    assert response.status_code == 422
