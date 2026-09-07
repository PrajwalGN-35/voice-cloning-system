import asyncio

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app
from backend.app.security.rate_limiter import RateLimiter, rate_limiter


client = TestClient(
    app,
    headers={
        "X-API-Key": settings.api_key,
    },
)


def setup_function():
    rate_limiter.reset()


def test_security_headers():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Permissions-Policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )
    assert response.headers["Cache-Control"] == "no-store"


def test_rate_limiter_configuration():
    assert settings.rate_limit_enabled is True
    assert settings.rate_limit_requests > 0
    assert settings.rate_limit_window_seconds > 0


def test_rate_limiter_returns_429():
    limiter = RateLimiter()

    class Client:
        host = "127.0.0.1"

    class URL:
        path = "/test"

    class FakeRequest:
        client = Client()
        url = URL()

    async def run_test():
        await limiter.check(
            FakeRequest(),
            x_api_key="test-secret",
            limit=2,
            window_seconds=60,
        )

        await limiter.check(
            FakeRequest(),
            x_api_key="test-secret",
            limit=2,
            window_seconds=60,
        )

        try:
            await limiter.check(
                FakeRequest(),
                x_api_key="test-secret",
                limit=2,
                window_seconds=60,
            )
        except HTTPException as exc:
            assert exc.status_code == 429
            assert exc.headers["Retry-After"]
            assert exc.headers["X-RateLimit-Limit"] == "2"
            assert exc.headers["X-RateLimit-Remaining"] == "0"
            return

        raise AssertionError("Expected HTTP 429")


    asyncio.run(run_test())


def test_rate_limiter_does_not_store_raw_api_key():
    limiter = RateLimiter()

    secret = "THIS-MUST-NOT-BE-STORED"

    class Client:
        host = "127.0.0.1"

    class URL:
        path = "/test"

    class FakeRequest:
        client = Client()
        url = URL()

    async def run_test():
        await limiter.check(
            FakeRequest(),
            x_api_key=secret,
            limit=5,
            window_seconds=60,
        )

    asyncio.run(run_test())

    internal_state = repr(limiter._requests)

    assert secret not in internal_state


def test_unauthenticated_analyze_is_rejected():
    response = client.post(
        "/api/v1/voice/analyze",
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401

def test_clone_rate_limiter_configuration():
    assert settings.clone_rate_limit_requests > 0
    assert settings.clone_rate_limit_requests == 5
