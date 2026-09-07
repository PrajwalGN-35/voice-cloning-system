import secrets

from fastapi import Header, HTTPException

from backend.app.config import settings


def require_api_key(
    x_api_key: str | None = Header(default=None),
) -> None:
    """Require a valid API key when API authentication is enabled."""

    if not settings.api_auth_enabled:
        return

    configured_key = settings.api_key.strip()

    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is enabled but no API key is configured.",
        )

    if not x_api_key or not secrets.compare_digest(
        x_api_key,
        configured_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Valid API key required.",
        )
