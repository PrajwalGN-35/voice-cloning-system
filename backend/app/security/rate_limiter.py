import hashlib
import time
from collections import defaultdict
from threading import Lock

from fastapi import Header, HTTPException, Request

from backend.app.config import settings


class RateLimiter:
    """Simple in-memory per-client rate limiter."""

    def __init__(self):
        self._requests = defaultdict(list)
        self._lock = Lock()

    def _identifier(
        self,
        request: Request,
        x_api_key: str | None = None,
    ) -> str:
        if x_api_key:
            digest = hashlib.sha256(
                x_api_key.encode("utf-8")
            ).hexdigest()
            return f"api:{digest}"

        client = request.client
        host = client.host if client else "unknown"
        return f"ip:{host}"

    def reset(self) -> None:
        """Clear in-memory rate-limit state."""
        with self._lock:
            self._requests.clear()

    async def check(
        self,
        request: Request,
        x_api_key: str | None = None,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> dict:
        """Check and enforce a request limit."""

        if not settings.rate_limit_enabled:
            configured_limit = (
                limit
                if limit is not None
                else settings.rate_limit_requests
            )
            return {
                "allowed": True,
                "limit": configured_limit,
                "remaining": configured_limit,
                "reset": 0,
            }

        configured_limit = (
            limit
            if limit is not None
            else settings.rate_limit_requests
        )
        configured_window = (
            window_seconds
            if window_seconds is not None
            else settings.rate_limit_window_seconds
        )

        if configured_limit <= 0 or configured_window <= 0:
            raise HTTPException(
                status_code=503,
                detail="Rate limiter is misconfigured.",
            )

        now = time.monotonic()
        identifier = self._identifier(request, x_api_key)

        with self._lock:
            timestamps = self._requests[identifier]

            cutoff = now - configured_window
            timestamps[:] = [
                timestamp
                for timestamp in timestamps
                if timestamp > cutoff
            ]

            if len(timestamps) >= configured_limit:
                oldest = timestamps[0]
                retry_after = max(
                    1,
                    int(configured_window - (now - oldest)) + 1,
                )

                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later.",
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(configured_limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            timestamps.append(now)

            remaining = max(
                0,
                configured_limit - len(timestamps),
            )

        return {
            "allowed": True,
            "limit": configured_limit,
            "remaining": remaining,
            "reset": int(now + configured_window),
        }


rate_limiter = RateLimiter()


async def rate_limit_analyze(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> dict:
    """Rate limit voice analysis requests."""
    return await rate_limiter.check(
        request,
        x_api_key=x_api_key,
        limit=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )


async def rate_limit_clone(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> dict:
    """Rate limit voice cloning requests."""
    return await rate_limiter.check(
        request,
        x_api_key=x_api_key,
        limit=settings.clone_rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
