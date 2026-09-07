from fastapi import Request
from starlette.responses import Response


async def security_headers_middleware(
    request: Request,
    call_next,
) -> Response:
    """Add defensive HTTP security headers."""

    response = await call_next(request)

    response.headers.setdefault(
        "X-Content-Type-Options",
        "nosniff",
    )
    response.headers.setdefault(
        "X-Frame-Options",
        "DENY",
    )
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()",
    )
    response.headers.setdefault(
        "Cache-Control",
        "no-store",
    )

    return response
