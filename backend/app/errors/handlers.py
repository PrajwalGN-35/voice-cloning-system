import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from backend.app.errors.exceptions import AppError


def create_request_id() -> str:
    """Create a short correlation ID for one API request."""
    return uuid.uuid4().hex[:12]


def app_error_response(error: AppError, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={
            "success": False,
            "error": {
                "code": error.code,
                "message": error.message,
                "request_id": request_id,
            },
        },
    )


async def app_error_handler(
    request: Request,
    exc: AppError,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        create_request_id(),
    )

    return app_error_response(
        error=exc,
        request_id=request_id,
    )
