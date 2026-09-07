import logging


logger = logging.getLogger("voiceguard.audit")


def log_audit_event(
    event: str,
    request,
    status_code: int,
    **details,
) -> None:
    """Write a structured security/audit event without secrets."""

    safe_details = {
        key: value
        for key, value in details.items()
        if key.lower() not in {
            "api_key",
            "x_api_key",
            "authorization",
            "token",
            "password",
        }
    }

    logger.info(
        "AUDIT event=%s method=%s path=%s status=%s details=%s",
        event,
        request.method,
        request.url.path,
        status_code,
        safe_details,
    )
