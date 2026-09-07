from .auth import require_api_key
from .security_log import log_security_event
from .risk_engine import assess_risk
from .rate_limiter import rate_limiter, rate_limit_analyze, rate_limit_clone
from .headers import security_headers_middleware
from .audit import log_audit_event

# Backward compatibility for any older code importing this name.
add_security_headers = security_headers_middleware

__all__ = [
    "require_api_key",
    "log_security_event",
    "assess_risk",
    "rate_limiter",
    "rate_limit_analyze",
    "rate_limit_clone",
    "security_headers_middleware",
    "log_audit_event",
    "add_security_headers",
]
