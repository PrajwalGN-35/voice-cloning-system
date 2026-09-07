from typing import Optional


class AppError(Exception):
    """Base application exception with a stable API error code."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class AudioProcessingError(AppError):
    """Raised when uploaded audio cannot be safely processed."""

    def __init__(
        self,
        message: str = "The uploaded audio file could not be processed.",
        code: str = "AUDIO_INVALID",
        status_code: int = 400,
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status_code,
        )


class DetectionError(AppError):
    """Raised when deepfake detection cannot be completed."""

    def __init__(
        self,
        message: str = "Voice authenticity analysis could not be completed.",
        code: str = "DETECTION_FAILED",
        status_code: int = 500,
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status_code,
        )


class SecurityProcessingError(AppError):
    """Raised when risk/security processing cannot be completed."""

    def __init__(
        self,
        message: str = "Security risk assessment could not be completed.",
        code: str = "SECURITY_PROCESSING_FAILED",
        status_code: int = 500,
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status_code,
        )
