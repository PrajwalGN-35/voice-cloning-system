from pydantic import BaseModel, Field


class VoiceCloneResponse(BaseModel):
    success: bool = True
    message: str = "Voice generated successfully"
    filename: str
    sample_rate: int
    text: str
    audio_url: str


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail: str | None = None
