from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Voice Cloning System"
    app_version: str = "1.0.0"
    debug: bool = False

    max_audio_size_mb: int = 50
    supported_audio_formats: str = "wav,mp3,m4a,flac,ogg"

    upload_dir: Path = Path("data/uploads")
    processed_dir: Path = Path("data/processed")
    generated_dir: Path = Path("data/generated")
    model_dir: Path = Path("models")

    frontend_origins: str = "http://localhost:3000,http://localhost:5173"

    api_key: str = Field(
        default="",
        validation_alias="VOICEGUARD_API_KEY",
    )
    api_auth_enabled: bool = False

    # Rate limiting / abuse protection
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60
    clone_rate_limit_requests: int = 5

    # Production API controls

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
