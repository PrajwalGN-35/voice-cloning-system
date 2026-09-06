from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "AI Voice Cloning System"
    app_version: str = "1.0.0"
    debug: bool = True
    max_audio_size_mb: int = 50
    supported_audio_formats: str = "wav,mp3,m4a,flac,ogg"
    upload_dir: Path = Path("data/uploads")
    processed_dir: Path = Path("data/processed")
    generated_dir: Path = Path("data/generated")
    model_dir: Path = Path("models")
    frontend_origins: str = "http://localhost:3000,http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()
