import subprocess
import uuid
from pathlib import Path
import shutil

import soundfile as sf
from fastapi import UploadFile, HTTPException

from backend.app.config import settings


def find_ffmpeg() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        r"C:\Users\Prajwal G N\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build-shared\bin\ffmpeg.exe",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    raise RuntimeError(
        "FFmpeg executable not found. Install FFmpeg or update the path in audio_service.py."
    )


class AudioService:

    async def process_upload(self, file: UploadFile):
        filename = Path(file.filename or "audio").name
        extension = Path(filename).suffix.lower().lstrip(".")

        supported = {
            x.strip().lower()
            for x in settings.supported_audio_formats.split(",")
        }

        if extension not in supported:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Allowed: {sorted(supported)}"
            )

        content = await file.read()
        max_size = settings.max_audio_size_mb * 1024 * 1024

        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {settings.max_audio_size_mb} MB limit"
            )

        file_id = uuid.uuid4().hex
        input_path = settings.upload_dir / f"{file_id}.{extension}"
        processed_path = settings.processed_dir / f"{file_id}.wav"

        input_path.write_bytes(content)

        try:
            ffmpeg = find_ffmpeg()

            command = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(processed_path),
            ]

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if completed.returncode != 0:
                error = completed.stderr.strip() or "Unknown FFmpeg error"
                raise RuntimeError(error)

            if not processed_path.exists():
                raise RuntimeError("Audio conversion failed")

            audio, sample_rate = sf.read(
                str(processed_path),
                dtype="float32"
            )

            if len(audio) == 0:
                raise ValueError("Audio contains no samples")

            duration = len(audio) / sample_rate

            if duration < 1:
                raise ValueError("Audio must be at least 1 second long")

            if duration > 120:
                raise ValueError("Audio must not exceed 120 seconds")

        except subprocess.TimeoutExpired:
            input_path.unlink(missing_ok=True)
            processed_path.unlink(missing_ok=True)

            raise HTTPException(
                status_code=400,
                detail="Audio conversion timed out"
            )

        except HTTPException:
            input_path.unlink(missing_ok=True)
            processed_path.unlink(missing_ok=True)
            raise

        except Exception as exc:
            input_path.unlink(missing_ok=True)
            processed_path.unlink(missing_ok=True)

            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio file: {exc}"
            )

        return {
            "file_id": file_id,
            "original_filename": filename,
            "format": extension,
            "size_bytes": len(content),
            "duration_seconds": round(duration, 2),
            "sample_rate": sample_rate,
            "channels": 1,
            "processed_file": str(processed_path),
        }


audio_service = AudioService()
