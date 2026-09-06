import uuid
from pathlib import Path
import soundfile as sf
import librosa
from fastapi import UploadFile, HTTPException

from backend.app.config import settings

class AudioService:

    async def process_upload(self, file: UploadFile):
        filename = Path(file.filename or "audio").name
        extension = Path(filename).suffix.lower().replace(".", "")

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
        input_path.write_bytes(content)

        try:
            audio, sample_rate = librosa.load(
                str(input_path),
                sr=16000,
                mono=True
            )

            if len(audio) == 0:
                raise ValueError("Audio contains no samples")

            duration = len(audio) / sample_rate

            if duration < 1:
                raise ValueError("Audio must be at least 1 second long")

            if duration > 120:
                raise ValueError("Audio must not exceed 120 seconds")

            processed_path = settings.processed_dir / f"{file_id}.wav"

            sf.write(
                str(processed_path),
                audio,
                16000,
                subtype="PCM_16"
            )

        except Exception as exc:
            input_path.unlink(missing_ok=True)
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
            "sample_rate": 16000,
            "channels": 1,
            "processed_file": str(processed_path),
        }

audio_service = AudioService()
