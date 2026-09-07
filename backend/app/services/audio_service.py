import logging
import shutil
import subprocess
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import UploadFile

from backend.app.config import settings
from backend.app.errors.exceptions import AudioProcessingError

logger = logging.getLogger(__name__)


def find_ffmpeg() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        r"C:\Users\Prajwal G N\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build-shared\bin\ffmpeg.exe",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    raise AudioProcessingError(
        message="Audio processing service is unavailable because FFmpeg was not found.",
        code="FFMPEG_NOT_FOUND",
        status_code=500,
    )


class AudioService:

    async def process_upload(self, file: UploadFile):
        filename = Path(file.filename or "audio").name
        extension = Path(filename).suffix.lower().lstrip(".")

        supported = {
            x.strip().lower()
            for x in settings.supported_audio_formats.split(",")
            if x.strip()
        }

        if not extension or extension not in supported:
            raise AudioProcessingError(
                message=f"Unsupported audio format. Allowed formats: {sorted(supported)}",
                code="UNSUPPORTED_AUDIO_FORMAT",
                status_code=400,
            )

        try:
            content = await file.read()
        except Exception as exc:
            logger.exception(
                "Failed to read uploaded audio file filename=%s",
                filename,
            )
            raise AudioProcessingError(
                message="The uploaded audio file could not be read.",
                code="AUDIO_READ_FAILED",
                status_code=400,
            ) from exc

        max_size = settings.max_audio_size_mb * 1024 * 1024

        if len(content) > max_size:
            raise AudioProcessingError(
                message=f"File exceeds {settings.max_audio_size_mb} MB limit.",
                code="AUDIO_TOO_LARGE",
                status_code=413,
            )

        if len(content) == 0:
            raise AudioProcessingError(
                message="The uploaded audio file is empty.",
                code="AUDIO_EMPTY",
                status_code=400,
            )

        file_id = uuid.uuid4().hex
        input_path = settings.upload_dir / f"{file_id}.{extension}"
        processed_path = settings.processed_dir / f"{file_id}.wav"

        try:
            input_path.write_bytes(content)

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

            logger.info(
                "Starting audio conversion file_id=%s format=%s size=%d",
                file_id,
                extension,
                len(content),
            )

            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                logger.warning(
                    "FFmpeg conversion timed out file_id=%s",
                    file_id,
                )
                raise AudioProcessingError(
                    message="Audio conversion timed out.",
                    code="FFMPEG_TIMEOUT",
                    status_code=400,
                ) from exc
            except OSError as exc:
                logger.exception(
                    "FFmpeg could not be executed file_id=%s",
                    file_id,
                )
                raise AudioProcessingError(
                    message="Audio processing service could not start FFmpeg.",
                    code="FFMPEG_EXECUTION_FAILED",
                    status_code=500,
                ) from exc

            if completed.returncode != 0:
                error_detail = completed.stderr.strip()

                logger.warning(
                    "FFmpeg conversion failed file_id=%s returncode=%s error=%s",
                    file_id,
                    completed.returncode,
                    error_detail[:500],
                )

                raise AudioProcessingError(
                    message="The uploaded file is not a valid or supported audio file.",
                    code="AUDIO_CONVERSION_FAILED",
                    status_code=400,
                )

            if not processed_path.exists():
                logger.error(
                    "FFmpeg completed without creating output file file_id=%s",
                    file_id,
                )
                raise AudioProcessingError(
                    message="Audio conversion completed without producing an output file.",
                    code="AUDIO_OUTPUT_MISSING",
                    status_code=500,
                )

            try:
                audio, sample_rate = sf.read(
                    str(processed_path),
                    dtype="float32",
                )
            except Exception as exc:
                logger.warning(
                    "Processed audio could not be decoded file_id=%s error=%s",
                    file_id,
                    str(exc)[:500],
                )
                raise AudioProcessingError(
                    message="The processed audio file is invalid or corrupted.",
                    code="AUDIO_DECODE_FAILED",
                    status_code=400,
                ) from exc

            if len(audio) == 0:
                raise AudioProcessingError(
                    message="Audio contains no samples.",
                    code="AUDIO_NO_SAMPLES",
                    status_code=400,
                )

            if sample_rate <= 0:
                raise AudioProcessingError(
                    message="Audio has an invalid sample rate.",
                    code="AUDIO_INVALID_SAMPLE_RATE",
                    status_code=400,
                )

            duration = len(audio) / sample_rate

            if duration < 1:
                raise AudioProcessingError(
                    message="Audio must be at least 1 second long.",
                    code="AUDIO_TOO_SHORT",
                    status_code=400,
                )

            if duration > 120:
                raise AudioProcessingError(
                    message="Audio must not exceed 120 seconds.",
                    code="AUDIO_TOO_LONG",
                    status_code=400,
                )

            logger.info(
                "Audio processing completed file_id=%s duration=%.2fs sample_rate=%d",
                file_id,
                duration,
                sample_rate,
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

        except AudioProcessingError:
            raise

        except OSError as exc:
            logger.exception(
                "Filesystem error during audio processing file_id=%s",
                file_id,
            )
            raise AudioProcessingError(
                message="Audio file processing could not be completed.",
                code="AUDIO_FILESYSTEM_ERROR",
                status_code=500,
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected audio processing failure file_id=%s",
                file_id,
            )
            raise AudioProcessingError(
                message="An unexpected error occurred while processing the audio file.",
                code="AUDIO_PROCESSING_FAILED",
                status_code=500,
            ) from exc

        finally:
            # The original uploaded file is temporary and can always be removed.
            input_path.unlink(missing_ok=True)

            # The processed WAV must remain available to the caller.
            # The caller is responsible for deleting it after downstream
            # processing, such as deepfake detection, has completed.
            if processed_path.exists() and not processed_path.is_file():
                logger.warning(
                    "Processed path is not a regular file file_id=%s",
                    file_id,
                )

    def cleanup_processed_file(self, processed_file: str) -> None:
        """Delete a processed WAV after downstream processing is complete."""
        path = Path(processed_file)

        try:
            path.unlink(missing_ok=True)
            logger.info(
                "Processed audio cleaned up path=%s",
                path,
            )
        except OSError:
            logger.exception(
                "Failed to clean up processed audio path=%s",
                path,
            )
    async def close_upload(self, file: UploadFile) -> None:
        try:
            await file.close()
        except Exception:
            logger.warning("Failed to close uploaded file handle.")


audio_service = AudioService()


