from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from collections import deque
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from backend.app.config import settings
from backend.app.security import require_api_key
from backend.app.services.audio_service import find_ffmpeg
from backend.app.services.audio_quality_service import AudioQualityService
from backend.app.services.deepfake_service import get_deepfake_service
from backend.app.security.adaptive_risk_service import assess_adaptive_risk
from backend.app.services.live_session_service import live_session_service


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/live", tags=["live"])

MAX_CHUNK_BYTES = 8 * 1024 * 1024
LIVE_WINDOW_SECONDS = 5
MAX_BUFFER_BYTES = 40 * 1024 * 1024

audio_quality_service = AudioQualityService()


def _allowed_origins() -> set[str]:
    return {
        origin.strip().rstrip("/")
        for origin in settings.frontend_origins.split(",")
        if origin.strip()
    }


@router.post("/session")
def create_live_session(
    _: None = Depends(require_api_key),
):
    token = live_session_service.create_session()

    return {
        "success": True,
        "token": token,
        "expires_in_seconds": live_session_service.ttl_seconds,
    }


@router.get("/status")
def live_status():
    return {
        "success": True,
        "service": "VoiceGuard live streaming",
        "status": "ready",
        "transport": "websocket",
        "window_seconds": LIVE_WINDOW_SECONDS,
        "analysis": "AASIST + reliability + adaptive risk",
        "windowing": "rolling",
    }


def _process_live_window(
    binary_data: bytes,
    connection_id: str,
    sequence_number: int,
) -> dict:
    """
    Analyse one complete rolling live-audio window.

    Browser MediaRecorder data may be WebM/Opus or another supported
    container. FFmpeg converts the window to 16 kHz mono PCM WAV before
    sending it through the existing VoiceGuard detection pipeline.
    """

    started = time.perf_counter()
    window_id = uuid4().hex

    input_path = settings.processed_dir / f"live_window_{window_id}.webm"
    wav_path = settings.processed_dir / f"live_window_{window_id}.wav"

    try:
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(binary_data)

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
            str(wav_path),
        ]

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if completed.returncode != 0 or not wav_path.exists():
            detail = completed.stderr.strip()[:500]
            raise RuntimeError(
                f"Live audio conversion failed: {detail}"
            )

        detector = get_deepfake_service()

        detection = detector.analyze(str(wav_path))
        quality = audio_quality_service.analyze(str(wav_path))

        risk = assess_adaptive_risk(
            prediction=detection["prediction"],
            confidence=detection["confidence"],
            audio_quality=quality["quality"],
            quality_flags=quality["flags"],
        )

        elapsed_ms = round(
            (time.perf_counter() - started) * 1000,
            1,
        )

        return {
            "type": "ANALYSIS_RESULT",
            "connection_id": connection_id,
            "window_id": window_id,
            "sequence_number": sequence_number,
            "window_seconds": LIVE_WINDOW_SECONDS,
            "processing_ms": elapsed_ms,
            "detection": {
                "prediction": detection["prediction"],
                "confidence": detection["confidence"],
                "real_probability": detection["original_probability"],
                "fake_probability": detection["fake_probability"],
                "threshold": detection["threshold"],
                "raw_scores": detection.get("raw_scores"),
            },
            "audio_quality": quality,
            "risk": risk,
        }

    finally:
        input_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)


async def _send_json(websocket: WebSocket, payload: dict):
    await websocket.send_json(payload)


@router.websocket("/ws")
async def live_websocket(websocket: WebSocket):
    origin = websocket.headers.get("origin", "").rstrip("/")

    if origin and origin not in _allowed_origins():
        await websocket.close(
            code=1008,
            reason="Origin not allowed",
        )
        return

    token = websocket.query_params.get("token", "")

    if not token or not live_session_service.validate(token):
        await websocket.close(
            code=1008,
            reason="Invalid live session",
        )
        return

    connection_id = str(uuid4())

    await websocket.accept()

    await _send_json(
        websocket,
        {
            "type": "CONNECTED",
            "connection_id": connection_id,
            "window_seconds": LIVE_WINDOW_SECONDS,
            "windowing": "rolling",
            "message": "Live audio stream connected.",
        },
    )

    logger.info(
        "Live websocket connected connection_id=%s",
        connection_id,
    )

    audio_buffer: deque[bytes] = deque()
    buffered_bytes = 0
    sequence_number = 0
    analysis_in_progress = False

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            binary_data = message.get("bytes")

            if binary_data is None:
                text_data = message.get("text")

                if text_data == "PING":
                    await _send_json(
                        websocket,
                        {
                            "type": "PONG",
                            "connection_id": connection_id,
                        },
                    )

                continue

            if not binary_data:
                continue

            if len(binary_data) > MAX_CHUNK_BYTES:
                await _send_json(
                    websocket,
                    {
                        "type": "ERROR",
                        "code": "CHUNK_TOO_LARGE",
                        "message": "Live audio chunk exceeds the allowed size.",
                    },
                )
                continue

            if not live_session_service.validate(token):
                await _send_json(
                    websocket,
                    {
                        "type": "ERROR",
                        "code": "LIVE_SESSION_EXPIRED",
                        "message": "Live protection session expired.",
                    },
                )
                break

            audio_buffer.append(binary_data)
            buffered_bytes += len(binary_data)

            while buffered_bytes > MAX_BUFFER_BYTES and audio_buffer:
                removed = audio_buffer.popleft()
                buffered_bytes -= len(removed)

            await _send_json(
                websocket,
                {
                    "type": "CHUNK_RECEIVED",
                    "connection_id": connection_id,
                    "bytes": len(binary_data),
                    "buffered_bytes": buffered_bytes,
                },
            )

            # The frontend sends approximately one second per chunk.
            # Five chunks therefore form one analysis window.
            if len(audio_buffer) < LIVE_WINDOW_SECONDS:
                continue

            if analysis_in_progress:
                continue

            window_chunks = list(audio_buffer)

            # Consume one rolling window while retaining the newest
            # portion for the next window.
            #
            # Removing one chunk creates a 4-second overlap between
            # consecutive 5-second windows when chunks are ~1 second.
            oldest = audio_buffer.popleft()
            buffered_bytes -= len(oldest)

            window_data = b"".join(window_chunks)

            if not window_data:
                continue

            sequence_number += 1
            analysis_in_progress = True

            analysis_started = time.perf_counter()

            try:
                await _send_json(
                    websocket,
                    {
                        "type": "WINDOW_READY",
                        "connection_id": connection_id,
                        "sequence_number": sequence_number,
                        "window_seconds": LIVE_WINDOW_SECONDS,
                        "buffered_bytes": buffered_bytes,
                    },
                )

                result = await asyncio.to_thread(
                    _process_live_window,
                    window_data,
                    connection_id,
                    sequence_number,
                )

                result["transport_processing_ms"] = round(
                    (time.perf_counter() - analysis_started) * 1000,
                    1,
                )

                await _send_json(websocket, result)

                logger.info(
                    "Live rolling analysis completed "
                    "connection_id=%s sequence=%s prediction=%s "
                    "confidence=%.2f action=%s processing_ms=%.1f",
                    connection_id,
                    sequence_number,
                    result["detection"]["prediction"],
                    result["detection"]["confidence"],
                    result["risk"]["action"],
                    result["processing_ms"],
                )

            except Exception as exc:
                logger.exception(
                    "Live rolling window analysis failed "
                    "connection_id=%s sequence=%s",
                    connection_id,
                    sequence_number,
                )

                await _send_json(
                    websocket,
                    {
                        "type": "ANALYSIS_ERROR",
                        "connection_id": connection_id,
                        "sequence_number": sequence_number,
                        "message": (
                            "Live audio analysis failed for this "
                            "window. The stream remains active."
                        ),
                        "detail": str(exc)[:500],
                    },
                )

            finally:
                analysis_in_progress = False

    except WebSocketDisconnect:
        logger.info(
            "Live websocket disconnected connection_id=%s",
            connection_id,
        )

    except Exception:
        logger.exception(
            "Live websocket failure connection_id=%s",
            connection_id,
        )

        try:
            await websocket.close(
                code=1011,
                reason="Live stream failure",
            )
        except Exception:
            pass

    finally:
        live_session_service.revoke(token)

        logger.info(
            "Live session revoked connection_id=%s",
            connection_id,
        )
