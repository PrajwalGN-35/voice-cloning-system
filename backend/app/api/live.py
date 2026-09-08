from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi import WebSocket, WebSocketDisconnect

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
    }


def _process_live_chunk(binary_data: bytes, connection_id: str) -> dict:
    """
    Convert one browser MediaRecorder segment into 16 kHz mono WAV,
    then run the existing VoiceGuard detection, reliability, and
    adaptive-risk pipeline.
    """

    started = time.perf_counter()
    chunk_id = uuid4().hex

    input_path = settings.processed_dir / f"live_{chunk_id}.webm"
    wav_path = settings.processed_dir / f"live_{chunk_id}.wav"

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
            "chunk_id": chunk_id,
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
            "message": "Live audio stream connected.",
        },
    )

    logger.info(
        "Live websocket connected connection_id=%s",
        connection_id,
    )

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

            if len(binary_data) == 0:
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

            await _send_json(
                websocket,
                {
                    "type": "CHUNK_RECEIVED",
                    "connection_id": connection_id,
                    "bytes": len(binary_data),
                },
            )

            analysis_started = time.perf_counter()

            try:
                result = await asyncio.to_thread(
                    _process_live_chunk,
                    binary_data,
                    connection_id,
                )

                result["transport_processing_ms"] = round(
                    (time.perf_counter() - analysis_started) * 1000,
                    1,
                )

                await _send_json(websocket, result)

                logger.info(
                    "Live analysis completed connection_id=%s "
                    "prediction=%s confidence=%.2f action=%s "
                    "processing_ms=%.1f",
                    connection_id,
                    result["detection"]["prediction"],
                    result["detection"]["confidence"],
                    result["risk"]["action"],
                    result["processing_ms"],
                )

            except Exception as exc:
                logger.exception(
                    "Live chunk analysis failed connection_id=%s",
                    connection_id,
                )

                await _send_json(
                    websocket,
                    {
                        "type": "ANALYSIS_ERROR",
                        "connection_id": connection_id,
                        "message": (
                            "Live audio analysis failed for this "
                            "window. The stream remains active."
                        ),
                        "detail": str(exc)[:500],
                    },
                )

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
