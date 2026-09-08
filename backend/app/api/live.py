from __future__ import annotations
from backend.app.security.domain_calibration import calibrate_live_result

import asyncio
import io
import subprocess
import time
import wave
from pathlib import Path

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.security.auth import require_api_key
from backend.app.services.audio_quality_service import AudioQualityService
from backend.app.services.deepfake_service import DeepfakeDetectionService
from backend.app.services.live_session_service import live_session_service
from backend.app.security.adaptive_risk_service import assess_adaptive_risk

router = APIRouter(prefix="/api/v1/live", tags=["live"])

LIVE_SAMPLE_RATE = 16000
LIVE_CHANNELS = 1
LIVE_SAMPLE_WIDTH = 2
LIVE_WINDOW_SECONDS = 5
MAX_PCM_BYTES = 8 * 1024 * 1024


def _pcm_to_wav(pcm_bytes: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(LIVE_CHANNELS)
        wav.setsampwidth(LIVE_SAMPLE_WIDTH)
        wav.setframerate(LIVE_SAMPLE_RATE)
        wav.writeframes(pcm_bytes)
    return output.getvalue()


def _extract_latest_window(pcm_bytes: bytes) -> bytes:
    required_bytes = (
        LIVE_SAMPLE_RATE
        * LIVE_WINDOW_SECONDS
        * LIVE_CHANNELS
        * LIVE_SAMPLE_WIDTH
    )

    if len(pcm_bytes) < required_bytes:
        raise RuntimeError("Not enough live PCM audio for a 5-second analysis window.")

    return _pcm_to_wav(pcm_bytes[-required_bytes:])


def _analyze_window(wav_bytes: bytes, sequence_number: int) -> dict:
    started = time.perf_counter()

    detector = DeepfakeDetectionService()
    quality_service = AudioQualityService()

    processed_dir = Path(settings.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    window_path = (
        processed_dir
        / f"live_window_{sequence_number}_{int(time.time() * 1000)}.wav"
    )

    try:
        window_path.write_bytes(wav_bytes)

        detection = detector.analyze(str(window_path))
        quality = quality_service.analyze(str(window_path))

        prediction = str(detection.get("prediction", "UNKNOWN"))
        confidence = float(detection.get("confidence", 0.0))
        fake_probability = float(
            detection.get("fake_probability", 0.0)
        )
        real_probability = float(
            detection.get("original_probability", 0.0)
        )
        threshold = float(detection.get("threshold", 0.0))

        quality_level = (
            quality.get("quality")
            or quality.get("quality_level")
            or quality.get("status")
            or "GOOD"
        )

        quality_flags = (
            quality.get("quality_flags")
            or quality.get("flags")
            or []
        )

        risk = assess_adaptive_risk(
            prediction=prediction,
            confidence=confidence,
            audio_quality=str(quality_level),
            quality_flags=quality_flags,
        )

        processing_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return {
            "type": "ANALYSIS_RESULT",
            "success": True,
            "window_id": f"live-{sequence_number}",
            "sequence_number": sequence_number,
            "window_seconds": LIVE_WINDOW_SECONDS,
            "processing_ms": processing_ms,
            "prediction": prediction,
            "confidence": confidence,
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "threshold": threshold,
            "audio_quality": quality,
            "risk": risk,
            "risk_level": risk.get("risk_level", "UNKNOWN"),
            "action": risk.get("action", "UNKNOWN"),
        }

    finally:
        try:
            window_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/session")
async def create_live_session(
    _: object = Depends(require_api_key),
):
    token = live_session_service.create_session()

    return JSONResponse(
        {
            "success": True,
            "token": token,
            "expires_in_seconds": 900,
        }
    )


@router.get("/status")
async def live_status(
    _: object = Depends(require_api_key),
):
    return {
        "success": True,
        "service": "live_voice_protection",
        "status": "ready",
        "sample_rate": LIVE_SAMPLE_RATE,
        "window_seconds": LIVE_WINDOW_SECONDS,
        "transport": "websocket_pcm",
    }


@router.websocket("/ws")
async def live_websocket(websocket: WebSocket):
    origin = websocket.headers.get("origin")

    allowed_origins = [
        item.strip()
        for item in settings.frontend_origins.split(",")
        if item.strip()
    ]

    if origin and origin not in allowed_origins:
        await websocket.close(code=1008)
        return

    token = websocket.query_params.get("token")

    if not token or not live_session_service.validate(token):
        await websocket.close(code=1008)
        return

    await websocket.accept()

    buffer = bytearray()
    sequence_number = 0
    analyzing = False

    await websocket.send_json(
        {
            "type": "CONNECTED",
            "success": True,
            "window_seconds": LIVE_WINDOW_SECONDS,
            "windowing": "rolling",
            "transport": "pcm_s16le",
            "sample_rate": LIVE_SAMPLE_RATE,
        }
    )

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            text = message.get("text")

            if text:
                if text == "PING":
                    await websocket.send_json(
                        {
                            "type": "PONG",
                            "success": True,
                        }
                    )
                continue

            data = message.get("bytes")

            if not data:
                continue

            if len(data) > MAX_PCM_BYTES:
                await websocket.send_json(
                    {
                        "type": "ERROR",
                        "success": False,
                        "message": "Live PCM chunk exceeds the allowed size.",
                    }
                )
                continue

            buffer.extend(data)

            await websocket.send_json(
                {
                    "type": "CHUNK_RECEIVED",
                    "success": True,
                    "bytes": len(data),
                    "buffer_bytes": len(buffer),
                }
            )

            required_bytes = (
                LIVE_SAMPLE_RATE
                * LIVE_WINDOW_SECONDS
                * LIVE_CHANNELS
                * LIVE_SAMPLE_WIDTH
            )

            if len(buffer) < required_bytes or analyzing:
                continue

            sequence_number += 1

            window_pcm = bytes(buffer[-required_bytes:])

            keep_bytes = (
                LIVE_SAMPLE_RATE
                * 1
                * LIVE_CHANNELS
                * LIVE_SAMPLE_WIDTH
            )

            buffer = bytearray(buffer[-(required_bytes - keep_bytes):])

            await websocket.send_json(
                {
                    "type": "WINDOW_READY",
                    "success": True,
                    "sequence_number": sequence_number,
                    "window_seconds": LIVE_WINDOW_SECONDS,
                }
            )

            analyzing = True

            async def analyze_and_send(
                pcm_window: bytes,
                seq: int,
            ):
                nonlocal analyzing

                try:
                    wav_bytes = _pcm_to_wav(pcm_window)

                    result = await asyncio.to_thread(
                        _analyze_window,
                        wav_bytes,
                        seq,
                    )

                    result = calibrate_live_result(result)
                    await websocket.send_json(result)

                except Exception as error:
                    await websocket.send_json(
                        {
                            "type": "ANALYSIS_ERROR",
                            "success": False,
                            "sequence_number": seq,
                            "message": str(error),
                        }
                    )

                finally:
                    analyzing = False

            asyncio.create_task(
                analyze_and_send(
                    window_pcm,
                    sequence_number,
                )
            )

    except WebSocketDisconnect:
        live_session_service.revoke(token)

    except Exception:
        live_session_service.revoke(token)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
