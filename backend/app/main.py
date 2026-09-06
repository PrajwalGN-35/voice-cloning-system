from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.config import settings
from fastapi import HTTPException

from backend.app.services.audio_service import audio_service
from backend.app.services.model_service import model_service
from backend.app.services.voice_service import voice_service
from backend.app.services.deepfake_service import get_deepfake_service
from backend.app.security.security_service import process_voice_security
from backend.app.schemas.voice import VoiceCloneResponse
from backend.app.utils.paths import initialize_directories


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_directories()
    model_service.load_model()
    yield


app = FastAPI(
    title="AI Voice Cloning System",
    description="AI-powered authorized voice cloning backend",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.frontend_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "AI Voice Cloning System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "service": "AI Voice Cloning System",
        "version": "1.0.0",
    }


@app.get("/api/v1/model/status")
def model_status():
    return {
        "model": model_service.model_name,
        "device": model_service.device,
        "loaded": model_service.is_loaded(),
    }


@app.post("/api/v1/audio/upload")
async def upload_audio(file: UploadFile = File(...)):
    result = await audio_service.process_upload(file)

    return {
        "success": True,
        "message": "Audio uploaded and preprocessed successfully",
        **result,
    }



@app.post("/api/v1/voice/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    """
    Analyze uploaded audio for voice authenticity.

    Pipeline:
    Audio preprocessing → Member 5 deepfake detector → Member 4 risk engine.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required"
        )

    # Reuse the existing audio upload and preprocessing pipeline.
    uploaded = await audio_service.process_upload(file)

    # Run Member 5 deepfake detector.
    detector = get_deepfake_service()
    detection = detector.analyze(uploaded["processed_file"])

    # Run Member 4 security/risk engine.
    security = process_voice_security(
        detection["prediction"].lower(),
        detection["confidence"]
    )

    original_probability = detection["original_probability"]
    fake_probability = detection["fake_probability"]

    return {
        "success": True,
        "prediction": detection["prediction"],
        "original_probability": original_probability,
        "fake_probability": fake_probability,
        "original_percentage": round(original_probability * 100, 4),
        "fake_percentage": round(fake_probability * 100, 4),
        "confidence": detection["confidence"],
        "risk_level": security["risk_level"],
        "action": security["action"],
        "message": security["message"],
        "verification_required": security["verification_required"],
        "verification_method": security["verification_method"],
    }

@app.post(
    "/api/v1/voice/clone",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Generated cloned voice audio",
            "content": {
                "audio/wav": {}
            },
        },
        400: {
            "description": "Invalid audio or text"
        },
        500: {
            "description": "Voice generation failed"
        },
    },
)
async def clone_voice(
    reference_audio: UploadFile = File(...),
    text: str = Form(...),
):
    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    if not reference_audio.filename:
        raise HTTPException(
            status_code=400,
            detail="Reference audio file is required"
        )

    uploaded = await audio_service.process_upload(reference_audio)

    result = voice_service.generate_voice(
        reference_file=uploaded["processed_file"],
        text=text.strip(),
    )

    output_file = Path(result["output_file"])

    if not output_file.exists():
        raise HTTPException(
            status_code=500,
            detail="Voice generation completed but output file was not found"
        )

    return FileResponse(
        path=str(output_file),
        media_type="audio/wav",
        filename=output_file.name,
        headers={
            "X-Voice-Clone-Sample-Rate": str(result["sample_rate"]),
            "X-Voice-Clone-Text-Length": str(len(text.strip())),
        },
    )

