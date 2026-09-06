from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.services.audio_service import audio_service
from backend.app.services.model_service import model_service
from backend.app.services.voice_service import voice_service
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
        "http://localhost:3000",
        "http://localhost:5173",
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


@app.post(
    "/api/v1/voice/clone",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Generated cloned voice audio",
            "content": {
                "audio/wav": {}
            },
        }
    },
)
async def clone_voice(
    reference_audio: UploadFile = File(...),
    text: str = Form(...),
):
    uploaded = await audio_service.process_upload(reference_audio)

    result = voice_service.generate_voice(
        reference_file=uploaded["processed_file"],
        text=text,
    )

    output_file = Path(result["output_file"])

    if not output_file.exists():
        raise RuntimeError(
            "Voice generation completed but output file was not found"
        )

    return FileResponse(
        path=str(output_file),
        media_type="audio/wav",
        filename=output_file.name,
    )
