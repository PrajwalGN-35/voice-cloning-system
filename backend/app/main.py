from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.services.audio_service import audio_service
from backend.app.utils.paths import initialize_directories

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_directories()
    yield

app = FastAPI(
    title="AI Voice Cloning System",
    description="AI-powered authorized voice cloning backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
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

@app.post("/api/v1/audio/upload")
async def upload_audio(file: UploadFile = File(...)):
    result = await audio_service.process_upload(file)
    return {
        "success": True,
        "message": "Audio uploaded and preprocessed successfully",
        **result
    }
