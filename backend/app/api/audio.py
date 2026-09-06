from fastapi import APIRouter, File, UploadFile
from backend.app.services.audio_service import audio_service

router = APIRouter(tags=["Audio"])

@router.post("/api/v1/audio/upload")
async def upload_audio(file: UploadFile = File(...)):
    result = await audio_service.process_upload(file)
    return {
        "success": True,
        "message": "Audio uploaded and preprocessed successfully",
        **result
    }
