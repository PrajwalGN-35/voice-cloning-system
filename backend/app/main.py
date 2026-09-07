from contextlib import asynccontextmanager
from pathlib import Path
import logging

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
    Request,
    Depends,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.app.config import settings
from backend.app.security import (
    require_api_key,
    rate_limit_analyze,
    rate_limit_clone,
    security_headers_middleware,
)
from backend.app.services.audio_service import audio_service
from backend.app.services.model_service import model_service
from backend.app.services.voice_service import voice_service
from backend.app.services.deepfake_service import get_deepfake_service
from backend.app.security.security_service import process_voice_security
from backend.app.utils.paths import initialize_directories
from backend.app.errors.exceptions import AppError
from backend.app.errors.handlers import (
    app_error_handler,
    create_request_id,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_directories()
    model_service.load_model()
    yield


app = FastAPI(
    title=settings.app_name,
    description="AI-powered authorized voice cloning and voice authenticity analysis backend",
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    request_id = getattr(
        request.state,
        "request_id",
        create_request_id(),
    )

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "The request contains invalid or missing fields.",
                "request_id": request_id,
            },
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):
    request_id = getattr(
        request.state,
        "request_id",
        create_request_id(),
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "request_id": request_id,
            },
        },
        headers={"X-Request-ID": request_id},
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or create_request_id()
    request.state.request_id = request_id

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled request failure request_id=%s path=%s",
            request_id,
            request.url.path,
        )
        raise

    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def apply_security_headers(request: Request, call_next):
    return await security_headers_middleware(request, call_next)


app.add_exception_handler(AppError, app_error_handler)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.frontend_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "X-API-Key",
        "X-Request-ID",
    ],
)


@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/api/v1/ready")
def readiness():
    model_loaded = model_service.is_loaded()

    if not model_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "status": "not_ready",
                "service": settings.app_name,
                "model_loaded": False,
            },
        )

    return {
        "success": True,
        "status": "ready",
        "service": settings.app_name,
        "model_loaded": True,
    }


@app.get("/api/v1/model/status")
def model_status():
    return {
        "model": model_service.model_name,
        "device": model_service.device,
        "loaded": model_service.is_loaded(),
    }


@app.post(
    "/api/v1/audio/upload",
    dependencies=[
        Depends(require_api_key),
    ],
)
async def upload_audio(file: UploadFile = File(...)):
    result = await audio_service.process_upload(file)

    return {
        "success": True,
        "message": "Audio uploaded and preprocessed successfully",
        **result,
    }


@app.post(
    "/api/v1/voice/analyze",
    dependencies=[
        Depends(require_api_key),
        Depends(rate_limit_analyze),
    ],
)
async def analyze_voice(file: UploadFile = File(...)):
    """
    Analyze uploaded audio for voice authenticity.

    Pipeline:
    Audio preprocessing -> Member 5 deepfake detector -> Member 4 risk engine.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required",
        )

    uploaded = await audio_service.process_upload(file)

    try:
        detector = get_deepfake_service()
        detection = detector.analyze(uploaded["processed_file"])

        security = process_voice_security(
            detection["prediction"].lower(),
            detection["confidence"],
        )

        original_probability = detection["original_probability"]
        fake_probability = detection["fake_probability"]

        return {
            "success": True,
            "prediction": detection["prediction"],
            "original_probability": original_probability,
            "fake_probability": fake_probability,
            "original_percentage": round(
                original_probability * 100,
                4,
            ),
            "fake_percentage": round(
                fake_probability * 100,
                4,
            ),
            "confidence": detection["confidence"],
            "risk_level": security["risk_level"],
            "action": security["action"],
            "message": security["message"],
            "verification_required": security["verification_required"],
            "verification_method": security["verification_method"],
        }

    finally:
        processed_file = uploaded.get("processed_file")

        if processed_file:
            audio_service.cleanup_processed_file(processed_file)


@app.post(
    "/api/v1/voice/clone",
    dependencies=[
        Depends(require_api_key),
        Depends(rate_limit_clone),
    ],
    response_class=FileResponse,
    responses={
        200: {
            "description": "Generated cloned voice audio",
            "content": {
                "audio/wav": {},
            },
        },
        400: {
            "description": "Invalid audio or text",
        },
        500: {
            "description": "Voice generation failed",
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
            detail="Text cannot be empty",
        )

    if not reference_audio.filename:
        raise HTTPException(
            status_code=400,
            detail="Reference audio file is required",
        )

    uploaded = await audio_service.process_upload(reference_audio)

    try:
        result = voice_service.generate_voice(
            reference_file=uploaded["processed_file"],
            text=text.strip(),
        )

        output_file = Path(result["output_file"])

        if not output_file.exists():
            raise HTTPException(
                status_code=500,
                detail="Voice generation completed but output file was not found",
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

    finally:
        processed_file = uploaded.get("processed_file")

        if processed_file:
            audio_service.cleanup_processed_file(processed_file)
