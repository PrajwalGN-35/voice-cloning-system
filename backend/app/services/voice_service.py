from pathlib import Path

from fastapi import HTTPException

from backend.app.config import settings
from backend.app.services.model_service import model_service


class VoiceService:

    def generate_voice(self, reference_file: str, text: str):
        if not model_service.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Voice cloning model is not loaded"
            )

        reference_path = Path(reference_file)

        if not reference_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Reference audio not found"
            )

        text = text.strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )

        try:
            output_name = f"generated_{reference_path.stem}.wav"
            output_path = settings.generated_dir / output_name

            wavs, sample_rate = model_service.model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=str(reference_path),
                x_vector_only_mode=True,
            )

            if wavs is None or len(wavs) == 0:
                raise RuntimeError("Model did not generate audio")

            import soundfile as sf

            sf.write(
                str(output_path),
                wavs[0],
                sample_rate,
            )

            if not output_path.exists():
                raise RuntimeError("Generated audio file was not created")

            return {
                "success": True,
                "output_file": str(output_path),
                "sample_rate": sample_rate,
                "text": text,
            }

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Voice generation failed: {exc}"
            )


voice_service = VoiceService()
