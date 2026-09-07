from pathlib import Path
from typing import Dict

from backend.app.ml.inference.predict import VoiceSpoofingDetector


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "checkpoints"
    / "latest_calibrated.pth"
)


class DeepfakeDetectionService:
    """Service layer for Member 5 voice spoofing detection."""

    def __init__(self):
        self.detector = VoiceSpoofingDetector(
            model_path=str(MODEL_PATH)
        )

    def analyze(self, audio_path: str) -> Dict:
        """Analyze an audio file and return model prediction."""
        result = self.detector.predict(
            audio_path,
            return_raw=True
        )

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "original_probability": result["probabilities"]["real"],
            "fake_probability": result["probabilities"]["fake"],
            "threshold": result["threshold"],
            "raw_scores": result.get("raw_scores"),
        }


_detector_service = None


def get_deepfake_service() -> DeepfakeDetectionService:
    """Return a singleton detector service."""
    global _detector_service

    if _detector_service is None:
        _detector_service = DeepfakeDetectionService()

    return _detector_service
