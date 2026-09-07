import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from backend.app.ml.models.aasist import AASIST
from backend.app.ml.preprocessing.audio_preprocessing import AudioPreprocessor


logger = logging.getLogger(__name__)


class VoiceSpoofingDetector:
    """Production inference service for the calibrated voice spoofing model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_mels=128,
            duration=5.0,
        )

        self.model = AASIST(num_classes=2)
        self.model.to(self.device)
        self.model.eval()

        self.threshold = (
            float(threshold)
            if threshold is not None
            else 0.5
        )

        logger.info(
            "VoiceSpoofingDetector initialized on device=%s threshold=%s",
            self.device,
            self.threshold,
        )

        if model_path is not None:
            self.load_model(model_path)
        else:
            logger.warning(
                "No trained checkpoint was loaded. "
                "Predictions are unreliable."
            )

    def load_model(self, model_path: str) -> None:
        """Load model weights and calibrated threshold from checkpoint."""

        checkpoint_path = Path(model_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            model_state = checkpoint["model_state_dict"]

            if "threshold" in checkpoint:
                self.threshold = float(checkpoint["threshold"])

        else:
            model_state = checkpoint

        if not 0.0 < self.threshold < 1.0:
            raise ValueError(
                f"Invalid detection threshold: {self.threshold}"
            )

        self.model.load_state_dict(
            model_state,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Checkpoint loaded successfully path=%s threshold=%s",
            checkpoint_path,
            self.threshold,
        )

    @staticmethod
    def _validate_probabilities(
        probability_real: float,
        probability_fake: float,
    ) -> None:
        """Validate the model probability output."""

        values = (
            probability_real,
            probability_fake,
        )

        if not all(torch.isfinite(torch.tensor(value)) for value in values):
            raise RuntimeError(
                "Model produced non-finite probabilities."
            )

        if any(value < 0.0 or value > 1.0 for value in values):
            raise RuntimeError(
                "Model produced probabilities outside [0, 1]."
            )

        if abs(
            (probability_real + probability_fake) - 1.0
        ) > 1e-4:
            raise RuntimeError(
                "Model probabilities do not sum to 1."
            )

    @torch.no_grad()
    def predict(
        self,
        audio_path: str,
        return_raw: bool = False,
    ) -> Dict:
        """Run calibrated voice spoofing inference."""

        start_time = time.perf_counter()

        audio_file = Path(audio_path)

        if not audio_file.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_file}"
            )

        mel_spectrogram, audio_information = (
            self.preprocessor.preprocess(
                str(audio_file),
                verbose=False,
            )
        )

        input_tensor = torch.tensor(
            mel_spectrogram,
            dtype=torch.float32,
        )

        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        logits = self.model(input_tensor)

        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2:
            raise RuntimeError(
                f"Unexpected model output shape: {tuple(logits.shape)}"
            )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        probability_real = float(
            probabilities[0].cpu()
        )

        probability_fake = float(
            probabilities[1].cpu()
        )

        self._validate_probabilities(
            probability_real,
            probability_fake,
        )

        if probability_fake >= self.threshold:
            prediction = "FAKE"
            confidence = probability_fake * 100
        else:
            prediction = "REAL"
            confidence = probability_real * 100

        inference_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        result = {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "fake_score": round(
                probability_fake,
                6,
            ),
            "probabilities": {
                "real": round(
                    probability_real,
                    6,
                ),
                "fake": round(
                    probability_fake,
                    6,
                ),
            },
            "threshold": self.threshold,
            "audio_path": str(audio_file),
            "inference_time_ms": round(
                inference_time_ms,
                2,
            ),
        }

        if return_raw:
            result["raw_scores"] = {
                "real": float(
                    logits[0][0].cpu()
                ),
                "fake": float(
                    logits[0][1].cpu()
                ),
            }

        logger.info(
            "Voice inference completed prediction=%s "
            "confidence=%.2f inference_time_ms=%.2f",
            prediction,
            confidence,
            inference_time_ms,
        )

        return result

    def predict_batch(self, audio_paths):
        """Run inference over multiple audio files."""

        results = []

        for audio_path in audio_paths:
            try:
                results.append(
                    self.predict(audio_path)
                )
            except Exception as error:
                logger.exception(
                    "Batch inference failed for %s",
                    audio_path,
                )
                results.append(
                    {
                        "audio_path": str(audio_path),
                        "prediction": "ERROR",
                        "error": str(error),
                    }
                )

        return results


_detector_instance = None


def get_detector(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    threshold: Optional[float] = None,
):
    """Return the process-wide detector singleton."""

    global _detector_instance

    if _detector_instance is None:
        _detector_instance = VoiceSpoofingDetector(
            model_path=model_path,
            device=device,
            threshold=threshold,
        )

    return _detector_instance


def predict_audio(
    audio_path: str,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    threshold: Optional[float] = None,
):
    """Convenience wrapper for voice spoofing prediction."""

    detector = get_detector(
        model_path=model_path,
        device=device,
        threshold=threshold,
    )

    return detector.predict(audio_path)
