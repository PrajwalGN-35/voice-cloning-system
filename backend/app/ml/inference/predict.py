import logging
from pathlib import Path
from typing import Dict, Optional

import torch

from backend.app.ml.models.aasist import AASIST
from backend.app.ml.preprocessing.audio_preprocessing import AudioPreprocessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceSpoofingDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)
        self.threshold = threshold

        logger.info(
            f"VoiceSpoofingDetector initialized "
            f"on device: {self.device}"
        )

        self.preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_mels=128,
            duration=5.0,
        )

        # This is the same model used in train.py
        self.model = AASIST(num_classes=2)
        self.model.to(self.device)
        self.model.eval()

        if model_path is not None:
            self.load_model(model_path)
        else:
            logger.warning(
                "No trained checkpoint was loaded. "
                "Predictions are unreliable."
            )

    def load_model(self, model_path: str):
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
                self.threshold = float(
                    checkpoint["threshold"]
                )
        else:
            model_state = checkpoint

        self.model.load_state_dict(
            model_state,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Checkpoint loaded successfully: "
            f"{checkpoint_path}"
        )

    @torch.no_grad()
    def predict(
        self,
        audio_path: str,
        return_raw: bool = False,
    ) -> Dict:
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

        # Shape:
        # (128, 501) -> (1, 1, 128, 501)
        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        logits = self.model(input_tensor)

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

        if probability_fake >= self.threshold:
            prediction = "FAKE"
            confidence = probability_fake * 100
        else:
            prediction = "REAL"
            confidence = probability_real * 100

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

        return result

    def predict_batch(self, audio_paths):
        results = []

        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
            except Exception as error:
                result = {
                    "audio_path": str(audio_path),
                    "prediction": "ERROR",
                    "error": str(error),
                }

            results.append(result)

        return results


_detector_instance = None


def get_detector(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    threshold: float = 0.5,
):
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
    threshold: float = 0.5,
):
    detector = get_detector(
        model_path=model_path,
        device=device,
        threshold=threshold,
    )

    return detector.predict(audio_path)

