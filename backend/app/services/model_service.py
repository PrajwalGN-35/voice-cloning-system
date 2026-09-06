import torch
from qwen_tts import Qwen3TTSModel


class ModelService:
    def __init__(self):
        self.model = None
        self.model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        self.device = "cpu"

    def load_model(self):
        if self.model is not None:
            return

        print("Loading Qwen3-TTS model...")

        self.model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=torch.float32,
        )

        print("Qwen3-TTS model loaded successfully.")

    def is_loaded(self):
        return self.model is not None


model_service = ModelService()
