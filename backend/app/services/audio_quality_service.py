from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf


class AudioQualityService:
    """
    Lightweight audio reliability assessment for VoiceGuard.

    This layer does NOT determine whether a voice is real or fake.
    It determines whether the audio itself is reliable enough for
    downstream authenticity decisions.
    """

    def analyze(self, audio_path: str) -> Dict:
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, sample_rate = sf.read(str(path), always_2d=False)

        if audio is None or len(audio) == 0:
            raise ValueError("Audio contains no samples.")

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        duration = len(audio) / float(sample_rate)

        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))

        silence_threshold = 0.01
        silence_ratio = float(
            np.mean(np.abs(audio) < silence_threshold)
        )

        clipping_threshold = 0.999
        clipping_ratio = float(
            np.mean(np.abs(audio) >= clipping_threshold)
        )

        peak = float(np.max(np.abs(audio)))

        flags = []

        if duration < 1.0:
            flags.append("VERY_SHORT_AUDIO")
        elif duration < 3.0:
            flags.append("SHORT_AUDIO")

        if silence_ratio >= 0.80:
            flags.append("HIGH_SILENCE")

        if clipping_ratio >= 0.01:
            flags.append("CLIPPING_DETECTED")

        if rms < 0.005:
            flags.append("VERY_LOW_SIGNAL")

        if rms > 0.50:
            flags.append("VERY_HIGH_SIGNAL")

        if (
            "VERY_SHORT_AUDIO" in flags
            or "HIGH_SILENCE" in flags
            or "VERY_LOW_SIGNAL" in flags
            or "CLIPPING_DETECTED" in flags
        ):
            quality = "POOR"

        elif (
            "SHORT_AUDIO" in flags
            or "VERY_HIGH_SIGNAL" in flags
            or len(flags) > 0
        ):
            quality = "DEGRADED"

        else:
            quality = "GOOD"

        return {
            "quality": quality,
            "duration_seconds": round(duration, 3),
            "sample_rate": int(sample_rate),
            "channels": 1,
            "rms": round(rms, 6),
            "peak_amplitude": round(peak, 6),
            "silence_ratio": round(silence_ratio, 4),
            "clipping_ratio": round(clipping_ratio, 6),
            "flags": flags,
        }
