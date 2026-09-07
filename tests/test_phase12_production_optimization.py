import torch
import pytest

from backend.app.ml.inference.predict import VoiceSpoofingDetector


CHECKPOINT = (
    "backend/app/ml/checkpoints/latest_calibrated.pth"
)

REAL_AUDIO = (
    r"C:\Users\Prajwal G N\Desktop\member5-voice-detector\test_audio.wav"
)

FAKE_AUDIO = (
    r"C:\Users\Prajwal G N\Desktop\member5-voice-detector"
    r"\dataset\LA_demo\test\LA_test_0003_0000000.flac"
)


def test_calibrated_threshold_is_loaded():
    detector = VoiceSpoofingDetector(
        model_path=CHECKPOINT
    )

    assert detector.threshold == pytest.approx(
        0.9912435405405405
    )


def test_probability_validation_accepts_valid_output():
    VoiceSpoofingDetector._validate_probabilities(
        0.25,
        0.75,
    )


def test_probability_validation_rejects_invalid_sum():
    with pytest.raises(RuntimeError):
        VoiceSpoofingDetector._validate_probabilities(
            0.25,
            0.60,
        )


def test_probability_validation_rejects_out_of_range():
    with pytest.raises(RuntimeError):
        VoiceSpoofingDetector._validate_probabilities(
            1.2,
            -0.2,
        )


def test_real_audio_regression():
    detector = VoiceSpoofingDetector(
        model_path=CHECKPOINT
    )

    result = detector.predict(
        REAL_AUDIO,
        return_raw=True,
    )

    assert result["prediction"] == "REAL"
    assert result["probabilities"]["real"] == pytest.approx(
        0.653503,
        abs=1e-5,
    )
    assert result["probabilities"]["fake"] == pytest.approx(
        0.346497,
        abs=1e-5,
    )
    assert result["inference_time_ms"] >= 0


def test_fake_audio_regression():
    detector = VoiceSpoofingDetector(
        model_path=CHECKPOINT
    )

    result = detector.predict(
        FAKE_AUDIO,
        return_raw=True,
    )

    assert result["prediction"] == "FAKE"
    assert result["probabilities"]["fake"] == pytest.approx(
        0.999998,
        abs=1e-5,
    )
    assert result["probabilities"]["real"] == pytest.approx(
        0.000002,
        abs=1e-5,
    )
    assert result["inference_time_ms"] >= 0


def test_missing_audio_is_rejected():
    detector = VoiceSpoofingDetector(
        model_path=CHECKPOINT
    )

    with pytest.raises(FileNotFoundError):
        detector.predict(
            "this-file-does-not-exist.wav"
        )


def test_model_output_shape_is_validated():
    detector = VoiceSpoofingDetector(
        model_path=CHECKPOINT
    )

    class InvalidModel:
        def __call__(self, _input):
            return torch.zeros((1, 3))

    detector.model = InvalidModel()

    with pytest.raises(RuntimeError):
        detector.predict(REAL_AUDIO)
