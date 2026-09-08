from backend.app.security.domain_calibration import DomainCalibration


def test_real_microphone_overlap_requires_verification():
    c = DomainCalibration()
    result = c.calibrate(
        real_probability=0.001038,
        fake_probability=0.998962,
        quality="GOOD",
    )
    assert result["decision"] == "VERIFY"


def test_generated_fake_remains_blocked():
    c = DomainCalibration()
    result = c.calibrate(
        real_probability=0.000078,
        fake_probability=0.999922,
        quality="GOOD",
    )
    assert result["decision"] == "BLOCK"


def test_degraded_audio_requires_verification():
    c = DomainCalibration()
    result = c.calibrate(
        real_probability=0.001,
        fake_probability=0.998,
        quality="DEGRADED",
    )
    assert result["decision"] == "VERIFY"
