from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class DomainCalibration:
    real_fake_mean: float = 0.9980968
    real_fake_min: float = 0.994287
    real_fake_max: float = 0.999341
    reference_fake_mean: float = 0.9994023333
    reference_fake_min: float = 0.998374
    reference_fake_max: float = 0.999922

    def calibrate(
        self,
        real_probability: float,
        fake_probability: float,
        quality: str = "GOOD",
    ) -> Dict:
        fake_probability = max(0.0, min(1.0, float(fake_probability)))
        real_probability = max(0.0, min(1.0, float(real_probability)))
        quality = str(quality).upper()

        if quality == "VERY_POOR":
            return {
                "calibrated_fake_probability": fake_probability,
                "calibrated_real_probability": real_probability,
                "decision": "VERIFY",
                "source": "QUALITY_GATE",
            }

        lo = self.real_fake_min
        hi = self.reference_fake_max

        if hi <= lo:
            return {
                "calibrated_fake_probability": fake_probability,
                "calibrated_real_probability": real_probability,
                "decision": "VERIFY",
                "source": "FALLBACK",
            }

        normalized = (fake_probability - lo) / (hi - lo)
        normalized = max(0.0, min(1.0, normalized))

        calibrated_fake = normalized
        calibrated_real = 1.0 - calibrated_fake

        if quality in {"POOR", "DEGRADED"}:
            decision = "BLOCK" if calibrated_fake >= 0.98 else "VERIFY"
        elif calibrated_fake >= 0.92:
            decision = "BLOCK"
        elif calibrated_fake >= 0.20:
            decision = "VERIFY"
        else:
            decision = "ALLOW"

        return {
            "calibrated_fake_probability": round(calibrated_fake, 6),
            "calibrated_real_probability": round(calibrated_real, 6),
            "decision": decision,
            "source": "MIC_DOMAIN_CALIBRATION",
        }


_calibrator = DomainCalibration()


def calibrate_live_result(result: Dict) -> Dict:
    quality = (
        (result.get("audio_quality") or {})
        .get("quality", "GOOD")
    ).upper()

    calibration = _calibrator.calibrate(
        result.get("real_probability", 0.0),
        result.get("fake_probability", 1.0),
        quality,
    )

    result["calibration"] = calibration

    decision = calibration["decision"]
    risk = result.get("risk") or {}

    if decision == "ALLOW":
        risk.update(
            {
                "risk_level": "LOW",
                "action": "ALLOW",
                "message": "Voice appears consistent with the calibrated live-audio domain.",
                "verification_required": False,
                "verification_method": "NONE",
                "decision_source": "MIC_DOMAIN_CALIBRATION",
                "reliability_adjustment": True,
            }
        )

    elif decision == "VERIFY":
        risk.update(
            {
                "risk_level": "MEDIUM",
                "action": "VERIFY",
                "message": "Voice requires additional verification.",
                "verification_required": True,
                "verification_method": "OTP + SECONDARY_CHECK",
                "decision_source": "MIC_DOMAIN_CALIBRATION",
                "reliability_adjustment": True,
            }
        )

    elif decision == "BLOCK":
        risk.update(
            {
                "risk_level": "HIGH",
                "action": "BLOCK",
                "message": "Possible AI-generated voice detected.",
                "verification_required": True,
                "verification_method": "OTP + SECONDARY_CHECK",
                "decision_source": "MIC_DOMAIN_CALIBRATION",
                "reliability_adjustment": True,
            }
        )

    result["risk"] = risk
    result["risk_level"] = risk.get("risk_level")
    result["action"] = risk.get("action")

    return result
