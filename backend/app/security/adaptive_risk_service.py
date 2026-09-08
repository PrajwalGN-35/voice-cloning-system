from typing import Dict

from backend.app.security.risk_engine import assess_risk


def assess_adaptive_risk(
    prediction: str,
    confidence: float,
    audio_quality: str,
    quality_flags: list[str],
) -> Dict:
    """
    VoiceGuard adaptive risk layer.

    Member 4 remains responsible for the base security decision.
    This wrapper adds audio reliability awareness without modifying
    the original risk engine.
    """

    base_risk = assess_risk(prediction, confidence)

    quality = str(audio_quality).upper().strip()
    flags = quality_flags if isinstance(quality_flags, list) else []

    # Preserve a high-confidence FAKE decision.
    if base_risk["action"] == "BLOCK":
        result = dict(base_risk)
        result["decision_source"] = "BASE_RISK_ENGINE"
        result["reliability_adjustment"] = False
        return result

    # Poor-quality audio should not receive an unconditional ALLOW.
    if quality == "POOR":
        return {
            "risk_level": "MEDIUM",
            "action": "VERIFY",
            "message": (
                "Voice authenticity result is affected by poor audio "
                "reliability. Additional verification is required."
            ),
            "verification_required": True,
            "verification_method": "OTP + SECONDARY_CHECK",
            "decision_source": "ADAPTIVE_RELIABILITY_LAYER",
            "reliability_adjustment": True,
            "quality_flags": flags,
        }

    # Degraded audio receives additional verification when the
    # underlying engine would otherwise allow the request.
    if quality == "DEGRADED" and base_risk["action"] == "ALLOW":
        return {
            "risk_level": "MEDIUM",
            "action": "VERIFY",
            "message": (
                "Voice appears authentic, but audio quality is degraded. "
                "Additional verification is recommended."
            ),
            "verification_required": True,
            "verification_method": "OTP",
            "decision_source": "ADAPTIVE_RELIABILITY_LAYER",
            "reliability_adjustment": True,
            "quality_flags": flags,
        }

    result = dict(base_risk)
    result["decision_source"] = "BASE_RISK_ENGINE"
    result["reliability_adjustment"] = False
    result["quality_flags"] = flags

    return result
