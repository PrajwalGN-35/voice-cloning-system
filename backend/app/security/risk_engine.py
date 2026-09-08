def assess_risk(prediction, confidence):
    """
    Analyze the AI voice detection result
    and decide the security action.
    """

    # Validate prediction
    if not isinstance(prediction, str):
        return {
            "risk_level": "HIGH",
            "action": "BLOCK",
            "message": "Invalid prediction received.",
            "verification_required": True,
            "verification_method": "OTP + SECONDARY_CHECK"
        }

    prediction = prediction.lower().strip()

    # Validate confidence
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 100:
        return {
            "risk_level": "HIGH",
            "action": "BLOCK",
            "message": "Invalid confidence score received.",
            "verification_required": True,
            "verification_method": "OTP + SECONDARY_CHECK"
        }

    # Validate prediction value
    if prediction not in ["real", "fake"]:
        return {
            "risk_level": "HIGH",
            "action": "BLOCK",
            "message": "Unknown voice prediction received.",
            "verification_required": True,
            "verification_method": "OTP + SECONDARY_CHECK"
        }

    # HIGH RISK
    if prediction == "fake" and confidence >= 80:
        return {
            "risk_level": "HIGH",
            "action": "BLOCK",
            "message": "Possible AI-generated voice detected.",
            "verification_required": True,
            "verification_method": "OTP + SECONDARY_CHECK"
        }

    # MEDIUM RISK
    elif prediction == "fake" and confidence >= 50:
        return {
            "risk_level": "MEDIUM",
            "action": "VERIFY",
            "message": "Suspicious voice detected. Additional verification required.",
            "verification_required": True,
            "verification_method": "OTP"
        }

    # LOW RISK
    else:
        return {
            "risk_level": "LOW",
            "action": "ALLOW",
            "message": "Voice appears authentic.",
            "verification_required": False,
            "verification_method": "NONE"
        }
