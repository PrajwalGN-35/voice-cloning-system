from datetime import datetime
import json


def log_security_event(prediction, confidence, risk_result):
    """
    Record a voice-security event in the audit log.
    """

    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": risk_result["risk_level"],
        "action": risk_result["action"],
        "verification_required": risk_result["verification_required"],
        "verification_method": risk_result["verification_method"]
    }

    with open("security_audit.log", "a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")

    return event