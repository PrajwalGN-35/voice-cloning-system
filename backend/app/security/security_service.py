from .risk_engine import assess_risk
from .security_log import log_security_event


def process_voice_security(prediction, confidence):
    """
    Run the risk assessment and record the security event.
    """

    # Step 1: Assess the risk
    risk_result = assess_risk(prediction, confidence)

    # Step 2: Record the security event
    log_security_event(
        prediction,
        confidence,
        risk_result
    )

    # Step 3: Return the complete security result
    return risk_result