# Voice Cloning API Integration

## Base URL

http://127.0.0.1:8001

## Swagger

http://127.0.0.1:8001/docs

## Health Check

GET /api/v1/health

## Model Status

GET /api/v1/model/status

## Voice Cloning

POST /api/v1/voice/clone

Multipart fields:

* reference_audio - authorized reference voice recording
* text - text to synthesize

## Response

HTTP 200 - audio/wav

## Errors

400 - Invalid audio or text
422 - Validation error
500 - Voice generation failure

## CORS

Configured using FRONTEND_ORIGINS.

## Authorization

Only use reference audio when you have authorization to clone the speaker's voice.

# Voice Authenticity Analysis API



## POST /api/v1/voice/analyze



Analyzes an uploaded audio file through the integrated voice authenticity pipeline.



### Processing Pipeline



Audio Upload -> Audio Validation -> Member 5 Deepfake Detector -> REAL/FAKE Prediction -> Member 4 Risk Engine -> Risk Level -> ALLOW/VERIFY/BLOCK



### Request



The endpoint accepts an audio file using multipart form data.



file: <audio file>



Example:



curl -X POST http://127.0.0.1:8001/api/v1/voice/analyze -F file=@sample.wav



### Response



{

&#x20; "success": true,

&#x20; "prediction": "FAKE",

&#x20; "original_probability": 0.000002,

&#x20; "fake_probability": 0.999998,

&#x20; "original_percentage": 0.0002,

&#x20; "fake_percentage": 99.9998,

&#x20; "confidence": 100.0,

&#x20; "risk_level": "HIGH",

&#x20; "action": "BLOCK",

&#x20; "message": "Possible AI-generated voice detected.",

&#x20; "verification_required": true,

&#x20; "verification_method": "OTP + SECONDARY_CHECK"

}



The example response shows the API contract. Actual probability and confidence values are produced by the integrated model.



### Team Integration



- Member 5 provides the AASIST-based voice spoofing/deepfake detector.

- Member 4 provides the security risk engine.

- The backend combines both modules into a single API response.

- The existing audio preprocessing pipeline is reused.



### Model



The calibrated detector checkpoint is:



backend/app/ml/checkpoints/latest_calibrated.pth



### Testing



Integration tests:



tests/test_voice_analyze.py



Committed FAKE test fixture:



tests/fixtures/fake_voice.flac



REAL audio testing can be performed locally with the VOICE_REAL_TEST_AUDIO environment variable. Personal recordings are not committed to the repository.

