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
- reference_audio - authorized reference voice recording
- text - text to synthesize

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
