from __future__ import annotations

import secrets
import time
from dataclasses import dataclass


@dataclass
class LiveSession:
    token: str
    created_at: float
    last_seen: float


class LiveSessionService:
    def __init__(self, ttl_seconds: int = 900):
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, LiveSession] = {}

    def create_session(self) -> str:
        self._cleanup()

        token = secrets.token_urlsafe(32)
        now = time.time()

        self._sessions[token] = LiveSession(
            token=token,
            created_at=now,
            last_seen=now,
        )

        return token

    def validate(self, token: str) -> bool:
        self._cleanup()

        session = self._sessions.get(token)

        if session is None:
            return False

        session.last_seen = time.time()
        return True

    def revoke(self, token: str) -> None:
        self._sessions.pop(token, None)

    def _cleanup(self) -> None:
        now = time.time()

        expired = [
            token
            for token, session in self._sessions.items()
            if now - session.last_seen > self.ttl_seconds
        ]

        for token in expired:
            self._sessions.pop(token, None)


live_session_service = LiveSessionService()
