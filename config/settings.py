# This file defines the configuration settings for the Voice Scam Shield application.
# It uses Pydantic's dataclasses to structure settings and loads values from environment variables (.env file).

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


@dataclass
class Settings:
    """Dataclass holding all application settings, loaded from environment variables with defaults."""
    # API & Models
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")

    # Audio
    sample_rate: int = _get_int("SAMPLE_RATE", 16000)
    channels: int = _get_int("CHANNELS", 1)
    chunk_seconds: float = _get_float("CHUNK_SECONDS", 5.0)

    # Detection
    scam_sensitivity: float = _get_float("SCAM_SENSITIVITY", 0.6)
    scam_patterns: List[str] = field(
        default_factory=lambda: [
            "gift card",
            "amazon card",
            "bank account",
            "routing number",
            "one-time password",
            "otp",
            "social security",
            "ssn",
            "bitcoin",
            "crypto",
            "wire transfer",
            "western union",
            "remote access",
            "teamviewer",
            "anydesk",
            "logmein",
            "pay immediately",
            "urgent payment",
            "warranty expired",
            "lawsuit",
            "arrest warrant",
            "verify your account",
            "confirm your identity",
            "password reset",
            "limited time",
            "prize",
            "lottery",
            "tax office",
            "irs",
        ]
    )


def get_settings() -> Settings:
    """Factory function to create and return a Settings instance."""
    return Settings()
