from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def load_backend_environment() -> None:
    load_dotenv(BACKEND_ROOT / ".env")


def get_groq_api_key() -> str | None:
    load_backend_environment()
    value = os.getenv("GROQ_API_KEY", "").strip()
    return value or None
