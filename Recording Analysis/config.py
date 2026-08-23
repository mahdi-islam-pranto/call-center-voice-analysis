"""
Central configuration for the call-recording analyzer.

All values can be overridden via environment variables or a `.env` file
(see .env.example). Keeping the model name here means you can move from
gemini-3.1-flash-lite to any other Gemini model later without touching code.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_api_key: str
    gemini_model: str = "gemini-3.1-flash-lite"

    # Gemini "Files API" upload polling (recordings are processed async on Google's side)
    file_processing_poll_interval_sec: float = 2.0
    file_processing_timeout_sec: float = 120.0

    # Reject absurdly large uploads before we ever touch the network
    max_audio_size_mb: int = 200

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
