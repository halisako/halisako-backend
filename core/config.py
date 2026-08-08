"""
Application configuration, read from environment variables (and a local
.env file in development — see .env.example). Import `get_settings()`
rather than instantiating Settings() directly, so the whole app shares
one cached instance.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App metadata ---
    app_name: str = "Halisako Chess2Fight API"
    app_version: str = "0.1.0"
    environment: str = "development"  # "development" | "production"
    log_level: str = "INFO"

    # --- CORS ---
    # Comma-separated in the env var, e.g. "https://halisako.com,http://localhost:3000"
    cors_origins: str = "http://localhost:3000,https://halisako.com"

    # --- AI provider selection ---
    # "template" requires no credentials and is the safe default so the
    # service works immediately after deploy. Switch to a real provider
    # once credentials are configured.
    ai_provider: str = "template"  # "template" | "openai" | "anthropic" | "gemini" | "local"


    # OpenAI — verify current model names at platform.openai.com/docs/models
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_image_model: str = "dall-e-3"

    # Anthropic — verify current model names at docs.claude.com
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    # Gemini — uses the current `google-genai` SDK (the older
    # `google-generativeai` package is fully deprecated). Verify the
    # current model name at ai.google.dev/gemini-api/docs/models before
    # relying on this default; Google's lineup moves fast.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    # Local / self-hosted model server (e.g. Ollama). No API key needed;
    # point local_model_url at whatever's serving an OpenAI- or
    # Ollama-style completion endpoint.
    local_model_url: str = "http://localhost:11434/api/generate"
    local_model: str = "llama3"

    # Generic timeout for any outbound AI provider call, in seconds.
    ai_request_timeout: float = 20.0

        # --- Image generation ---
    # "mock" is the safe default for local development and testing.
    # Switch to a real provider once credentials are configured.
    image_provider: str = "mock"
    image_output_dir: str = "generated_images"

    # --- Render storage ---
    render_storage_root: str = "storage"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

render_storage_root: str = "storage"