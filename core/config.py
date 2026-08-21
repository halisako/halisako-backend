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

    # --- Animation generation ---
    # "mock" is the safe default for local development and testing.
    # Switch to a real provider once one is implemented (Sprint 4,
    # later prompt) and credentials are configured.
    animation_provider: str = "mock"
    animation_output_dir: str = "generated_animations"

    # --- ComfyUI / Wan 2.2 5B animation provider (Sprint 4 Prompt 3) ---
    # "http://localhost:8188" is ComfyUI's own standard default when run
    # locally — a reasonable development default, not a hardcoded
    # production URL; a real deployment overrides this via environment
    # variable. No credentials here: this environment has no ComfyUI
    # installation to authenticate against (see this feature's
    # engineering report), and the local ComfyUI API these calls target
    # does not require one by default.
    comfyui_base_url: str = "http://localhost:8188"
    comfyui_timeout_seconds: float = 300.0
    comfyui_workflow_path: str = "products/chess2fight/rendering/workflows/wan22_i2v_5b.json"
    # Used only to convert AnimationInstruction.duration_seconds into a
    # frame count when instruction.fps is unset. VERIFIED (Sprint 4
    # Prompt 4): the supplied, experimentally-validated wan22_i2v_5b.json
    # workflow's own node 57 (CreateVideo) has fps=24 — this is no
    # longer a placeholder guess, it's read directly from that node.
    comfyui_default_fps: int = 24

    # --- ComfyUI / FLUX image provider (Sprint 4 Prompt 5) ---
    # Reuses comfyui_base_url / comfyui_timeout_seconds above rather
    # than duplicating them - the same ComfyUI server is expected to
    # eventually serve both FLUX (images) and Wan (animation) on one
    # GPU worker, per this task's own explicit instruction.
    comfyui_image_workflow_path: str = "products/chess2fight/rendering/workflows/flux2_klein_t2i_4b.json"
    # 1280x704 - exactly 2x the experimentally-validated Wan 2.2 5B
    # resolution (640x352, from Prompt 4), so a FLUX keyframe needs no
    # cropping or aspect-ratio distortion before Wan conditioning.
    # Divisible by 16 (Wan's own confirmed alignment requirement) and
    # independently cited elsewhere as a common Wan resolution - not
    # just derived by doubling. Not verified against a live FLUX
    # instance in this environment.
    comfyui_image_default_width: int = 1280
    comfyui_image_default_height: int = 704

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

render_storage_root: str = "storage"