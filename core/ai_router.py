"""
AI provider abstraction.

Nothing else in this codebase imports openai/anthropic/google-generativeai
directly — every caller (narrative_generator.py) talks to the
`AIProvider` interface, so adding a fifth provider later or swapping
which one is active is a config change, not a code change.

`get_ai_provider()` is the single place that decides which concrete
provider to instantiate, based on Settings.ai_provider.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx

from core.config import Settings, get_settings
from core.exceptions import AIProviderError

logger = logging.getLogger(__name__)


class AIProvider(ABC):
    """Common interface every text-generation backend implements."""

    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        """Return generated text for the given prompt.

        Implementations should raise AIProviderError on failure rather
        than letting SDK-specific exceptions escape, so callers only
        ever need to handle one exception type.
        """
        raise NotImplementedError


class TemplateProvider(AIProvider):
    """
    Deterministic, dependency-free fallback. No network calls, no API
    key, always available — this is what the service uses out of the
    box, and what narrative_generator.py falls back to if a real
    provider errors. It "generates text" by pulling the facts
    narrative_generator already embedded in the prompt (see the
    ANALYSIS FACTS block it writes) back out with a regex and dropping
    them into a few hand-written sentence templates. It never invents
    facts that weren't already in the prompt.
    """

    _FACT_RE = re.compile(r"^([A-Za-z ]+):\s*(.+)$", re.MULTILINE)

    async def generate_text(self, prompt: str) -> str:
        facts = dict(self._FACT_RE.findall(prompt))

        num_moves = int(facts.get("Total moves", 0) or 0)
        num_captures = int(facts.get("Captures", 0) or 0)
        is_checkmate = facts.get("Ends in checkmate", "no").strip().lower() == "yes"
        winner = facts.get("Winner", "White").strip()
        opening = facts.get("Opening", "an unnamed opening").strip()
        is_draw = winner.lower() == "draw"

        intensity = num_captures / max(num_moves, 1)
        if is_draw:
            fight_style = "Evenly Matched Standoff"
        elif is_checkmate and num_moves <= 8:
            fight_style = "Blitz Aggression"
        elif intensity > 0.4:
            fight_style = "Relentless Exchange"
        elif is_checkmate:
            fight_style = "Calculated Finish"
        else:
            fight_style = "Measured Standoff"

        moves_phrase = f"{num_moves} move{'s' if num_moves != 1 else ''}"
        blows_phrase = f"{num_captures} blow{'s' if num_captures != 1 else ''}"

        if is_draw:
            battle_summary = (
                f"Out of the {opening}, both fighters trade {blows_phrase} across "
                f"{moves_phrase}, neither able to land a final blow — the duel ends "
                "in a hard-fought standstill."
            )
        else:
            battle_summary = (
                f"Out of the {opening}, {winner} presses the advantage across "
                f"{moves_phrase}, trading {blows_phrase} before sealing the fight "
                "in a decisive final strike."
            )

        beats = max(num_moves // 3, 1)
        finish_line = (
            "FINISH: Neither fighter yields — the duel ends at a standstill."
            if is_draw
            else f"FINISH: {winner} lands the decisive strike."
        )
        scene_prompt = (
            "SCENE: Neutral arena, dramatic side lighting.\n"
            f"FIGHTERS: White and Black, styles shaped by {opening}.\n"
            f"BEATS: {beats} escalating exchange{'s' if beats != 1 else ''} building to "
            "the final blow.\n"
            f"{finish_line}\n"
            "STYLE: Sharp linework, fast cuts on impact frames."
        )

        return (
            f"FIGHT_STYLE: {fight_style}\n"
            f"BATTLE_SUMMARY: {battle_summary}\n"
            f"SCENE_PROMPT: {scene_prompt}"
        )


class OpenAIProvider(AIProvider):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise AIProviderError("OPENAI_API_KEY is not configured.")
        from openai import AsyncOpenAI  # local import: optional dependency at runtime

        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._timeout = settings.ai_request_timeout

    async def generate_text(self, prompt: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self._timeout,
            )
            content = response.choices[0].message.content
            if not content:
                raise AIProviderError("OpenAI returned an empty response.")
            return content
        except AIProviderError:
            raise
        except Exception as exc:  # SDK-specific errors -> one common type
            logger.warning("OpenAI generation failed: %s", exc)
            raise AIProviderError(str(exc)) from exc


class AnthropicProvider(AIProvider):
    def __init__(self, settings: Settings):
        if not settings.anthropic_api_key:
            raise AIProviderError("ANTHROPIC_API_KEY is not configured.")
        from anthropic import AsyncAnthropic  # local import: optional dependency at runtime

        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model
        self._timeout = settings.ai_request_timeout

    async def generate_text(self, prompt: str) -> str:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
                timeout=self._timeout,
            )
            blocks = [b.text for b in response.content if b.type == "text"]
            if not blocks:
                raise AIProviderError("Anthropic returned no text content.")
            return "\n".join(blocks)
        except AIProviderError:
            raise
        except Exception as exc:
            logger.warning("Anthropic generation failed: %s", exc)
            raise AIProviderError(str(exc)) from exc


class GeminiProvider(AIProvider):
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise AIProviderError("GEMINI_API_KEY is not configured.")
        from google import genai  # local import: optional dependency at runtime

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    async def generate_text(self, prompt: str) -> str:
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model, contents=prompt
            )
            if not response.text:
                raise AIProviderError("Gemini returned an empty response.")
            return response.text
        except AIProviderError:
            raise
        except Exception as exc:
            logger.warning("Gemini generation failed: %s", exc)
            raise AIProviderError(str(exc)) from exc


class LocalProvider(AIProvider):
    """Talks to a self-hosted model server using Ollama's /api/generate
    contract (a common baseline for local models). Point local_model_url
    at a different local server and adjust the payload/parsing below if
    yours speaks a different protocol."""

    def __init__(self, settings: Settings):
        self._url = settings.local_model_url
        self._model = settings.local_model
        self._timeout = settings.ai_request_timeout

    async def generate_text(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._url,
                    json={"model": self._model, "prompt": prompt, "stream": False},
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("response")
                if not text:
                    raise AIProviderError("Local model server returned no text.")
                return text
        except AIProviderError:
            raise
        except Exception as exc:
            logger.warning("Local model generation failed: %s", exc)
            raise AIProviderError(str(exc)) from exc


_PROVIDERS: dict[str, type[AIProvider]] = {
    "template": TemplateProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "local": LocalProvider,
}


@lru_cache
def get_ai_provider() -> AIProvider:
    """FastAPI dependency — one cached provider instance per process,
    selected by AI_PROVIDER. Falls back to TemplateProvider (logging a
    warning) if the configured provider fails to initialize, e.g. a
    missing API key, so a config mistake degrades gracefully instead of
    crashing the whole app at startup."""
    settings = get_settings()
    provider_name = settings.ai_provider.lower()
    provider_cls = _PROVIDERS.get(provider_name)

    if provider_cls is None:
        logger.warning(
            "Unknown AI_PROVIDER '%s' — falling back to TemplateProvider.", provider_name
        )
        return TemplateProvider()

    if provider_cls is TemplateProvider:
        return TemplateProvider()

    try:
        return provider_cls(settings)  # type: ignore[call-arg]
    except AIProviderError as exc:
        logger.warning(
            "Could not initialize %s (%s) — falling back to TemplateProvider.",
            provider_cls.__name__,
            exc,
        )
        return TemplateProvider()
