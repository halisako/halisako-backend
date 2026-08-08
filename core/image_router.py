"""Image provider abstraction — mirrors core/ai_router.py's
architecture exactly, one layer over for images instead of text.

Whatever orchestration code eventually turns a shot's `image_prompt`
into an actual image (a future "Director"-level component — none
exists yet; nothing in this backend currently calls into image
generation at all, including this task) must go through `ImageRouter`,
never a concrete `ImageProvider` implementation directly. That's not
just a docstring aspiration: `MockImageProvider` (and any future
provider) has zero knowledge of `ImageRouter`, the registry, or any
other provider — it only implements the `ImageProvider` interface —
and the only way to reach a working provider from outside this module
is through `get_image_router()` / `ImageRouter.generate_image()`. See
this module's test suite for a test that exercises exactly that path
without ever importing `MockImageProvider` by name.

Four pieces, matching the task's four named deliverables:

- `ImageProvider` — the interface every provider implements (parallel
  to `AIProvider`).
- `ImageGenerationResult` — the provider-agnostic result shape.
- `ImageProviderRegistry` — tracks which provider implementations are
  available, keyed by name; adding a new provider is one `register()`
  call, never a change to `ImageRouter`'s own dispatch logic.
- `ImageRouter` — the single entry point; resolves the configured
  provider through the registry and delegates to it.

`MockImageProvider` is the only concrete provider implemented here. It
writes real, valid PNG files to disk (via Pillow — a new dependency
this task introduces, noted explicitly since nothing else in this
backend needed an imaging library before) rather than faking a path
string, so a future rendering pipeline can be developed and tested end
to end — including anything that opens and inspects the resulting
file — without any external API, network access, or cost.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from core.config import get_settings
settings = get_settings()
from core.exceptions import ImageProviderError

logger = logging.getLogger(__name__)


class ImageGenerationResult(BaseModel):
    """The output of one image generation call — identical shape
    regardless of which ImageProvider produced it."""

    image_path: str = Field(..., min_length=1, description="Filesystem path to the generated image.")
    provider: str = Field(..., min_length=1, description="Name of the provider that produced this image.")
    prompt: str = Field(..., min_length=1, description="The prompt used to generate this image.")
    width: int = Field(..., gt=0, description="Image width, in pixels.")
    height: int = Field(..., gt=0, description="Image height, in pixels.")
    generation_time_seconds: float = Field(
        ..., ge=0, description="Wall-clock time the generation call took. A real measurement, "
        "not deterministic content — unlike the image itself, this is expected to vary run to run."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra detail, if any."
    )


class ImageProvider(ABC):
    """A text-to-image generation backend. Every implementation is
    self-contained — it must not import or depend on `ImageRouter`,
    `ImageProviderRegistry`, or any other `ImageProvider`
    implementation, so providers stay independently swappable."""

    @abstractmethod
    async def generate_image(self, prompt: str, width: int = 1024, height: int = 1024) -> ImageGenerationResult:
        """Generates an image for the given prompt.

        Args:
            prompt: The text-to-image prompt.
            width: Requested image width, in pixels.
            height: Requested image height, in pixels.
        """


def _deterministic_hash(prompt: str) -> str:
    """A short, stable hex digest of a prompt — used for both the
    placeholder filename and its color, so the same prompt always
    produces the same file, without any randomness."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _deterministic_color(prompt_hash: str) -> tuple[int, int, int]:
    """Derives an RGB color from a prompt's hash — deterministic, and
    spread reasonably across the color space so different prompts are
    visually distinguishable at a glance."""
    return (
        int(prompt_hash[0:2], 16),
        int(prompt_hash[2:4], 16),
        int(prompt_hash[4:6], 16),
    )


class MockImageProvider(ImageProvider):
    """Generates real, valid placeholder PNG files without calling any
    external API — for developing and testing a future rendering
    pipeline without cost, network access, or nondeterministic output.

    The generated file's path and visual content (background color,
    overlaid text) are fully deterministic — the same prompt always
    produces the same image, byte for byte, which is what makes this
    provider actually useful for tests (a test can assert on the exact
    resulting path or re-generate and diff). Only
    `generation_time_seconds` varies between calls, because it's a
    genuine wall-clock measurement, not generated content.
    """

    def __init__(self, output_dir: str | None = None) -> None:
        """Initializes the provider.

        Args:
            output_dir: Where to write generated placeholder images.
                Defaults to `settings.image_output_dir`.
        """
        self._output_dir = Path(output_dir if output_dir is not None else settings.image_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_image(self, prompt: str, width: int = 1024, height: int = 1024) -> ImageGenerationResult:
        start = time.monotonic()

        prompt_hash = _deterministic_hash(prompt)
        color = _deterministic_color(prompt_hash)

        image = Image.new("RGB", (width, height), color=color)
        draw = ImageDraw.Draw(image)
        label = f"MOCK IMAGE\n{prompt[:80]}"
        draw.multiline_text((20, 20), label, fill=(255, 255, 255))

        image_path = self._output_dir / f"mock_{prompt_hash}.png"
        image.save(image_path, format="PNG")

        elapsed = time.monotonic() - start
        return ImageGenerationResult(
            image_path=str(image_path),
            provider="MockImageProvider",
            prompt=prompt,
            width=width,
            height=height,
            generation_time_seconds=elapsed,
            metadata={"placeholder": True, "prompt_hash": prompt_hash},
        )


class ImageProviderRegistry:
    """Tracks which ImageProvider implementations are available, keyed
    by name.

    Providers are registered as factories (zero-argument callables
    returning a fresh `ImageProvider`), not pre-built instances, so
    registering a provider never has to construct one until it's
    actually needed.
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], ImageProvider]] = {}

    def register(self, name: str, factory: Callable[[], ImageProvider]) -> None:
        """Registers (or replaces) a provider factory under `name`."""
        self._factories[name] = factory

    def is_registered(self, name: str) -> bool:
        """Returns True if `name` has a registered provider factory."""
        return name in self._factories

    def create(self, name: str) -> ImageProvider:
        """Constructs a fresh provider instance for `name`.

        Raises:
            ImageProviderError: If `name` is not registered.
        """
        factory = self._factories.get(name)
        if factory is None:
            raise ImageProviderError(
                f"Image provider {name!r} is not registered. Known providers: {sorted(self._factories)}."
            )
        return factory()

    def list_providers(self) -> list[str]:
        """Returns every currently registered provider name."""
        return list(self._factories)


def _default_registry() -> ImageProviderRegistry:
    """Builds the registry pre-populated with every provider this
    backend ships. Adding a real provider later (a DALL-E or Stable
    Diffusion implementation, say) means one `register()` call here —
    never a change to `ImageRouter` itself."""
    registry = ImageProviderRegistry()
    registry.register("mock", MockImageProvider)
    return registry


class ImageRouter:
    """The single entry point for image generation. Resolves the
    configured provider (`settings.image_provider`, falling back to
    "mock" if unrecognized) through an `ImageProviderRegistry` and
    delegates to it — callers never need to know which concrete
    provider is active, and switching providers is a configuration
    change, never a code change at the call site.
    """

    def __init__(self, registry: ImageProviderRegistry | None = None) -> None:
        """Initializes the router.

        Args:
            registry: The provider registry to resolve against.
                Defaults to a fresh registry pre-populated with every
                provider this backend ships.
        """
        self._registry = registry or _default_registry()
        self._provider: ImageProvider | None = None
        self._active_provider_name: str = ""

    async def generate_image(self, prompt: str, width: int = 1024, height: int = 1024) -> ImageGenerationResult:
        """Generates an image for the given prompt, via whichever
        provider is currently configured.

        Args:
            prompt: The text-to-image prompt.
            width: Requested image width, in pixels.
            height: Requested image height, in pixels.
        """
        provider = self._resolve_provider()
        return await provider.generate_image(prompt, width=width, height=height)

    def active_provider_name(self) -> str:
        """Returns the name of whichever provider is actually active
        (may differ from `settings.image_provider` if that value
        wasn't a registered provider and this router fell back)."""
        self._resolve_provider()
        return self._active_provider_name

    def _resolve_provider(self) -> ImageProvider:
        if self._provider is not None:
            return self._provider

        requested = settings.image_provider
        if not self._registry.is_registered(requested):
            logger.warning(
                "Image provider %r is not registered — falling back to 'mock'. Known providers: %s.",
                requested, self._registry.list_providers(),
            )
            requested = "mock"

        self._provider = self._registry.create(requested)
        self._active_provider_name = requested
        return self._provider


_router_instance: ImageRouter | None = None


def get_image_router() -> ImageRouter:
    """FastAPI-dependency-friendly accessor: returns a cached singleton
    ImageRouter, matching `core.ai_router.get_ai_provider`'s role for
    AIProvider."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ImageRouter()
    return _router_instance
