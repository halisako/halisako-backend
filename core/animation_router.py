"""Animation provider abstraction — mirrors core/image_router.py's
architecture exactly, one layer over for image-to-video animation
instead of still-image generation.

    Cinematic Shot -> AnimationInstruction -> AnimationProvider -> AnimationResult -> Video/rendering pipeline

Whatever future component turns a shot into an animated clip must
never know which animation model produced the result — it only
consumes `AnimationResult`. The Animation Provider, symmetrically,
must never know anything about FastAPI, HTTP, chess, PGN, Battle
Director, or cinematic timeline generation — it only receives a
structured `AnimationInstruction` and returns an `AnimationResult`.
Every implementation here has zero knowledge of `AnimationRouter`, the
registry, or any other provider — the only way to reach a working
provider from outside this module is through `get_animation_router()`
/ `AnimationRouter.generate_animation()`.

Not yet wired into `orchestrator.py`, `/generate`, or `/render` —
deliberately. This is Sprint 4 Prompt 1: the contract and a mock
implementation only. See this module's docstring end for the exact
integration point a later prompt would use.

Six pieces, matching this task's six named deliverables:

- `AnimationType` — the supported generation modality
  (image_to_video / text_to_video / image_sequence_to_video). Not the
  *content* of the animation (attack, dodge, idle, ...) — that's
  `AnimationInstruction.subject_motion`, a free-form string. Kept to
  three values, per this task's own "do not over-engineer" guidance;
  only `IMAGE_TO_VIDEO` is actually implemented by `MockAnimationProvider`
  today, since that's all the current pipeline needs.
- `AnimationInstruction` — a provider-neutral description of an
  animation request. Carries everything a provider needs (source
  image, prompt, motion, duration, output dimensions, seed, ...)
  without describing *how* any particular model should produce it —
  no ComfyUI node IDs, no vendor-specific sampler settings.
- `AnimationResult` — the provider-agnostic result shape. Unlike
  `ImageGenerationResult` (which only ever represents success — a
  failed call raises `ImageProviderError`), `AnimationResult` models
  failure as data via `success` / `error_message`. Deliberate, not an
  oversight: animation generation is typically longer-running and more
  failure-prone than still-image generation, and a caller processing
  many shots may want to continue past one shot's failure rather than
  have the whole batch raise. `AnimationProviderError` is reserved for
  infrastructure-level problems (an unregistered provider name) —
  never for an individual generation attempt failing.
- `AnimationProvider` — the interface every provider implements
  (parallel to `ImageProvider`). Takes only an `AnimationInstruction`
  — not a separate `image_path` parameter — since the instruction
  already carries `source_image_path`; splitting the image reference
  across two parameters that would always need to agree seemed like
  the wrong kind of duplication for a "provider-neutral request
  description" whose whole point is to carry everything about the
  request in one place.
- `AnimationProviderRegistry` — tracks which provider implementations
  are available, keyed by name; adding a new provider is one
  `register()` call, never a change to `AnimationRouter`'s own
  dispatch logic.
- `AnimationRouter` — the single entry point; resolves the configured
  provider through the registry and delegates to it.

One concrete provider is implemented here:

- `MockAnimationProvider` — produces a real, valid MP4 without calling
  any external AI service, by holding the source image static for the
  instruction's duration. Reuses
  `products.chess2fight.rendering.video_builder.VideoBuilder` for the
  actual ffmpeg invocation rather than duplicating that logic. Does
  not differentiate behavior by `animation_type` — it always animates
  from `source_image_path` the same way regardless of the requested
  modality, which is a deliberate simplification for a mock whose job
  is "produce something real and testable," not "faithfully emulate
  each modality's distinct behavior."

On `camera_motion` / `subject_motion` being plain strings rather than
enums: `products.chess2fight.cinematic.schemas.CameraMotion` already
exists with an overlapping vocabulary. Reusing it here would mean
`core/` (cross-product infrastructure) depending on a
Chess2Fight-specific schema — backwards layering. A closed vocabulary
belongs at the product layer if one product wants it, not baked into
a contract meant to outlive any one product.

Integration point for a later Sprint 4 prompt: `FightVideoPipeline`
(products/chess2fight/rendering/pipeline.py) currently goes straight
from `RenderPipeline` (one static frame per shot) to `VideoBuilder`
(holding each frame for a fixed duration). A later prompt would insert
an animation step between those two — turning each rendered frame into
an `AnimationInstruction`, calling `AnimationRouter.generate_animation()`
per shot, and having `VideoBuilder` (or a successor) concatenate the
resulting clips instead of holding static frames. Nothing in this
module assumes that shape, but nothing here has been built to enable
anything else either — this prompt intentionally stops before that
wiring exists.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from core.config import get_settings
from core.exceptions import AnimationProviderError

settings = get_settings()
logger = logging.getLogger(__name__)


class AnimationType(str, Enum):
    """The supported animation generation modality — not the content
    of the animation (see `AnimationInstruction.subject_motion` for
    that). Kept small: only `IMAGE_TO_VIDEO` is implemented by
    `MockAnimationProvider` today, since that's all the current
    pipeline needs; the other two exist so the contract doesn't have
    to be redesigned when a provider supporting them arrives.
    """

    IMAGE_TO_VIDEO = "image_to_video"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_SEQUENCE_TO_VIDEO = "image_sequence_to_video"


class AnimationInstruction(BaseModel):
    """A provider-neutral description of one animation request — what
    should be produced, not how any particular model should produce
    it. No provider-specific parameters (ComfyUI node IDs, a specific
    vendor's sampler settings, a specific API's request fields) belong
    here; those live inside a concrete provider, never in this
    contract.
    """

    shot_id: str = Field(..., min_length=1, description="ID of the Shot this instruction animates.")
    source_image_path: str = Field(..., min_length=1, description="Path to the primary reference image to animate.")
    reference_image_paths: list[str] = Field(
        default_factory=list, description="Optional secondary/reference images, for providers that use them."
    )
    prompt: str = Field(..., min_length=1, description="Text description of the desired motion/scene.")
    negative_prompt: str | None = Field(
        default=None, description="What to avoid in the generated animation, for providers that support it."
    )
    duration_seconds: float = Field(..., gt=0, description="How long the animated clip should be, in seconds.")
    camera_motion: str = Field(
        ..., min_length=1, description="Free-form description of the camera's movement, e.g. 'tracking'."
    )
    subject_motion: str = Field(
        ..., min_length=1, description="Free-form description of the subject's movement, e.g. 'forward_strike'."
    )
    motion_intensity: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How much movement/energy the animation should carry, 0.0-1.0. Defaults to a "
        "neutral midpoint: the Chess2Fight Shot model this contract is built from (Sprint 4 Prompt 2) "
        "has no numeric intensity field to derive this from, so it's left at default rather than "
        "fabricated — see products/chess2fight/rendering/animation_pipeline.py's mapping notes.",
    )
    width: int = Field(default=1024, gt=0, description="Output width, in pixels.")
    height: int = Field(default=1024, gt=0, description="Output height, in pixels.")
    fps: int | None = Field(default=None, gt=0, description="Output frame rate, for providers that need one upfront.")
    seed: int | None = Field(default=None, description="Reproducibility seed, for providers that support one.")
    animation_type: AnimationType = Field(
        default=AnimationType.IMAGE_TO_VIDEO, description="Which generation modality this instruction calls for."
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Free-form extra context.")


class AnimationResult(BaseModel):
    """The output of one animation provider call — identical shape
    regardless of which AnimationProvider produced it, and regardless
    of whether generation succeeded. A downstream renderer consumes
    only this — never anything provider-specific (not ComfyUI,
    Replicate, RunPod, Hugging Face, or any other provider's own
    response shape).
    """

    success: bool = Field(..., description="Whether animation generation succeeded.")
    shot_id: str = Field(..., min_length=1, description="ID of the Shot this result corresponds to.")
    provider: str = Field(..., min_length=1, description="Name of the provider that produced this result.")
    video_path: str | None = Field(
        default=None, description="Path to the generated animated clip. None if generation failed."
    )
    duration_seconds: float | None = Field(default=None, description="Duration of the generated clip, in seconds.")
    width: int | None = Field(default=None, description="Width of the generated clip, in pixels, if available.")
    height: int | None = Field(default=None, description="Height of the generated clip, in pixels, if available.")
    fps: int | None = Field(default=None, description="Frame rate of the generated clip, if available.")
    error_message: str | None = Field(
        default=None, description="Why generation failed, if success is False. None on success."
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Provider-specific extra detail, if any.")


class AnimationProvider(ABC):
    """An image-to-video animation backend. Every implementation is
    self-contained — it must not import or depend on `AnimationRouter`,
    `AnimationProviderRegistry`, or any other `AnimationProvider`
    implementation, so providers stay independently swappable. Must
    know only an `AnimationInstruction` — nothing about FastAPI, HTTP,
    chess, PGN, Battle Director, or cinematic timeline generation.
    """

    @abstractmethod
    async def generate_animation(self, instruction: AnimationInstruction) -> AnimationResult:
        """Generates an animated clip.

        Args:
            instruction: What should be produced.
        """


class MockAnimationProvider(AnimationProvider):
    """Produces a real, valid MP4 without calling any external AI
    service — for developing and testing the animation pipeline
    without cost, network access, or an actual image-to-video model.

    Holds `instruction.source_image_path` static for
    `instruction.duration_seconds`, via
    `products.chess2fight.rendering.video_builder.VideoBuilder` — the
    same ffmpeg-invocation logic the rendering pipeline already uses,
    reused rather than duplicated. Behavior does not vary by
    `animation_type` — see this module's docstring for why.
    """

    def __init__(self, output_dir: str | None = None, video_builder: Any | None = None) -> None:
        """Initializes the provider.

        Args:
            output_dir: Where to save generated clips. Defaults to
                `settings.animation_output_dir`.
            video_builder: The VideoBuilder to delegate ffmpeg
                invocation to. Defaults to a fresh
                `products.chess2fight.rendering.video_builder.VideoBuilder`.
                Typed `Any` here (rather than imported at module level)
                specifically so this cross-product `core/` module
                never has a hard import-time dependency on
                `products.chess2fight` — the import happens lazily,
                only if no video_builder is injected.
        """
        self._output_dir = Path(output_dir if output_dir is not None else settings.animation_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._video_builder = video_builder or self._default_video_builder()

    @staticmethod
    def _default_video_builder() -> Any:
        from products.chess2fight.rendering.video_builder import VideoBuilder

        return VideoBuilder()

    async def generate_animation(self, instruction: AnimationInstruction) -> AnimationResult:
        if not Path(instruction.source_image_path).exists():
            return AnimationResult(
                success=False, shot_id=instruction.shot_id, provider="MockAnimationProvider",
                error_message=f"Reference image not found: {instruction.source_image_path!r}.",
            )

        try:
            with tempfile.TemporaryDirectory() as frame_dir:
                shutil.copyfile(instruction.source_image_path, Path(frame_dir) / "frame0001.png")
                output_path = self._output_dir / f"animation_{instruction.shot_id}.mp4"
                build_kwargs: dict[str, Any] = {
                    "frame_directory": frame_dir,
                    "output_path": str(output_path),
                    "frame_count": 1,
                    "frame_duration_seconds": instruction.duration_seconds,
                    "width": instruction.width,
                    "height": instruction.height,
                }
                if instruction.fps is not None:
                    build_kwargs["fps"] = instruction.fps
                build_result = await self._video_builder.build_video(**build_kwargs)
        except Exception as exc:
            return AnimationResult(
                success=False, shot_id=instruction.shot_id, provider="MockAnimationProvider",
                error_message=str(exc),
            )

        return AnimationResult(
            success=True,
            shot_id=instruction.shot_id,
            provider="MockAnimationProvider",
            video_path=build_result.video_path,
            duration_seconds=build_result.duration_seconds,
            width=build_result.width,
            height=build_result.height,
            fps=build_result.fps,
            metadata={
                "placeholder": True,
                "animation_type": instruction.animation_type.value,
                "motion_intensity": instruction.motion_intensity,
            },
        )


class AnimationProviderRegistry:
    """Tracks which AnimationProvider implementations are available,
    keyed by name. Providers are registered as factories (zero-argument
    callables returning a fresh `AnimationProvider`), matching
    `ImageProviderRegistry`'s own pattern exactly.
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], AnimationProvider]] = {}

    def register(self, name: str, factory: Callable[[], AnimationProvider]) -> None:
        """Registers (or replaces) a provider factory under `name`."""
        self._factories[name] = factory

    def is_registered(self, name: str) -> bool:
        """Returns True if `name` has a registered provider factory."""
        return name in self._factories

    def create(self, name: str) -> AnimationProvider:
        """Constructs a fresh provider instance for `name`.

        Raises:
            AnimationProviderError: If `name` is not registered.
        """
        factory = self._factories.get(name)
        if factory is None:
            raise AnimationProviderError(
                f"Animation provider {name!r} is not registered. Known providers: {sorted(self._factories)}."
            )
        return factory()

    def list_providers(self) -> list[str]:
        """Returns every currently registered provider name."""
        return list(self._factories)


def _default_registry() -> AnimationProviderRegistry:
    """Builds the registry pre-populated with every provider this
    backend ships.

    This function — not `AnimationRouter` itself — is where knowledge
    of concrete provider classes is allowed to live: it's a
    composition root ("which providers exist"), not routing logic
    ("which provider handles this request"), and `AnimationRouter`'s
    own code never names a concrete provider. `ComfyUIAnimationProvider`
    is imported locally (not at module level) specifically so
    `core/animation_router.py` — imported by every provider, including
    `MockAnimationProvider` in this same file — never has a hard
    import-time dependency on `httpx`-based ComfyUI networking code
    that most callers (anything just using the mock) don't need.
    """
    from core.animation_providers.comfyui import ComfyUIAnimationProvider

    registry = AnimationProviderRegistry()
    registry.register("mock", MockAnimationProvider)
    registry.register("comfyui", ComfyUIAnimationProvider)
    return registry


class AnimationRouter:
    """The single entry point for animation generation. Resolves the
    configured provider (`settings.animation_provider`, falling back
    to "mock" if unrecognized) through an `AnimationProviderRegistry`
    and delegates to it — callers never need to know which concrete
    provider is active.
    """

    def __init__(self, registry: AnimationProviderRegistry | None = None) -> None:
        """Initializes the router.

        Args:
            registry: The provider registry to resolve against.
                Defaults to a fresh registry pre-populated with every
                provider this backend ships.
        """
        self._registry = registry or _default_registry()
        self._provider: AnimationProvider | None = None
        self._active_provider_name: str = ""

    async def generate_animation(self, instruction: AnimationInstruction) -> AnimationResult:
        """Generates an animated clip via whichever provider is
        currently configured.

        Never raises for a generation-attempt failure — even if the
        resolved provider misbehaves and raises unexpectedly, this
        returns a failed `AnimationResult` rather than propagating.
        `AnimationProviderError` can still surface from provider
        *resolution* (an unregistered configured name reaching
        `registry.create`), which is an infrastructure problem, not a
        generation-attempt one.

        Args:
            instruction: What should be produced.
        """
        provider = self._resolve_provider()
        try:
            return await provider.generate_animation(instruction)
        except Exception as exc:
            logger.exception("Animation provider %r raised unexpectedly.", self._active_provider_name)
            return AnimationResult(
                success=False, shot_id=instruction.shot_id, provider=self._active_provider_name,
                error_message=str(exc),
            )

    def active_provider_name(self) -> str:
        """Returns the name of whichever provider is actually active
        (may differ from `settings.animation_provider` if that value
        wasn't a registered provider and this router fell back)."""
        self._resolve_provider()
        return self._active_provider_name

    def _resolve_provider(self) -> AnimationProvider:
        if self._provider is not None:
            return self._provider

        requested = settings.animation_provider
        if not self._registry.is_registered(requested):
            logger.warning(
                "Animation provider %r is not registered — falling back to 'mock'. Known providers: %s.",
                requested, self._registry.list_providers(),
            )
            requested = "mock"

        self._provider = self._registry.create(requested)
        self._active_provider_name = requested
        return self._provider


_router_instance: AnimationRouter | None = None


def get_animation_router() -> AnimationRouter:
    """FastAPI-dependency-friendly accessor: returns a cached singleton
    AnimationRouter, matching `core.image_router.get_image_router`'s
    role for `ImageRouter`."""
    global _router_instance
    if _router_instance is None:
        _router_instance = AnimationRouter()
    return _router_instance
