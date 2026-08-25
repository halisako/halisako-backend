"""RenderPipeline: PromptedTimeline -> rendered frames on disk.

    Prompt Generator -> Render Pipeline

For every shot in a PromptedTimeline: reads its `image_prompt`, sends
it to `ImageRouter` (core/image_router.py — a cross-product, generic
service; RenderPipeline never imports or references a concrete
ImageProvider), receives back a generated image, assigns it a frame
number, and saves it via `AssetManager` under
`storage/renders/{fight_id}/frameNNNN.png`. Generates exactly one
frame per shot — no video assembly, no interpolation between shots;
"only frames" is a hard boundary this module doesn't cross.

Frame numbers are simply each shot's own `sequence_order` — the
Timeline Engine already established a correct, gapless 1..N ordering
(products/chess2fight/cinematic/timeline_engine.py), so there's no
reason for this module to invent a second numbering scheme.

Shots are rendered sequentially, not concurrently: the brief describes
"iterate through every Shot," and sequential processing sidesteps any
risk of two concurrent saves racing for the same frame number or
image-provider file name — there is no benefit here worth that risk,
since nothing about a fight's frame count makes concurrency
necessary.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from core.image_router import ImageRouter, get_image_router
from products.chess2fight.cinematic.schemas import PromptedShot, PromptedTimeline
from products.chess2fight.rendering.asset_manager import AssetManager, FrameMetadata, RenderManifest


def _derive_seed(prompt: str) -> int:
    """Derives a deterministic "generation seed" from a prompt — used
    as FrameMetadata.generation_seed's fallback when the actual
    provider doesn't report one of its own (see that field's own
    docstring, and this method's one remaining call site in
    `_render_one_shot`, for the current, non-fallback case)."""
    return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)


class RenderedFrame(BaseModel):
    """One rendered frame: where it was saved, and its full metadata."""

    frame_number: int = Field(..., ge=1, description="1-indexed frame number — the shot's sequence_order.")
    frame_path: str = Field(..., min_length=1, description="Path the frame was saved to in fight storage.")
    metadata: FrameMetadata = Field(..., description="This frame's complete metadata record.")


class RenderOutput(BaseModel):
    """The RenderPipeline's complete result for one fight."""

    fight_id: str = Field(..., min_length=1, description="Which fight this render belongs to.")
    frames: list[RenderedFrame] = Field(..., min_length=1, description="Every rendered frame, in frame_number order.")
    frame_count: int = Field(..., ge=1, description="Number of frames rendered — len(frames).")
    output_dir: str = Field(..., min_length=1, description="Directory all frames and metadata.json were saved to.")
    manifest_path: str = Field(..., min_length=1, description="Path to the written metadata.json.")


class RenderPipeline:
    """Turns a PromptedTimeline into rendered frames on disk.

    `image_router` and `asset_manager` are injected (defaulting to the
    shared singleton / a fresh manager, respectively) so a test can
    substitute a router pointed at a temp directory without any
    global state.
    """

    def __init__(self, image_router: ImageRouter | None = None, asset_manager: AssetManager | None = None) -> None:
        """Initializes the pipeline.

        Args:
            image_router: Where to send prompts for image generation.
                Defaults to the shared `get_image_router()` singleton.
            asset_manager: Where to save frames and metadata. Defaults
                to a fresh AssetManager using `settings.render_storage_root`.
        """
        self._image_router = image_router or get_image_router()
        self._asset_manager = asset_manager or AssetManager()

    async def render(
        self, timeline: PromptedTimeline, fight_id: str, width: int | None = None, height: int | None = None,
    ) -> RenderOutput:
        """Renders every shot in `timeline` to a frame on disk.

        Args:
            timeline: The Prompt Generator's output — one prompt per shot.
            fight_id: Identifies which fight this render belongs to;
                determines the storage subdirectory
                (`storage/renders/{fight_id}/`). Not derived
                internally — see this module's accompanying
                engineering notes for why: nothing upstream in this
                pipeline currently carries a persistent fight/battle
                identifier for this module to reuse.
            width: Output image width, in pixels. Sprint 4 Prompt 10:
                left `None` by default — this generic, provider-
                agnostic pipeline has no opinion on resolution and
                doesn't hard-code any provider-specific policy (e.g.
                FLUX's own validated 1280x704); when `None`,
                `ImageRouter.generate_image()`'s own generic default
                (1024x1024) applies, exactly as before this parameter
                existed. A Chess2Fight-specific caller that wants a
                particular provider's validated resolution (FLUX,
                say) should resolve that policy itself and pass it
                here explicitly — see FightVideoPipeline.run() and
                SingleShotAcceptanceRunner.execute(), which now both
                do exactly that from the same settings values.
            height: Output image height, in pixels. Same `None`-means-
                generic-default reasoning as `width`.

        Returns:
            A RenderOutput describing every frame written and where
            the fight's metadata.json ended up.
        """
        rendered_frames = [
            await self._render_one_shot(shot, fight_id, width, height) for shot in timeline.shots
        ]

        manifest = RenderManifest(
            fight_id=fight_id,
            frame_count=len(rendered_frames),
            frames=[frame.metadata for frame in rendered_frames],
        )
        manifest_path = self._asset_manager.write_manifest(fight_id, manifest)

        return RenderOutput(
            fight_id=fight_id,
            frames=rendered_frames,
            frame_count=len(rendered_frames),
            output_dir=str(self._asset_manager.fight_directory(fight_id)),
            manifest_path=str(manifest_path),
        )

    async def _render_one_shot(
        self, shot: PromptedShot, fight_id: str, width: int | None = None, height: int | None = None,
    ) -> RenderedFrame:
        """Renders a single shot: generate via ImageRouter, save via
        AssetManager, build its metadata record."""
        frame_number = shot.sequence_order

        image_kwargs = {}
        if width is not None:
            image_kwargs["width"] = width
        if height is not None:
            image_kwargs["height"] = height
        result = await self._image_router.generate_image(shot.image_prompt, **image_kwargs)
        saved_path = self._asset_manager.save_frame(fight_id, frame_number, result.image_path)

        # Sprint 4 Prompt 12.1: use the actual seed the provider
        # reports it used (ImageGenerationResult.metadata["seed"] —
        # already a generic field on the generic result type, so this
        # stays provider-agnostic; no concrete-provider import here)
        # when one is available. Falls back to the deterministic
        # prompt-derived value for a provider that doesn't report a
        # seed at all (the default configured provider's own metadata
        # carries no "seed" key, so this fallback is exactly what runs
        # there — unchanged behavior). Correct even under a non-default
        # seed policy: a configured provider's own seed_override can
        # now correctly be recorded as the seed actually used, not a
        # value re-derived from the prompt that could silently disagree
        # with it.
        generation_seed = result.metadata.get("seed")
        if generation_seed is None:
            generation_seed = _derive_seed(shot.image_prompt)

        metadata = FrameMetadata(
            frame_number=frame_number,
            prompt=shot.image_prompt,
            camera_angle=shot.camera_angle.value,
            camera_motion=shot.camera_motion.value,
            shot_id=shot.shot_id,
            shot_type=shot.shot_type.value,
            source_moves=list(shot.source_moves),
            timestamp=datetime.now(timezone.utc).isoformat(),
            generation_seed=generation_seed,
        )
        return RenderedFrame(frame_number=frame_number, frame_path=str(saved_path), metadata=metadata)
