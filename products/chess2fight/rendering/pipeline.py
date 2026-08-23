"""FightVideoPipeline: the complete Sprint 3 + Sprint 4 pipeline, end to end.

    PGN -> Battle Director -> Battle Intelligence -> Fight Story ->
    Timeline Engine -> Scene Composer -> Prompt Generator ->
    ImageRouter -> Render Pipeline -> Animation Pipeline ->
    AnimationRouter -> Video Builder (concatenation) -> fight.mp4

This file isn't one any brief explicitly named — it only named
`video_builder.py` (Sprint 3) — but "integrate the complete pipeline"
needs this orchestration to live somewhere, and putting it here
(rather than inline in the API route) keeps `api/chess2fight.py` thin,
matching that module's own stated principle ("no chess logic or AI
logic lives here"), and keeps `orchestrator.py` completely untouched —
this module calls `FightOrchestrator.generate_fight()` as-is, adding
nothing to it, so every existing Sprint 2 test is unaffected by this
file's existence.

`FightVideoPipeline.run()` is the single entry point: it reuses
`FightOrchestrator.generate_fight()` to get everything through
`prompted_timeline`, renders every shot to a static reference frame via
`RenderPipeline`, animates each frame into a per-shot clip via
`AnimationPipeline` (Sprint 4 Prompt 2 — new), and concatenates those
clips into the final `fight.mp4` via `VideoBuilder.concatenate_clips`.

Sprint 4 Prompt 2 change, specifically: this used to call
`VideoBuilder.build_video()` directly on `RenderPipeline`'s static
frame directory (holding each frame still for a uniform duration).
It now calls `AnimationPipeline.animate()` first and
`VideoBuilder.concatenate_clips()` on the resulting per-shot clips
instead — the final video's per-shot timing now comes from each
`Shot.duration_seconds` (via `AnimationInstruction`), not a single
uniform `frame_duration_seconds` applied to everything. The
`frame_duration_seconds` parameter (and `RenderVideoRequest` field) is
therefore no longer read by this method — kept on the request schema
untouched for backward compatibility (an existing caller passing it
doesn't get a validation error), but it no longer does anything; see
this module's test suite for a regression test confirming the field
is still accepted.
"""

from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel, Field

from core.ai_router import AIProvider
from core.config import get_settings
from products.chess2fight.cinematic.schemas import Shot
from products.chess2fight.orchestrator import FightOrchestrator
from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.render_pipeline import RenderedFrame, RenderPipeline
from products.chess2fight.rendering.video_builder import VideoBuilder
from products.chess2fight.schemas import BattlePreferences

logger = logging.getLogger(__name__)


class RenderVideoRequest(BaseModel):
    """Request shape for the full pipeline route. Mirrors
    GenerateRequest's `pgn`/`style`/`preferences` fields exactly (so
    the same preference-resolution logic api/chess2fight.py already
    has for `/generate` applies unchanged), plus video-specific
    configuration the existing request has no reason to carry."""

    pgn: str = Field(..., min_length=1, description="PGN text of the game to render.")
    style: str = Field(default="anime", description="Visual/narrative style, same as GenerateRequest.")
    preferences: BattlePreferences | None = Field(
        default=None, description="Optional structured preferences, same as GenerateRequest."
    )
    fps: int = Field(
        default_factory=lambda: get_settings().comfyui_default_fps,
        gt=0,
        description="Output video frame rate. Defaults to the current Wan-validated policy "
        "(settings.comfyui_default_fps, currently 8) — not a hardcoded value that could drift "
        "out of sync with it, as a prior literal default of 24 did (Sprint 4 Prompt 10.1).",
    )
    width: int = Field(
        default_factory=lambda: get_settings().comfyui_animation_default_width,
        gt=0,
        description="Output video width, in pixels. Defaults to the Wan-validated resolution "
        "(settings.comfyui_animation_default_width), not a generic value — see this task's "
        "engineering notes on why the animation path must never silently resolve to 1024x1024.",
    )
    height: int = Field(
        default_factory=lambda: get_settings().comfyui_animation_default_height,
        gt=0,
        description="Output video height, in pixels. Defaults to settings.comfyui_animation_default_height.",
    )
    frame_duration_seconds: float = Field(
        default=2.0, gt=0, description="How long each frame is held in the output video, in seconds."
    )


class FightVideoResponse(BaseModel):
    """The complete pipeline's response shape."""

    status: str = Field(default="completed", description="Pipeline completion status.")
    fight_id: str = Field(..., min_length=1, description="Identifier assigned to this render.")
    frame_count: int = Field(..., ge=1, description="Number of frames rendered and assembled.")
    video_path: str = Field(..., min_length=1, description="Path to the assembled fight.mp4.")
    timeline: list[Shot] = Field(..., min_length=1, description="The Timeline Engine's shot-by-shot plan.")
    frames: list[RenderedFrame] = Field(..., min_length=1, description="Every rendered frame and its metadata.")


class FightVideoPipeline:
    """Runs the complete Sprint 2 + Sprint 3 + Sprint 4 pipeline for
    one PGN, producing a finished animated video.

    Every dependency is injected — `render_pipeline`, `animation_pipeline`,
    and `video_builder` default to fresh instances (which themselves
    default to the shared `ImageRouter`/`AnimationRouter` singletons and
    the `ffmpeg` on PATH, respectively) so a test can substitute a
    pipeline wired to mock providers without any global state.
    """

    def __init__(
        self,
        ai_provider: AIProvider,
        render_pipeline: RenderPipeline | None = None,
        animation_pipeline: AnimationPipeline | None = None,
        video_builder: VideoBuilder | None = None,
        asset_manager: AssetManager | None = None,
    ) -> None:
        """Initializes the pipeline.

        Args:
            ai_provider: Passed through to FightOrchestrator, exactly
                as api/chess2fight.py already does for the existing
                `/generate` route.
            render_pipeline: Renders shots to static reference frames.
                Defaults to a fresh RenderPipeline (using the shared
                ImageRouter).
            animation_pipeline: Animates each reference frame into a
                per-shot clip. Defaults to a fresh AnimationPipeline
                (using the shared AnimationRouter) — never a concrete
                AnimationProvider; see animation_pipeline.py's own
                module docstring.
            video_builder: Concatenates the animated clips into the
                final video. Defaults to a fresh VideoBuilder (using
                `ffmpeg` on PATH).
            asset_manager: Used only to resolve where a fight's frames
                were saved, for locating the final video's output path.
                Defaults to a fresh AssetManager.
        """
        self._orchestrator = FightOrchestrator(ai_provider)
        self._render_pipeline = render_pipeline or RenderPipeline()
        self._animation_pipeline = animation_pipeline or AnimationPipeline()
        self._video_builder = video_builder or VideoBuilder()
        self._asset_manager = asset_manager or AssetManager()

    async def run(
        self,
        pgn: str,
        preferences: BattlePreferences,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        frame_duration_seconds: float = 2.0,
        fight_id: str | None = None,
    ) -> FightVideoResponse:
        """Runs the complete pipeline: analysis through a finished video.

        Args:
            pgn: PGN text of the game to render.
            preferences: Style/battle-mode preferences, exactly as the
                existing `/generate` route already accepts.
            fps: Output video frame rate. Defaults to
                `settings.comfyui_default_fps` (currently 8) — the
                current Wan-validated policy — when not given. A prior
                literal default of 24 here (and on `RenderVideoRequest`)
                had silently drifted out of sync with that setting once
                it changed from 24 to 8 (Sprint 4 Prompt 8); fixed to
                resolve from the setting directly, Sprint 4 Prompt 10.1.
            width: Output video (animation) width, in pixels. Defaults
                to `settings.comfyui_animation_default_width` (832) —
                the Wan-validated resolution — not a generic value, so
                this never silently resolves to 1024x1024. This is
                distinct from the FLUX reference-image resolution
                below: the two are independently resolved and passed
                to different steps. (Sprint 4 Prompt 10.1: an earlier
                version of this docstring said the reference image was
                "unaffected either way" and "never passed width/height
                here at all" — true before Sprint 4 Prompt 10's own
                fix, no longer true now that `render()` below is
                explicitly passed the FLUX policy; corrected here.)
            height: Output video (animation) height, in pixels.
                Defaults to `settings.comfyui_animation_default_height`
                (480).
            frame_duration_seconds: No longer used as of Sprint 4
                Prompt 2 — each shot's own duration (from
                Shot.duration_seconds, via AnimationInstruction) now
                controls its clip's length instead of one uniform
                value applied to every shot. Still accepted here
                (and still a field on RenderVideoRequest) purely so an
                existing caller passing it doesn't get a validation
                error; it has no effect on the resulting video.
            fight_id: Optional explicit fight identifier. If omitted,
                a new one is generated — nothing upstream of this
                pipeline currently carries a persistent fight
                identifier of its own (see render_pipeline.py's
                engineering notes from the prior task), so this is the
                first point in the whole pipeline where one exists.

        Returns:
            A FightVideoResponse with the finished video's path, every
            rendered frame's metadata, and the shot-by-shot timeline.
        """
        settings = get_settings()
        resolved_fps = fps if fps is not None else settings.comfyui_default_fps
        resolved_animation_width = width if width is not None else settings.comfyui_animation_default_width
        resolved_animation_height = height if height is not None else settings.comfyui_animation_default_height
        # Sprint 4 Prompt 10: the FLUX/Chess2Fight image-generation policy
        # (settings.comfyui_image_default_width/height, 1280x704) was
        # defined in config but never actually reached RenderPipeline —
        # generate_image() was called with only a prompt, silently using
        # ImageProvider's own generic 1024x1024 default instead. Resolved
        # here, at this Chess2Fight-specific call site, and passed through
        # explicitly — RenderPipeline itself stays provider-agnostic (see
        # its own docstring on why `None` there means "use the generic
        # default", not "use FLUX's").
        resolved_image_width = settings.comfyui_image_default_width
        resolved_image_height = settings.comfyui_image_default_height

        resolved_fight_id = fight_id or uuid.uuid4().hex

        generate_response = await self._orchestrator.generate_fight(pgn, preferences)
        logger.info(
            "Fight %s: analysis complete, %d shots to render.",
            resolved_fight_id, generate_response.prompted_timeline.shot_count,
        )

        render_output = await self._render_pipeline.render(
            generate_response.prompted_timeline, resolved_fight_id,
            width=resolved_image_width, height=resolved_image_height,
        )
        logger.info("Fight %s: rendered %d reference frames.", resolved_fight_id, render_output.frame_count)

        animation_output = await self._animation_pipeline.animate(
            render_output, generate_response.prompted_timeline,
            width=resolved_animation_width, height=resolved_animation_height, fps=resolved_fps,
        )
        logger.info("Fight %s: animated %d shot clips.", resolved_fight_id, animation_output.shot_count)

        clip_paths = [shot.video_path for shot in animation_output.animated_shots]
        total_duration_seconds = sum(shot.duration_seconds for shot in animation_output.animated_shots)

        video_path = str(self._asset_manager.fight_directory(resolved_fight_id) / "fight.mp4")
        video_result = await self._video_builder.concatenate_clips(
            clip_paths=clip_paths,
            output_path=video_path,
            total_duration_seconds=total_duration_seconds,
            fps=resolved_fps,
            width=resolved_animation_width,
            height=resolved_animation_height,
        )
        logger.info(
            "Fight %s: video assembled at %s (%.1fs, %d shot clips).",
            resolved_fight_id, video_result.video_path, video_result.duration_seconds, len(clip_paths),
        )

        return FightVideoResponse(
            status="completed",
            fight_id=resolved_fight_id,
            frame_count=render_output.frame_count,
            video_path=video_result.video_path,
            timeline=list(generate_response.shot_timeline.shots),
            frames=render_output.frames,
        )
