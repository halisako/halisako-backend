"""FightVideoPipeline: the complete Sprint 3 pipeline, end to end.

    PGN -> Battle Director -> Battle Intelligence -> Fight Story ->
    Timeline Engine -> Scene Composer -> Prompt Generator ->
    ImageRouter -> Render Pipeline -> Video Builder -> fight.mp4

This file isn't one the brief explicitly named — it only named
`video_builder.py` — but "integrate the complete pipeline" needs this
orchestration to live somewhere, and putting it here (rather than
inline in the API route) keeps `api/chess2fight.py` thin, matching
that module's own stated principle ("no chess logic or AI logic lives
here"), and keeps `orchestrator.py` completely untouched — this module
calls `FightOrchestrator.generate_fight()` as-is, adding nothing to
it, so every existing Sprint 2 test is unaffected by this file's
existence.

`FightVideoPipeline.run()` is the single entry point: it reuses
`FightOrchestrator.generate_fight()` to get everything through
`prompted_timeline` (already computed by prior Sprint 3 work), then
generates a `fight_id`, renders every shot to a frame via
`RenderPipeline`, and assembles those frames into `fight.mp4` via
`VideoBuilder`.
"""

from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel, Field

from core.ai_router import AIProvider
from products.chess2fight.cinematic.schemas import Shot
from products.chess2fight.orchestrator import FightOrchestrator
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
    fps: int = Field(default=24, gt=0, description="Output video frame rate.")
    width: int = Field(default=1024, gt=0, description="Output video width, in pixels.")
    height: int = Field(default=1024, gt=0, description="Output video height, in pixels.")
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
    """Runs the complete Sprint 2 + Sprint 3 pipeline for one PGN,
    producing a finished video.

    Every dependency is injected — `render_pipeline` and
    `video_builder` default to fresh instances (which themselves
    default to the shared `ImageRouter` singleton and the `ffmpeg` on
    PATH, respectively) so a test can substitute a `RenderPipeline`
    wired to `MockImageProvider` without any global state.
    """

    def __init__(
        self,
        ai_provider: AIProvider,
        render_pipeline: RenderPipeline | None = None,
        video_builder: VideoBuilder | None = None,
        asset_manager: AssetManager | None = None,
    ) -> None:
        """Initializes the pipeline.

        Args:
            ai_provider: Passed through to FightOrchestrator, exactly
                as api/chess2fight.py already does for the existing
                `/generate` route.
            render_pipeline: Renders shots to frames. Defaults to a
                fresh RenderPipeline (using the shared ImageRouter).
            video_builder: Assembles frames into a video. Defaults to
                a fresh VideoBuilder (using `ffmpeg` on PATH).
            asset_manager: Used only to resolve where a fight's frames
                were saved, for handing that directory to VideoBuilder.
                Defaults to a fresh AssetManager.
        """
        self._orchestrator = FightOrchestrator(ai_provider)
        self._render_pipeline = render_pipeline or RenderPipeline()
        self._video_builder = video_builder or VideoBuilder()
        self._asset_manager = asset_manager or AssetManager()

    async def run(
        self,
        pgn: str,
        preferences: BattlePreferences,
        fps: int = 24,
        width: int = 1024,
        height: int = 1024,
        frame_duration_seconds: float = 2.0,
        fight_id: str | None = None,
    ) -> FightVideoResponse:
        """Runs the complete pipeline: analysis through a finished video.

        Args:
            pgn: PGN text of the game to render.
            preferences: Style/battle-mode preferences, exactly as the
                existing `/generate` route already accepts.
            fps: Output video frame rate.
            width: Output video width, in pixels.
            height: Output video height, in pixels.
            frame_duration_seconds: How long each frame is held in the
                assembled video — see video_builder.py's docstring for
                why this is uniform rather than per-shot.
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
        resolved_fight_id = fight_id or uuid.uuid4().hex

        generate_response = await self._orchestrator.generate_fight(pgn, preferences)
        logger.info(
            "Fight %s: analysis complete, %d shots to render.",
            resolved_fight_id, generate_response.prompted_timeline.shot_count,
        )

        render_output = await self._render_pipeline.render(
            generate_response.prompted_timeline, resolved_fight_id
        )
        logger.info("Fight %s: rendered %d frames.", resolved_fight_id, render_output.frame_count)

        video_path = str(self._asset_manager.fight_directory(resolved_fight_id) / "fight.mp4")
        video_result = await self._video_builder.build_video(
            frame_directory=render_output.output_dir,
            output_path=video_path,
            frame_count=render_output.frame_count,
            fps=fps,
            width=width,
            height=height,
            frame_duration_seconds=frame_duration_seconds,
        )
        logger.info(
            "Fight %s: video assembled at %s (%.1fs).",
            resolved_fight_id, video_result.video_path, video_result.duration_seconds,
        )

        return FightVideoResponse(
            status="completed",
            fight_id=resolved_fight_id,
            frame_count=render_output.frame_count,
            video_path=video_result.video_path,
            timeline=list(generate_response.shot_timeline.shots),
            frames=render_output.frames,
        )
