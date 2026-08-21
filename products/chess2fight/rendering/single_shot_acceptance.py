"""SingleShotAcceptanceRunner: proves the real Chess2Fight rendering
architecture — RenderPipeline, AnimationPipeline, ImageRouter,
AnimationRouter — can drive one real cinematic shot end to end, without
rendering the whole fight.

    PGN -> FightOrchestrator -> PromptedTimeline -> select ONE Shot ->
    (a single-shot PromptedTimeline) -> RenderPipeline -> AnimationPipeline
    -> one real image path + one real video path

This exists because Sprint 4 Prompt 6 productionized both ComfyUI
providers against externally-validated real workflows, but neither had
been driven by the actual production Chess2Fight rendering classes —
only by hand-built AnimationInstruction/prompt values in tests. This
module closes that gap using the real production contracts unchanged:
`RenderPipeline.render()` and `AnimationPipeline.animate()` are called
exactly as `FightVideoPipeline.run()` calls them — the only difference
is the `PromptedTimeline` passed in contains one shot instead of eight.
Both methods already operate purely by iterating `timeline.shots` with
no assumption about a fixed shot count (verified directly against
their current source before writing this), so this needed zero changes
to either — or to `RenderPipeline`, `AnimationPipeline`,
`FightOrchestrator`, `BattleDirector`, `CinematicEngine`, or
`CombatMapper`. `FightVideoPipeline`/`api/chess2fight.py` (the
production `/generate` and `/render` routes) are equally untouched —
this is a new, separate module, not a modification of either.

Two-phase design, so a caller (the CLI script, or a dry-run test) can
inspect what *would* happen before any ComfyUI/network call is made:

- `prepare()` runs the real orchestration/cinematic pipeline and
  selects one shot. Makes zero ComfyUI or network calls — safe for
  dry-run and for tests that must never require a GPU.
- `execute()` takes a `SingleShotPlan` and actually renders + animates
  it, via the real, configured `ImageRouter`/`AnimationRouter` (mock
  or comfyui, whichever `settings.image_provider`/`.animation_provider`
  currently say).

Never calls `VideoBuilder.concatenate_clips()` — there is exactly one
clip, so there is nothing to concatenate; the single animated clip's
own path *is* the result.
"""

from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel, ConfigDict, Field

from core.ai_router import AIProvider
from core.animation_providers.comfyui import _duration_to_frame_count
from core.config import get_settings
from products.chess2fight.cinematic.schemas import PromptedShot, PromptedTimeline, SceneContinuity
from products.chess2fight.orchestrator import FightOrchestrator
from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
from products.chess2fight.rendering.render_pipeline import RenderPipeline
from products.chess2fight.schemas import BattlePreferences

logger = logging.getLogger(__name__)


class SingleShotPlan(BaseModel):
    """What would happen for one shot, computed with zero ComfyUI or
    network calls — the dry-run-safe half of this module's output.
    Carries the selected shot's own real image_prompt/duration/camera
    intent/cinematic metadata/combat context unchanged (it *is* the
    real `PromptedShot`, not a summary of it), plus everything needed
    to actually execute this plan later without re-running
    orchestration.
    """

    model_config = ConfigDict(frozen=True)

    fight_id: str = Field(..., min_length=1, description="Identifier this single-shot run will use.")
    shot_index: int = Field(..., ge=0, description="Which shot, by position in the full timeline, was selected.")
    shot: PromptedShot = Field(..., description="The actual selected Shot, unmodified — its own real image_prompt.")
    scene_continuity: SceneContinuity = Field(
        ..., description="The full timeline's scene continuity — needed to reconstruct a valid single-shot timeline."
    )
    total_shots_in_timeline: int = Field(..., ge=1, description="How many shots the full timeline actually had.")
    image_provider: str = Field(..., min_length=1, description="Configured image provider (e.g. 'mock', 'comfyui').")
    animation_provider: str = Field(..., min_length=1, description="Configured animation provider.")
    comfyui_base_url: str = Field(..., min_length=1, description="Configured ComfyUI server URL.")
    comfyui_image_workflow_path: str = Field(..., min_length=1)
    comfyui_animation_workflow_path: str = Field(..., min_length=1)
    calculated_wan_frame_count: int = Field(
        ..., ge=1, description="The Wan-valid (4n+1) frame count this shot's duration will produce."
    )
    fps: int = Field(..., gt=0, description="FPS the frame count above was calculated against.")


class SingleShotAcceptanceResult(BaseModel):
    """What actually happened, after `execute()` called the real
    (possibly mock, possibly comfyui) providers."""

    plan: SingleShotPlan = Field(..., description="The plan this result was executed from.")
    image_path: str = Field(..., min_length=1, description="Local path to the rendered reference image.")
    video_path: str = Field(..., min_length=1, description="Local path to the single animated clip.")
    video_duration_seconds: float = Field(..., gt=0, description="Actual duration of the produced clip.")


class ShotIndexOutOfRangeError(ValueError):
    """Raised by `prepare()` for an invalid `shot_index` — a plain,
    specific ValueError subclass rather than a new project-wide
    exception, since this never needs to cross the
    provider/router/pipeline error hierarchy those exist for; it's a
    caller-input-validation error local to this module."""


class SingleShotAcceptanceRunner:
    """Drives exactly one cinematic shot through the real rendering
    architecture. Every dependency is injected, matching
    `FightVideoPipeline`'s own pattern — `render_pipeline` and
    `animation_pipeline` default to fresh instances using the shared
    `ImageRouter`/`AnimationRouter` singletons, so whichever provider
    is currently configured (mock or comfyui) is what actually runs.
    """

    def __init__(
        self,
        ai_provider: AIProvider,
        render_pipeline: RenderPipeline | None = None,
        animation_pipeline: AnimationPipeline | None = None,
    ) -> None:
        self._orchestrator = FightOrchestrator(ai_provider)
        self._render_pipeline = render_pipeline or RenderPipeline()
        self._animation_pipeline = animation_pipeline or AnimationPipeline()

    async def prepare(
        self,
        pgn: str,
        preferences: BattlePreferences,
        shot_index: int = 0,
        fps: int | None = None,
    ) -> SingleShotPlan:
        """Runs the real orchestration/cinematic pipeline and selects
        one shot. Makes no ComfyUI or network calls — safe for
        dry-run.

        Args:
            pgn: PGN text of the game to analyze — the real
                FightOrchestrator/PGN analyzer runs unchanged.
            preferences: Style/battle-mode preferences, same as
                FightVideoPipeline.run() already accepts.
            shot_index: Which shot to select, 0-indexed into the full
                timeline's shot list.
            fps: FPS to calculate the Wan frame count against.
                Defaults to `settings.comfyui_default_fps`.

        Returns:
            A SingleShotPlan describing exactly what execute() would do.

        Raises:
            ShotIndexOutOfRangeError: If `shot_index` isn't a valid
                index into the timeline's actual shot list.
            InvalidPGNError: If `pgn` itself doesn't parse — propagates
                unchanged from FightOrchestrator/the PGN analyzer.
        """
        settings = get_settings()

        generate_response = await self._orchestrator.generate_fight(pgn, preferences)
        timeline = generate_response.prompted_timeline
        shots = timeline.shots

        if not (0 <= shot_index < len(shots)):
            raise ShotIndexOutOfRangeError(
                f"shot_index {shot_index} is out of range — this timeline has {len(shots)} shots "
                f"(valid indices: 0-{len(shots) - 1})."
            )

        selected_shot = shots[shot_index]
        resolved_fps = fps if fps is not None else settings.comfyui_default_fps

        logger.info(
            "Single-shot acceptance: selected shot %d/%d (%s, %.2fs).",
            shot_index, len(shots), selected_shot.shot_type.value, selected_shot.duration_seconds,
        )

        return SingleShotPlan(
            fight_id=f"single_shot_{uuid.uuid4().hex}",
            shot_index=shot_index,
            shot=selected_shot,
            scene_continuity=timeline.scene_continuity,
            total_shots_in_timeline=len(shots),
            image_provider=settings.image_provider,
            animation_provider=settings.animation_provider,
            comfyui_base_url=settings.comfyui_base_url,
            comfyui_image_workflow_path=settings.comfyui_image_workflow_path,
            comfyui_animation_workflow_path=settings.comfyui_workflow_path,
            calculated_wan_frame_count=_duration_to_frame_count(selected_shot.duration_seconds, resolved_fps),
            fps=resolved_fps,
        )

    async def execute(
        self,
        plan: SingleShotPlan,
        width: int = 1280,
        height: int = 704,
    ) -> SingleShotAcceptanceResult:
        """Actually renders and animates the planned shot, via the
        real, currently-configured ImageRouter/AnimationRouter.

        Args:
            plan: A plan from `prepare()`.
            width: Output width for the reference image/clip. Defaults
                to 1280 — the experimentally-validated FLUX resolution
                (Sprint 4 Prompt 6), not the generic 1024 default,
                since this path is specifically for exercising the
                real, validated ComfyUI-shaped providers.
            height: Output height. Defaults to 704, pairing with the
                width default above.

        Returns:
            A SingleShotAcceptanceResult with the real local image and
            video paths.
        """
        single_shot_timeline = self._build_single_shot_timeline(plan)

        render_output = await self._render_pipeline.render(single_shot_timeline, plan.fight_id)
        animation_output = await self._animation_pipeline.animate(
            render_output, single_shot_timeline, width=width, height=height, fps=plan.fps,
        )

        # Exactly one shot in, exactly one animated clip out — asserted,
        # not just assumed, since a silent mismatch here would be a
        # much more confusing failure mode downstream.
        assert len(animation_output.animated_shots) == 1, (
            f"Expected exactly 1 animated shot, got {len(animation_output.animated_shots)} — "
            "single-shot acceptance invariant violated."
        )
        animated_shot = animation_output.animated_shots[0]

        return SingleShotAcceptanceResult(
            plan=plan,
            image_path=render_output.frames[0].frame_path,
            video_path=animated_shot.video_path,
            video_duration_seconds=animated_shot.duration_seconds,
        )

    def _build_single_shot_timeline(self, plan: SingleShotPlan) -> PromptedTimeline:
        """Constructs a valid, minimal PromptedTimeline containing only
        the planned shot. RenderPipeline.render() and
        AnimationPipeline.animate() need no changes at all for this —
        both already operate purely via `for shot in timeline.shots`,
        with no assumption about a fixed shot count."""
        return PromptedTimeline(
            shots=[plan.shot],
            total_duration_seconds=plan.shot.duration_seconds,
            shot_count=1,
            scene_continuity=plan.scene_continuity,
        )
