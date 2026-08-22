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

Sprint 4 Prompt 7.1 added `max_animation_seconds` — an acceptance-only
cap on the duration passed to the animation step, so a low-cost GPU
smoke test (e.g. a real 49-frame Wan clip at 24fps/2s) can run before
spending GPU time animating a shot's full real duration (which can run
well past 100+ frames). The real `PromptedShot.duration_seconds` this
cap is applied against is never mutated — `prepare()` computes a
separate `effective_animation_duration_seconds` field, and `execute()`
builds a non-mutating `model_copy()` of the shot (new object, original
untouched) only when a cap actually changes the duration used; when no
cap is given, the exact same timeline object used for rendering is
reused for animation too, so that path is structurally — not just
numerically — identical to this module's original Prompt 7 behavior.
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

    `shot.duration_seconds` is always the real, unmodified cinematic
    duration — Sprint 4 Prompt 7.1's acceptance-only animation cap
    (`max_animation_seconds`) never touches it. `shot` is never
    mutated to reflect a cap; `effective_animation_duration_seconds`
    is the separate, explicit field that does, so the two are never
    confusable in code that reads this plan.
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
    max_animation_seconds: float | None = Field(
        default=None,
        description="Acceptance-only animation duration cap requested via prepare(), if any. "
        "None means no cap was requested — the full real shot duration is used, unchanged from "
        "Sprint 4 Prompt 7's original behavior.",
    )
    effective_animation_duration_seconds: float = Field(
        ...,
        gt=0,
        description="The duration actually used to build AnimationInstruction: "
        "min(shot.duration_seconds, max_animation_seconds) if a cap was requested, "
        "else exactly shot.duration_seconds, unchanged.",
    )
    calculated_wan_frame_count: int = Field(
        ..., ge=1, description="The Wan-valid (4n+1) frame count for effective_animation_duration_seconds."
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
        max_animation_seconds: float | None = None,
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
            max_animation_seconds: Sprint 4 Prompt 7.1 — an
                acceptance-only cap on the duration passed to the
                animation step, for a low-cost GPU smoke test before
                spending GPU time on a shot's full real duration. Must
                be > 0 if given. Never changes `shot.duration_seconds`
                itself (the real cinematic duration) — only
                `effective_animation_duration_seconds` on the returned
                plan. `None` (the default) means no cap: the full real
                shot duration is used, and every returned field
                behaves exactly as Sprint 4 Prompt 7's original
                implementation did.

        Returns:
            A SingleShotPlan describing exactly what execute() would do.

        Raises:
            ShotIndexOutOfRangeError: If `shot_index` isn't a valid
                index into the timeline's actual shot list.
            ValueError: If `max_animation_seconds` is given and isn't > 0.
            InvalidPGNError: If `pgn` itself doesn't parse — propagates
                unchanged from FightOrchestrator/the PGN analyzer.
        """
        if max_animation_seconds is not None and max_animation_seconds <= 0:
            raise ValueError(f"max_animation_seconds must be > 0, got {max_animation_seconds!r}.")

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

        effective_animation_duration = (
            min(selected_shot.duration_seconds, max_animation_seconds)
            if max_animation_seconds is not None
            else selected_shot.duration_seconds
        )

        logger.info(
            "Single-shot acceptance: selected shot %d/%d (%s, real=%.2fs, effective=%.2fs).",
            shot_index, len(shots), selected_shot.shot_type.value,
            selected_shot.duration_seconds, effective_animation_duration,
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
            max_animation_seconds=max_animation_seconds,
            effective_animation_duration_seconds=effective_animation_duration,
            calculated_wan_frame_count=_duration_to_frame_count(effective_animation_duration, resolved_fps),
            fps=resolved_fps,
        )

    async def execute(
        self,
        plan: SingleShotPlan,
        width: int | None = None,
        height: int | None = None,
    ) -> SingleShotAcceptanceResult:
        """Actually renders and animates the planned shot, via the
        real, currently-configured ImageRouter/AnimationRouter.

        Args:
            plan: A plan from `prepare()`.
            width: Output width for the animated clip. Defaults to
                `settings.comfyui_animation_default_width` (832) — the
                Wan-validated resolution (Sprint 4 Prompt 8/9), not a
                generic value. Note this affects only the animation
                step: `RenderPipeline.render()` below is never passed
                width/height at all, so the still reference image's
                resolution is whatever the configured ImageProvider
                itself defaults to, independent of this parameter. An
                earlier version of this docstring claimed this also
                controlled "the reference image," which the code never
                actually did — corrected here, not just the default
                value.
            height: Output height for the animated clip. Defaults to
                `settings.comfyui_animation_default_height` (480).

        Returns:
            A SingleShotAcceptanceResult with the real local image and
            video paths.
        """
        settings = get_settings()
        resolved_width = width if width is not None else settings.comfyui_animation_default_width
        resolved_height = height if height is not None else settings.comfyui_animation_default_height

        render_timeline = self._build_single_shot_timeline(plan, plan.shot)
        render_output = await self._render_pipeline.render(render_timeline, plan.fight_id)

        if plan.effective_animation_duration_seconds == plan.shot.duration_seconds:
            # No cap requested (or the requested cap wasn't below the
            # real duration) — reuse the exact same timeline object
            # rather than constructing an equivalent copy, so this
            # path is structurally identical to Sprint 4 Prompt 7's
            # original behavior, not just numerically equivalent to it.
            animation_timeline = render_timeline
        else:
            # plan.shot itself is never touched — model_copy returns a
            # new, separate PromptedShot; the original real cinematic
            # duration on plan.shot remains exactly what
            # FightOrchestrator produced.
            animation_shot = plan.shot.model_copy(
                update={"duration_seconds": plan.effective_animation_duration_seconds}
            )
            animation_timeline = self._build_single_shot_timeline(plan, animation_shot)

        animation_output = await self._animation_pipeline.animate(
            render_output, animation_timeline, width=resolved_width, height=resolved_height, fps=plan.fps,
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

    def _build_single_shot_timeline(self, plan: SingleShotPlan, shot: PromptedShot) -> PromptedTimeline:
        """Constructs a valid, minimal PromptedTimeline containing only
        `shot` (either `plan.shot` unmodified, or a duration-capped
        copy of it — the caller decides which). RenderPipeline.render()
        and AnimationPipeline.animate() need no changes at all for
        this — both already operate purely via
        `for shot in timeline.shots`, with no assumption about a fixed
        shot count."""
        return PromptedTimeline(
            shots=[shot],
            total_duration_seconds=shot.duration_seconds,
            shot_count=1,
            scene_continuity=plan.scene_continuity,
        )
