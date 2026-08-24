"""MultiShotAcceptanceRunner: generalizes
`SingleShotAcceptanceRunner`'s proven pattern (Sprint 4 Prompt 7,
GPU-validated Sprint 4 Prompt 10/10.1) from exactly one shot to a
capped range of N real cinematic shots, ending in one locally
concatenated MP4 — Sprint 4 Prompt 11.

    PGN -> FightOrchestrator -> PromptedTimeline -> select shots
    [start_index, start_index + shot_count) -> (a trimmed
    PromptedTimeline) -> RenderPipeline -> AnimationPipeline ->
    N real image paths + N real clip paths -> VideoBuilder.concatenate_clips()
    -> one final assembled MP4

Audited directly against current source (not assumed) before writing
this, per this task's own "First Task" — the findings that make this
generalization safe:

- `RenderPipeline.render()` and `AnimationPipeline.animate()` already
  operate purely by iterating `timeline.shots`, with zero hardcoded
  shot-count assumption — exactly the same property that made the
  single-shot generalization safe in Prompt 7, extends unchanged to N
  shots. Neither needed any modification.
- `RenderPipeline.render()`'s frame list is built via a sequential
  (not concurrent-gather) list comprehension over `timeline.shots` —
  output order is guaranteed to match input order, not just typically
  matching it.
- `AnimationPipeline.animate()` goes further: after its own sequential
  loop, it explicitly re-sorts `animated_shots` by `sequence_order`
  before returning — an intentional, already-existing ordering
  guarantee, not something this module needed to add. Verified with a
  dedicated test here anyway (this task explicitly asks for one), but
  the guarantee itself was already there.
- `VideoBuilder.concatenate_clips()`'s own docstring documents its
  ordering contract explicitly: "the output preserves this order" —
  and re-encodes rather than stream-copying, so minor per-clip
  variation between independently-generated Wan clips doesn't break
  concatenation.
- Both `RenderPipeline.render()` and `AnimationPipeline.animate()`
  have no internal try/except of their own — a mid-sequence failure
  (`ImageProviderError`/`AnimationProviderError`) propagates
  immediately, never silently skipped or reported as success. Because
  each shot's provider call writes its output file before the next
  shot's call begins (the loop is sequential, not concurrent), any
  shots that succeeded before the failure remain genuinely on disk —
  but neither method *returns* that partial list to the caller when it
  raises, so this runner cannot report "shots 0,1 succeeded, shot 2
  failed" as structured data from a single `execute()` call — only via
  the raised exception's own message (which does name the failing
  shot_id/frame_number) plus the artifact directories a human can
  inspect afterward. Documented here as a known, honest limitation
  rather than solved by restructuring RenderPipeline/AnimationPipeline
  — this task's core engineering principle is to preserve and reuse
  them unchanged, not redesign their internals for this one caller.

Two-phase design, matching `SingleShotAcceptanceRunner` exactly:

- `prepare()` runs the real orchestration/cinematic pipeline and
  selects a shot range. Makes zero ComfyUI or network calls.
- `execute()` takes a `MultiShotPlan` and actually renders + animates
  every selected shot, then concatenates the results via the real
  `VideoBuilder.concatenate_clips()` — never a separate, ad-hoc ffmpeg
  invocation.

A hard, explicit safety cap (`_DEFAULT_MAX_SHOT_COUNT = 3`) applies
unless the caller deliberately opts out via
`allow_exceeding_default_cap=True` — this task's own explicit "make it
difficult to accidentally render the entire timeline" requirement.

`max_animation_seconds` (the same acceptance-only cap
`SingleShotAcceptanceRunner` already established) applies uniformly to
every selected shot here, via the same non-mutating `model_copy()`
pattern per shot — the original `PromptedShot.duration_seconds` values
in the real timeline are never touched.
"""

from __future__ import annotations

import logging
import subprocess
import uuid

from pydantic import BaseModel, ConfigDict, Field

from core.ai_router import AIProvider
from core.animation_providers.comfyui import _duration_to_frame_count, _frame_count_to_duration
from core.config import get_settings
from products.chess2fight.cinematic.schemas import PromptedShot, PromptedTimeline, SceneContinuity
from products.chess2fight.orchestrator import FightOrchestrator
from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.render_pipeline import RenderPipeline
from products.chess2fight.rendering.video_builder import VideoBuilder
from products.chess2fight.schemas import BattlePreferences

logger = logging.getLogger(__name__)

# The first paid multi-shot milestone's own explicit cap (Sprint 4
# Prompt 11) — not a generic constant reused from anywhere else, since
# no other part of this codebase has previously needed a shot-count
# ceiling. Deliberately a plain module constant, not a settings field:
# this is a one-off acceptance-harness safeguard against accidentally
# rendering the full 8-shot timeline, not a durable runtime policy.
_DEFAULT_MAX_SHOT_COUNT = 3


class MultiShotPlan(BaseModel):
    """What would happen for a capped range of shots, computed with
    zero ComfyUI or network calls — the dry-run-safe half of this
    module's output.

    `shots[i].duration_seconds` is always each shot's real, unmodified
    cinematic duration — the acceptance-only animation cap never
    touches it (mirroring `SingleShotPlan`'s same guarantee).
    `effective_animation_durations_seconds[i]` is the separate,
    explicit field that reflects the cap, so the two can never be
    confused by code reading this plan.
    """

    model_config = ConfigDict(frozen=True)

    fight_id: str = Field(..., min_length=1, description="Identifier this multi-shot run will use.")
    start_shot_index: int = Field(..., ge=0, description="First timeline position selected.")
    shot_count: int = Field(..., ge=1, description="How many consecutive shots were selected.")
    selected_shot_indices: list[int] = Field(
        ..., min_length=1, description="Explicit list of timeline positions selected, in order — "
        "e.g. [0, 1, 2] for start_shot_index=0, shot_count=3."
    )
    shots: list[PromptedShot] = Field(
        ..., min_length=1, description="The actual selected Shots, unmodified, in timeline order — "
        "each one's own real image_prompt, not a summary of it."
    )
    scene_continuity: SceneContinuity = Field(
        ..., description="The full timeline's scene continuity — needed to reconstruct a valid trimmed timeline."
    )
    total_shots_in_timeline: int = Field(..., ge=1, description="How many shots the full timeline actually had.")
    image_provider: str = Field(..., min_length=1, description="Configured image provider (e.g. 'mock', 'comfyui').")
    animation_provider: str = Field(..., min_length=1, description="Configured animation provider.")
    comfyui_base_url: str = Field(..., min_length=1, description="Configured ComfyUI server URL.")
    comfyui_image_workflow_path: str = Field(..., min_length=1)
    comfyui_animation_workflow_path: str = Field(..., min_length=1)
    max_animation_seconds: float | None = Field(
        default=None,
        description="Acceptance-only per-shot animation duration cap requested via prepare(), if any. "
        "None means no cap — each shot's full real duration is used.",
    )
    effective_animation_durations_seconds: list[float] = Field(
        ..., min_length=1, description="Per-shot duration actually used to build each AnimationInstruction, "
        "same order as `shots`: min(shot.duration_seconds, max_animation_seconds) if a cap was requested, "
        "else exactly shot.duration_seconds.",
    )
    calculated_wan_frame_counts: list[int] = Field(
        ..., min_length=1, description="Per-shot Wan-valid (4n+1) frame count, same order as `shots`."
    )
    fps: int = Field(..., gt=0, description="FPS every frame count above was calculated against.")
    expected_comfyui_job_count: int = Field(
        ..., ge=2, description="2 * shot_count — one FLUX job and one Wan job per selected shot. "
        "Concatenation is local (VideoBuilder/ffmpeg) and never counted here.",
    )
    expected_assembled_duration_seconds: float = Field(
        ..., gt=0, description="Sum of each shot's actual post-frame-snapping clip duration "
        "(_frame_count_to_duration per shot) — the best available estimate before real "
        "container/encoding rounding, not a guarantee of the exact final measured value.",
    )


class MultiShotAcceptanceResult(BaseModel):
    """What actually happened, after `execute()` called the real
    (possibly mock, possibly comfyui) providers for every selected
    shot and concatenated the results."""

    plan: MultiShotPlan = Field(..., description="The plan this result was executed from.")
    image_paths: list[str] = Field(
        ..., min_length=1, description="Local paths to each shot's rendered reference image, in timeline order."
    )
    video_paths: list[str] = Field(
        ..., min_length=1, description="Local paths to each shot's own animated clip, in timeline order — "
        "these are the inputs concatenate_clips() was given, before assembly.",
    )
    final_video_path: str = Field(..., min_length=1, description="Path to the one final concatenated MP4.")
    final_video_duration_seconds: float = Field(
        ..., gt=0,
        description="The final MP4's duration as measured directly with ffprobe after concatenation "
        "completes — not VideoBuilder's own returned value, which is the sum of each clip's own "
        "reported duration (an expectation, not a measurement of the actual assembled container). "
        "Sprint 4 Prompt 11.1: an earlier version of this field held that predicted sum, mislabeled "
        "as 'actual'. `plan.expected_assembled_duration_seconds` remains the separate, pre-generation "
        "estimate for comparison.",
    )
    resolved_image_width: int = Field(..., gt=0, description="The actual FLUX image width used for this run.")
    resolved_image_height: int = Field(..., gt=0, description="The actual FLUX image height used for this run.")
    resolved_animation_width: int = Field(..., gt=0, description="The actual Wan animation width used for this run.")
    resolved_animation_height: int = Field(..., gt=0, description="The actual Wan animation height used for this run.")


class ShotRangeOutOfRangeError(ValueError):
    """Raised by `prepare()` for a shot range that doesn't fit within
    the real timeline — a plain, specific ValueError subclass, same
    reasoning as `single_shot_acceptance.py`'s `ShotIndexOutOfRangeError`:
    caller-input validation local to this module, not a
    provider/router/pipeline error."""


class ShotCountExceedsAcceptanceCapError(ValueError):
    """Raised by `prepare()` when `shot_count` exceeds
    `_DEFAULT_MAX_SHOT_COUNT` and the caller hasn't explicitly opted
    out via `allow_exceeding_default_cap=True` — this task's own
    explicit cost-control requirement: "make it difficult to
    accidentally render the entire timeline.\""""


class FinalVideoMeasurementError(RuntimeError):
    """Raised by `execute()` if the final concatenated MP4 cannot be
    measured with ffprobe after concatenation succeeds — Sprint 4
    Prompt 11.1's explicit requirement that a failed measurement must
    fail the acceptance run rather than silently falling back to the
    unmeasured, predicted duration and calling it "actual.\""""


def _measure_video_duration_seconds(path: str) -> float:
    """Measures a video file's real container duration with ffprobe —
    a small, focused acceptance-layer helper (Sprint 4 Prompt 11.1),
    not a second concatenation path and not ComfyUI-specific logic
    inside VideoBuilder. VideoBuilder itself has no reusable
    ffprobe-wrapping method to call instead (the same situation
    `core/animation_providers/comfyui.py`'s own `_verify_video()` — a
    different, ComfyUI-error-typed helper, not reused here — already
    documented); this is the acceptance-layer's own equivalent, scoped
    to exactly what this module needs: a duration, not full stream
    validation.

    Raises:
        FinalVideoMeasurementError: If ffprobe can't be run, or
            reports a failure, or its output can't be parsed as a
            duration. Never returns a fallback/guessed value.
    """
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FinalVideoMeasurementError(f"could not run ffprobe to measure {path!r}: {exc}") from exc

    if probe.returncode != 0:
        raise FinalVideoMeasurementError(
            f"ffprobe could not read the final video at {path!r}: {probe.stderr[-500:]}"
        )

    try:
        return float(probe.stdout.strip())
    except ValueError as exc:
        raise FinalVideoMeasurementError(
            f"ffprobe's output for {path!r} wasn't a parseable duration: {probe.stdout!r}"
        ) from exc


class MultiShotAcceptanceRunner:
    """Drives a capped range of real cinematic shots through the real
    rendering architecture, ending in one real concatenated clip.
    Every dependency is injected, matching `FightVideoPipeline`'s own
    pattern — `render_pipeline`, `animation_pipeline`, `video_builder`,
    and `asset_manager` default to fresh instances using the shared
    provider/router singletons, so whichever provider is currently
    configured (mock or comfyui) is what actually runs.
    """

    def __init__(
        self,
        ai_provider: AIProvider,
        render_pipeline: RenderPipeline | None = None,
        animation_pipeline: AnimationPipeline | None = None,
        video_builder: VideoBuilder | None = None,
        asset_manager: AssetManager | None = None,
    ) -> None:
        self._orchestrator = FightOrchestrator(ai_provider)
        self._render_pipeline = render_pipeline or RenderPipeline()
        self._animation_pipeline = animation_pipeline or AnimationPipeline()
        self._video_builder = video_builder or VideoBuilder()
        self._asset_manager = asset_manager or AssetManager()

    async def prepare(
        self,
        pgn: str,
        preferences: BattlePreferences,
        start_shot_index: int = 0,
        shot_count: int = _DEFAULT_MAX_SHOT_COUNT,
        fps: int | None = None,
        max_animation_seconds: float | None = None,
        allow_exceeding_default_cap: bool = False,
    ) -> MultiShotPlan:
        """Runs the real orchestration/cinematic pipeline and selects a
        capped shot range. Makes no ComfyUI or network calls — safe
        for dry-run.

        Args:
            pgn: PGN text of the game to analyze.
            preferences: Style/battle-mode preferences.
            start_shot_index: First timeline position to select, 0-indexed.
            shot_count: How many consecutive shots to select, starting
                at `start_shot_index`. Defaults to 3 — this task's own
                first paid milestone.
            fps: FPS to calculate every shot's Wan frame count against.
                Defaults to `settings.comfyui_default_fps`.
            max_animation_seconds: Acceptance-only cap applied
                uniformly to every selected shot's animation duration.
                Never changes any shot's own real cinematic duration —
                only `effective_animation_durations_seconds` on the
                returned plan.
            allow_exceeding_default_cap: Must be explicitly `True` to
                request more than `_DEFAULT_MAX_SHOT_COUNT` (3) shots —
                a deliberate friction point, not a convenience default,
                per this task's explicit cost-control requirement.

        Returns:
            A MultiShotPlan describing exactly what execute() would do.

        Raises:
            ShotCountExceedsAcceptanceCapError: If `shot_count` exceeds
                the safety cap without an explicit override.
            ShotRangeOutOfRangeError: If the requested range doesn't
                fit within the real timeline.
            ValueError: If `shot_count` isn't > 0, or
                `max_animation_seconds` is given and isn't > 0.
        """
        if shot_count <= 0:
            raise ValueError(f"shot_count must be > 0, got {shot_count!r}.")
        if shot_count > _DEFAULT_MAX_SHOT_COUNT and not allow_exceeding_default_cap:
            raise ShotCountExceedsAcceptanceCapError(
                f"shot_count={shot_count} exceeds the acceptance safety cap of "
                f"{_DEFAULT_MAX_SHOT_COUNT} — pass allow_exceeding_default_cap=True if this is an "
                "intentional, deliberate override for future development, not the first paid milestone."
            )
        if max_animation_seconds is not None and max_animation_seconds <= 0:
            raise ValueError(f"max_animation_seconds must be > 0, got {max_animation_seconds!r}.")

        settings = get_settings()

        generate_response = await self._orchestrator.generate_fight(pgn, preferences)
        timeline = generate_response.prompted_timeline
        all_shots = timeline.shots

        end_shot_index = start_shot_index + shot_count  # exclusive
        if not (0 <= start_shot_index < len(all_shots)) or end_shot_index > len(all_shots):
            raise ShotRangeOutOfRangeError(
                f"requested range [{start_shot_index}, {end_shot_index}) does not fit within this "
                f"timeline's {len(all_shots)} shots (valid indices: 0-{len(all_shots) - 1})."
            )

        selected_shots = all_shots[start_shot_index:end_shot_index]
        selected_indices = list(range(start_shot_index, end_shot_index))
        resolved_fps = fps if fps is not None else settings.comfyui_default_fps

        effective_durations = [
            min(shot.duration_seconds, max_animation_seconds) if max_animation_seconds is not None
            else shot.duration_seconds
            for shot in selected_shots
        ]
        frame_counts = [_duration_to_frame_count(duration, resolved_fps) for duration in effective_durations]
        expected_assembled_duration = sum(
            _frame_count_to_duration(count, resolved_fps) for count in frame_counts
        )

        logger.info(
            "Multi-shot acceptance: selected shots %s of %d (fps=%d, expected ~%.3fs assembled).",
            selected_indices, len(all_shots), resolved_fps, expected_assembled_duration,
        )

        return MultiShotPlan(
            fight_id=f"multi_shot_{uuid.uuid4().hex}",
            start_shot_index=start_shot_index,
            shot_count=shot_count,
            selected_shot_indices=selected_indices,
            shots=selected_shots,
            scene_continuity=timeline.scene_continuity,
            total_shots_in_timeline=len(all_shots),
            image_provider=settings.image_provider,
            animation_provider=settings.animation_provider,
            comfyui_base_url=settings.comfyui_base_url,
            comfyui_image_workflow_path=settings.comfyui_image_workflow_path,
            comfyui_animation_workflow_path=settings.comfyui_workflow_path,
            max_animation_seconds=max_animation_seconds,
            effective_animation_durations_seconds=effective_durations,
            calculated_wan_frame_counts=frame_counts,
            fps=resolved_fps,
            expected_comfyui_job_count=2 * shot_count,
            expected_assembled_duration_seconds=expected_assembled_duration,
        )

    async def execute(
        self,
        plan: MultiShotPlan,
        width: int | None = None,
        height: int | None = None,
    ) -> MultiShotAcceptanceResult:
        """Actually renders, animates, and concatenates every selected
        shot, via the real, currently-configured
        ImageRouter/AnimationRouter and the real VideoBuilder.

        Args:
            plan: A plan from `prepare()`.
            width: Output width for every animated clip. Defaults to
                `settings.comfyui_animation_default_width` (832) — the
                Wan-validated resolution. Same independent-from-FLUX
                reasoning as `SingleShotAcceptanceRunner.execute()`:
                the reference images use a separately-resolved policy
                (`settings.comfyui_image_default_width`, 1280),
                unconditionally.
            height: Output height for every animated clip. Defaults to
                `settings.comfyui_animation_default_height` (480).

        Returns:
            A MultiShotAcceptanceResult with every real local image and
            per-shot clip path, plus the one final concatenated video.

        Raises:
            ImageProviderError: If any shot's image generation fails —
                propagates unchanged from RenderPipeline; any shots
                before the failing one remain on disk (see this
                module's own docstring on why a precise per-shot
                success/failure list can't be returned as structured
                data from this method).
            AnimationProviderError: Same, for animation failures.
            FinalVideoMeasurementError: If concatenation succeeds but
                the resulting file's duration can't be measured with
                ffprobe afterward.
        """
        settings = get_settings()
        resolved_animation_width = width if width is not None else settings.comfyui_animation_default_width
        resolved_animation_height = height if height is not None else settings.comfyui_animation_default_height
        resolved_image_width = settings.comfyui_image_default_width
        resolved_image_height = settings.comfyui_image_default_height

        render_timeline = self._build_multi_shot_timeline(plan, plan.shots)
        render_output = await self._render_pipeline.render(
            render_timeline, plan.fight_id, width=resolved_image_width, height=resolved_image_height,
        )

        animation_shots = [
            shot if effective_duration == shot.duration_seconds
            else shot.model_copy(update={"duration_seconds": effective_duration})
            for shot, effective_duration in zip(plan.shots, plan.effective_animation_durations_seconds, strict=True)
        ]
        animation_timeline = self._build_multi_shot_timeline(plan, animation_shots)

        animation_output = await self._animation_pipeline.animate(
            render_output, animation_timeline,
            width=resolved_animation_width, height=resolved_animation_height, fps=plan.fps,
        )

        # AnimationPipeline.animate() already sorts its own output by
        # sequence_order (verified directly against its current source
        # before writing this module) — asserted here as an explicit
        # safety net, not silently trusted, since a silent ordering
        # mismatch would be a much more confusing failure mode
        # downstream (a shuffled cinematic sequence).
        assert len(animation_output.animated_shots) == plan.shot_count, (
            f"Expected exactly {plan.shot_count} animated shots, got "
            f"{len(animation_output.animated_shots)} — multi-shot acceptance invariant violated."
        )
        actual_order = [shot.sequence_order for shot in animation_output.animated_shots]
        assert actual_order == sorted(actual_order), (
            f"animated_shots is not in sequence_order ({actual_order}) — AnimationPipeline's own "
            "ordering guarantee was violated; the final concatenation would be misordered."
        )

        image_paths = [frame.frame_path for frame in render_output.frames]
        video_paths = [shot.video_path for shot in animation_output.animated_shots]

        total_duration = sum(shot.duration_seconds for shot in animation_output.animated_shots)
        output_path = str(self._asset_manager.fight_directory(plan.fight_id) / "multi_shot_acceptance.mp4")
        build_result = await self._video_builder.concatenate_clips(
            clip_paths=video_paths,
            output_path=output_path,
            total_duration_seconds=total_duration,
            fps=plan.fps,
            width=resolved_animation_width,
            height=resolved_animation_height,
        )

        # Sprint 4 Prompt 11.1: VideoBuilder.concatenate_clips() returns
        # the `total_duration_seconds` it was *given* as
        # `duration_seconds` — the sum of each clip's own reported
        # duration, not a measurement of the real assembled container.
        # The final MP4 genuinely exists on disk now (concatenation
        # already succeeded above) and ffprobe is already a required
        # acceptance dependency (this module's own preflight check
        # verifies it's present before generation even starts) — so
        # measure the real value directly rather than echo back a
        # prediction and call it "actual". A failed measurement fails
        # the acceptance run, per this task's explicit instruction, not
        # a silent fallback to the predicted number.
        measured_duration = _measure_video_duration_seconds(build_result.video_path)

        return MultiShotAcceptanceResult(
            plan=plan,
            image_paths=image_paths,
            video_paths=video_paths,
            final_video_path=build_result.video_path,
            final_video_duration_seconds=measured_duration,
            resolved_image_width=resolved_image_width,
            resolved_image_height=resolved_image_height,
            resolved_animation_width=resolved_animation_width,
            resolved_animation_height=resolved_animation_height,
        )

    def _build_multi_shot_timeline(self, plan: MultiShotPlan, shots: list[PromptedShot]) -> PromptedTimeline:
        """Constructs a valid, minimal PromptedTimeline containing only
        `shots` (either `plan.shots` unmodified, or duration-capped
        copies of them, in the same order — the caller decides which).
        RenderPipeline.render() and AnimationPipeline.animate() need no
        changes at all for this — both already operate purely via
        `for shot in timeline.shots`, with no assumption about a fixed
        shot count (verified directly, see module docstring)."""
        return PromptedTimeline(
            shots=shots,
            total_duration_seconds=sum(shot.duration_seconds for shot in shots),
            shot_count=len(shots),
            scene_continuity=plan.scene_continuity,
        )
