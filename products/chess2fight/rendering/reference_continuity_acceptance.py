"""ReferenceContinuityAcceptanceRunner — Sprint 4 Prompt 13.

The smallest architecture-faithful reference-conditioned experiment:

    PGN -> FightOrchestrator -> PromptedTimeline -> shots [0,1,2]

    shot 0: normal FLUX text-to-image -> the fight's canonical visual
            anchor (never regenerated, never itself reference-conditioned)

    shot 1: the SAME canonical anchor + shot 1's own action/camera
            prompt -> FLUX reference-conditioned generation

    shot 2: the SAME canonical anchor (never shot 1's own output) +
            shot 2's own action/camera prompt -> FLUX reference-
            conditioned generation

    -> 3 keyframes -> Wan x3 -> VideoBuilder.concatenate_clips() -> one final MP4

Explicitly NOT a chain (shot0 -> shot1 -> shot2): both shot 1 and shot
2 independently reference shot 0's own output. Verified directly in
`execute()` below, not just asserted in this docstring — every
reference-conditioned call passes `anchor.image_path`, a value fixed
before either call happens and never reassigned.

Reuses `MultiShotAcceptanceRunner.prepare()` internally (composition,
not inheritance) for shot selection, duration capping, and seed
resolution — completely unchanged, zero risk to that already
GPU-proven code path. `execute()` here is new: shot 0 goes through the
real `RenderPipeline.render()` (a genuine single-shot timeline, same
call every other acceptance path already uses); shots 1/2 go directly
through `ComfyUIImageProvider.generate_reference_conditioned_image()`
(a capability that exists only on the concrete ComfyUI provider — see
that module's own docstring on why this doesn't violate
RenderPipeline's provider-agnostic contract: RenderPipeline itself is
never touched by this module at all).

Preserves every Prompt 11/12/12.1 safety guarantee this experiment
doesn't change: preflight, output writability, ffprobe-measured final
duration, the 2-second Wan cap, no automatic retries, partial artifacts
preserved on failure, and the same planned-vs-actual FLUX seed evidence
check (extended here to cover the anchor and both reference-conditioned
shots, not just independently-generated ones).
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from core.exceptions import ImageProviderError
from core.image_providers.comfyui import ComfyUIImageProvider
from core.image_providers.comfyui import _derive_seed as _derive_flux_seed
from products.chess2fight.cinematic.prompt_generator import compose_reference_edit_prompt
from products.chess2fight.cinematic.schemas import ArenaContinuity, FighterAppearance, PromptedShot, PromptedTimeline
from products.chess2fight.rendering.asset_manager import AssetManager, FrameMetadata, RenderManifest
from products.chess2fight.rendering.multi_shot_acceptance import (
    MultiShotAcceptanceRunner,
    MultiShotPlan,
    SeedEvidenceMismatchError,
    _measure_video_duration_seconds,
)
from products.chess2fight.rendering.render_pipeline import RenderedFrame, RenderOutput, RenderPipeline
from products.chess2fight.rendering.video_builder import VideoBuilder

logger = logging.getLogger(__name__)


class ReferenceAnchorInvalidError(RuntimeError):
    """Raised by `execute()` if the canonical anchor (shot 0's own T2I
    output) doesn't exist or isn't a valid, readable image before any
    reference-conditioned generation is attempted for shots 1/2 —
    Sprint 4 Prompt 13's explicit requirement: never fall back to
    independent T2I on this failure, since that would invalidate the
    whole point of the experiment."""


class FightVisualAnchor(BaseModel):
    """The fight-level canonical visual reference: shot 0's own real
    FLUX output, reused unchanged (never regenerated, never itself
    edited or chained) as the reference image for every subsequent
    reference-conditioned shot in this experiment.

    Provider-neutral by design (Sprint 4 Prompt 13's own explicit
    requirement) — every field here is something any image provider
    could report (a path, a seed, dimensions, provenance), never a
    ComfyUI-specific value (no node IDs, no ComfyUI prompt_id, no
    workflow filename).
    """

    model_config = ConfigDict(frozen=True)

    source_shot_index: int = Field(..., ge=0, description="Timeline index of the shot that produced this anchor.")
    image_path: str = Field(..., min_length=1, description="Local path to the anchor image.")
    white_fighter: FighterAppearance = Field(..., description="The white fighter's continuity descriptors.")
    black_fighter: FighterAppearance = Field(..., description="The black fighter's continuity descriptors.")
    arena: ArenaContinuity = Field(..., description="The arena/environment continuity descriptors.")
    cinematic_art_style: str = Field(..., min_length=1, description="The fight's global visual style descriptor.")
    generation_seed: int = Field(..., description="The actual seed used to generate this anchor.")
    provider: str = Field(..., min_length=1, description="Which image provider generated this anchor.")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    provenance: str = Field(
        ..., min_length=1, description="How this anchor was itself produced — 't2i' for the only "
        "provenance this experiment currently generates (a normal, independent FLUX text-to-image call)."
    )


class ReferenceContinuityResult(BaseModel):
    """What actually happened, after `execute()` produced the anchor,
    both reference-conditioned shots, animated all three, and
    concatenated the result."""

    plan: MultiShotPlan = Field(..., description="The plan this result was executed from.")
    anchor: FightVisualAnchor = Field(..., description="The fight's canonical visual anchor.")
    generation_modes: list[str] = Field(
        ..., min_length=1, description="Per selected shot, in timeline order: 't2i' for the anchor shot, "
        "'reference_conditioned' for every other selected shot.",
    )
    image_paths: list[str] = Field(..., min_length=1, description="Each shot's keyframe path, in timeline order.")
    reference_anchor_paths: list[str | None] = Field(
        ..., min_length=1, description="Per shot, in timeline order: the anchor path it was reference-"
        "conditioned against, or None for the anchor shot itself (which has no reference of its own).",
    )
    video_paths: list[str] = Field(..., min_length=1, description="Each shot's animated clip path, in timeline order.")
    final_video_path: str = Field(..., min_length=1, description="Path to the one final concatenated MP4.")
    final_video_duration_seconds: float = Field(
        ..., gt=0, description="Measured directly with ffprobe after concatenation — same guarantee as "
        "Sprint 4 Prompt 11.1's MultiShotAcceptanceResult.final_video_duration_seconds."
    )
    resolved_image_width: int = Field(..., gt=0)
    resolved_image_height: int = Field(..., gt=0)
    resolved_animation_width: int = Field(..., gt=0)
    resolved_animation_height: int = Field(..., gt=0)
    actual_flux_seeds: list[int] = Field(
        ..., min_length=1, description="Each shot's actual, provider-reported FLUX seed, in timeline order — "
        "the anchor's own generation seed, plus each reference-conditioned shot's own.",
    )


class ReferenceContinuityAcceptanceRunner:
    """Drives the reference-conditioned three-shot experiment.

    `reference_provider` — a real `ComfyUIImageProvider` configured for
    reference-conditioned generation (see that class's own
    `generate_reference_conditioned_image`) — is required at
    `execute()` time, not construction time, matching this module's
    own fail-fast philosophy: a mock run has no meaningful
    reference-conditioned behavior to exercise, so requiring it only
    when actually needed keeps `prepare()` (and therefore dry-run)
    completely provider-agnostic and network-call-free.
    """

    def __init__(
        self,
        ai_provider,
        render_pipeline: RenderPipeline | None = None,
        animation_pipeline=None,
        video_builder: VideoBuilder | None = None,
        asset_manager: AssetManager | None = None,
    ) -> None:
        self._multi_shot_runner = MultiShotAcceptanceRunner(
            ai_provider, render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
            video_builder=video_builder, asset_manager=asset_manager,
        )
        self._render_pipeline = render_pipeline or RenderPipeline()
        if animation_pipeline is None:
            from products.chess2fight.rendering.animation_pipeline import AnimationPipeline

            animation_pipeline = AnimationPipeline()
        self._animation_pipeline = animation_pipeline
        self._video_builder = video_builder or VideoBuilder()
        self._asset_manager = asset_manager or AssetManager()

    async def prepare(
        self,
        pgn: str,
        preferences,
        start_shot_index: int = 0,
        shot_count: int = 3,
        fps: int | None = None,
        max_animation_seconds: float | None = None,
        visual_seed_policy=None,
        allow_exceeding_default_cap: bool = False,
    ) -> MultiShotPlan:
        """Shot selection and duration capping are identical to
        `MultiShotAcceptanceRunner.prepare()` — delegated directly
        (composition), not reimplemented. FLUX seed resolution starts
        from that same delegated call, then this method corrects it
        for every shot after the anchor (see the live-evidence-driven
        correction below) — it is deliberately NOT identical for
        reference-conditioned shots, unlike an earlier version of this
        docstring claimed.

        `visual_seed_policy` defaults to SHARED here (unlike
        `MultiShotAcceptanceRunner.prepare()`'s own DEFAULT), since a
        reference-conditioned experiment's point is testing whether
        conditioning succeeds where a shared seed alone didn't — an
        uncapped/per-prompt-hash seed makes little sense as this
        module's own default, though it remains overridable like
        everything else here.

        Live-evidence-driven correction: `plan.resolved_flux_seeds` as
        returned by `_multi_shot_runner.prepare()` plans every shot's
        FLUX seed from `shot.image_prompt` — correct for shot 0 (the
        T2I anchor, which genuinely does submit that exact prompt),
        but wrong for every other shot, since `execute()` below
        actually submits `compose_reference_edit_prompt(shot)` for
        those — a different string, hashing to a different seed under
        SHARED/DERIVED policy. Confirmed directly against real GPU
        history: `derive_shot_seed(base_seed, image_prompt)` for the
        sample fight's shots 1/2 does not match either shot's actual,
        provider-reported seed; `derive_shot_seed(base_seed,
        compose_reference_edit_prompt(shot))` matches exactly. This
        method now re-plans every shot after index 0 using the same
        prompt `execute()` will actually submit for it — the returned
        plan's `resolved_flux_seeds` therefore now agrees with reality
        for the anchor AND every reference-conditioned shot, closing
        the gap `execute()`'s own seed-evidence check exists to catch.
        An earlier version of this method's own docstring claimed seed
        resolution "is exactly the same regardless of whether shots
        1/2 end up independently generated or reference-conditioned"
        — that claim was wrong; corrected here alongside the code.
        """
        from products.chess2fight.rendering.visual_continuity import VisualSeedPolicy, build_seed_override

        resolved_policy = visual_seed_policy if visual_seed_policy is not None else VisualSeedPolicy.SHARED
        plan = await self._multi_shot_runner.prepare(
            pgn, preferences, start_shot_index=start_shot_index, shot_count=shot_count, fps=fps,
            max_animation_seconds=max_animation_seconds, visual_seed_policy=resolved_policy,
            allow_exceeding_default_cap=allow_exceeding_default_cap,
        )

        flux_seed_override = (
            build_seed_override(VisualSeedPolicy(plan.visual_seed_policy), plan.fight_base_visual_seed)
            if plan.fight_base_visual_seed is not None else None
        )
        corrected_flux_seeds = list(plan.resolved_flux_seeds)
        for i, shot in enumerate(plan.shots):
            if i == 0:
                continue  # the anchor: T2I, shot.image_prompt is genuinely what gets submitted
            reference_prompt = compose_reference_edit_prompt(shot)
            corrected_flux_seeds[i] = (
                flux_seed_override(reference_prompt) if flux_seed_override is not None
                else _derive_flux_seed(reference_prompt)
            )

        return plan.model_copy(update={"resolved_flux_seeds": corrected_flux_seeds})

    async def execute(
        self,
        plan: MultiShotPlan,
        reference_provider: ComfyUIImageProvider,
        width: int | None = None,
        height: int | None = None,
    ) -> ReferenceContinuityResult:
        """Actually generates the anchor, both reference-conditioned
        shots, animates all three, and concatenates the result.

        Args:
            plan: A plan from `prepare()`.
            reference_provider: A real `ComfyUIImageProvider`, required
                — see this class's own docstring on why this isn't a
                constructor argument.
            width: Output width for every animated clip. Same
                independent-from-FLUX reasoning as
                `MultiShotAcceptanceRunner.execute()`.
            height: Output height for every animated clip.

        Raises:
            ReferenceAnchorInvalidError: If the anchor isn't a valid,
                readable image with the expected dimensions, before
                any reference-conditioned call is attempted (Sprint 4
                Prompt 13.1 added the dimension check).
            ImageProviderError: If any generation fails — including a
                reference-conditioned call; never falls back to
                independent T2I on this failure.
            SeedEvidenceMismatchError: If any shot's actual FLUX seed
                disagrees with what `prepare()` planned — raised
                immediately after that specific shot's own generation
                completes, before any further paid FLUX or Wan job is
                submitted (a live-evidence-driven hardening: ComfyUI
                history from a real failed run showed all 3 FLUX jobs
                had already executed — including a second reference
                shot submitted AFTER the first one's seed had already
                mismatched — before this error was ever raised, because
                the check previously ran once, as a batch, only after
                every image generation had already completed). The
                batch check at the end is retained too, as defense in
                depth, but is no longer what does the real work.
            FinalVideoMeasurementError: If the final concatenated MP4
                can't be measured with ffprobe.
        """
        from core.config import get_settings

        settings = get_settings()
        resolved_animation_width = width if width is not None else settings.comfyui_animation_default_width
        resolved_animation_height = height if height is not None else settings.comfyui_animation_default_height
        resolved_image_width = settings.comfyui_image_default_width
        resolved_image_height = settings.comfyui_image_default_height

        def _check_seed_immediately(index: int, actual_seed: int, mode: str) -> None:
            """Sprint 4 hotfix: called right after each individual
            paid FLUX generation — anchor or reference-conditioned —
            with that shot's own actual, provider-reported seed. Never
            batches multiple shots' checks together; raises before the
            calling loop can proceed to the next shot's own paid call."""
            planned_seed = plan.resolved_flux_seeds[index]
            if planned_seed != actual_seed:
                raise SeedEvidenceMismatchError(
                    f"Shot at timeline index {plan.selected_shot_indices[index]} "
                    f"(shot_id={plan.shots[index].shot_id!r}, mode={mode!r}): planned FLUX seed "
                    f"{planned_seed} does not match the actual provider-reported seed {actual_seed} — "
                    "stopping immediately, before any further paid FLUX or Wan job."
                )

        # --- Shot 0: normal T2I, via the real, unchanged RenderPipeline ---
        anchor_shot = plan.shots[0]
        anchor_timeline = self._build_single_shot_timeline(plan, anchor_shot)
        anchor_render_output = await self._render_pipeline.render(
            anchor_timeline, plan.fight_id, width=resolved_image_width, height=resolved_image_height,
        )
        anchor_frame = anchor_render_output.frames[0]

        anchor = FightVisualAnchor(
            source_shot_index=plan.selected_shot_indices[0],
            image_path=anchor_frame.frame_path,
            white_fighter=anchor_shot.scene.white_fighter,
            black_fighter=anchor_shot.scene.black_fighter,
            arena=anchor_shot.scene.arena,
            cinematic_art_style=anchor_shot.scene.cinematic_art_style,
            generation_seed=anchor_frame.metadata.generation_seed,
            provider=plan.image_provider,
            width=resolved_image_width,
            height=resolved_image_height,
            provenance="t2i",
        )

        # Fail loudly before any reference-conditioned (paid) call —
        # never fall back to T2I silently on this failure. Runs before
        # the seed check below: an anchor with the wrong dimensions is
        # a more fundamental problem (nothing downstream would be
        # trustworthy regardless of whether its own seed happened to
        # match), and this ordering preserves this class's own
        # pre-existing, already-tested ReferenceAnchorInvalidError
        # behavior for that case.
        self._validate_anchor(anchor, resolved_image_width, resolved_image_height)
        logger.info("Reference continuity: anchor established from shot %d at %s.", anchor.source_shot_index, anchor.image_path)

        # Immediate check #1 (anchor) — before any reference-conditioned
        # (paid) call is even attempted.
        _check_seed_immediately(0, anchor.generation_seed, "t2i")

        # --- Shots 1/2: reference-conditioned, BOTH against the SAME anchor —
        # never chained shot0 -> shot1 -> shot2. anchor.image_path is fixed
        # above and never reassigned; both calls below pass the identical value.
        rendered_frames: list[RenderedFrame] = [anchor_frame]
        generation_modes = ["t2i"]
        reference_anchor_paths: list[str | None] = [None]

        for index, shot in enumerate(plan.shots[1:], start=1):
            reference_prompt = compose_reference_edit_prompt(shot)
            try:
                result = await reference_provider.generate_reference_conditioned_image(
                    reference_prompt, anchor.image_path, width=resolved_image_width, height=resolved_image_height,
                )
            except ImageProviderError:
                raise  # no silent fallback to independent T2I — see this class's own docstring

            actual_seed = result.metadata.get("seed", anchor.generation_seed)
            # Immediate check #2/#3 (each reference-conditioned shot) —
            # BEFORE this shot's own frame is even saved, and therefore
            # before the loop can reach the NEXT shot's own paid call.
            _check_seed_immediately(index, actual_seed, "reference_conditioned")

            saved_path = self._asset_manager.save_frame(plan.fight_id, shot.sequence_order, result.image_path)
            frame_metadata = FrameMetadata(
                frame_number=shot.sequence_order,
                prompt=reference_prompt,
                camera_angle=shot.camera_angle.value,
                camera_motion=shot.camera_motion.value,
                shot_id=shot.shot_id,
                shot_type=shot.shot_type.value,
                source_moves=list(shot.source_moves),
                timestamp=anchor_frame.metadata.timestamp,
                generation_seed=actual_seed,
            )
            rendered_frames.append(RenderedFrame(frame_number=shot.sequence_order, frame_path=str(saved_path), metadata=frame_metadata))
            generation_modes.append("reference_conditioned")
            reference_anchor_paths.append(anchor.image_path)

        # Sprint 4 Prompt 13.1 (Optional Evidence Cleanup) — refresh
        # metadata.json to truthfully represent all three final
        # keyframes. RenderPipeline.render() already wrote it after
        # shot 0 alone (the anchor timeline only ever had one shot);
        # without this refresh, metadata.json would permanently
        # under-report this run (1 frame) while the acceptance
        # manifest correctly reports all 3 — using RenderManifest and
        # AssetManager.write_manifest() exactly as RenderPipeline
        # itself already does, not a new mechanism.
        self._asset_manager.write_manifest(
            plan.fight_id,
            RenderManifest(fight_id=plan.fight_id, frame_count=len(rendered_frames), frames=[f.metadata for f in rendered_frames]),
        )

        combined_render_output = RenderOutput(
            fight_id=plan.fight_id, frames=rendered_frames, frame_count=len(rendered_frames),
            output_dir=anchor_render_output.output_dir, manifest_path=anchor_render_output.manifest_path,
        )

        # --- Seed evidence, batch form: defense in depth only. Every shot
        # already passed its own immediate check above before its frame
        # was ever appended to rendered_frames — this can only ever
        # re-confirm what's already true, never catch something the
        # per-shot checks missed, since it draws from the exact same
        # plan.resolved_flux_seeds and the exact same actual seeds those
        # checks already validated one at a time. ---
        actual_flux_seeds = [frame.metadata.generation_seed for frame in rendered_frames]
        for i, (planned, actual) in enumerate(zip(plan.resolved_flux_seeds, actual_flux_seeds, strict=True)):
            if planned != actual:
                raise SeedEvidenceMismatchError(
                    f"Shot at timeline index {plan.selected_shot_indices[i]} "
                    f"(shot_id={plan.shots[i].shot_id!r}, mode={generation_modes[i]!r}): planned FLUX seed "
                    f"{planned} does not match the actual provider-reported seed {actual}."
                )

        # --- Animate all three keyframes and concatenate — same pattern as MultiShotAcceptanceRunner ---
        animation_shots = [
            shot if effective_duration == shot.duration_seconds
            else shot.model_copy(update={"duration_seconds": effective_duration})
            for shot, effective_duration in zip(plan.shots, plan.effective_animation_durations_seconds, strict=True)
        ]
        animation_timeline = self._build_multi_shot_timeline(plan, animation_shots)
        animation_output = await self._animation_pipeline.animate(
            combined_render_output, animation_timeline,
            width=resolved_animation_width, height=resolved_animation_height, fps=plan.fps,
        )

        image_paths = [frame.frame_path for frame in rendered_frames]
        video_paths = [shot.video_path for shot in animation_output.animated_shots]

        total_duration = sum(shot.duration_seconds for shot in animation_output.animated_shots)
        output_path = str(self._asset_manager.fight_directory(plan.fight_id) / "reference_continuity_acceptance.mp4")
        build_result = await self._video_builder.concatenate_clips(
            clip_paths=video_paths, output_path=output_path, total_duration_seconds=total_duration,
            fps=plan.fps, width=resolved_animation_width, height=resolved_animation_height,
        )
        measured_duration = _measure_video_duration_seconds(build_result.video_path)

        return ReferenceContinuityResult(
            plan=plan, anchor=anchor, generation_modes=generation_modes, image_paths=image_paths,
            reference_anchor_paths=reference_anchor_paths, video_paths=video_paths,
            final_video_path=build_result.video_path, final_video_duration_seconds=measured_duration,
            resolved_image_width=resolved_image_width, resolved_image_height=resolved_image_height,
            resolved_animation_width=resolved_animation_width, resolved_animation_height=resolved_animation_height,
            actual_flux_seeds=actual_flux_seeds,
        )

    def _validate_anchor(self, anchor: FightVisualAnchor, expected_width: int, expected_height: int) -> None:
        """Validates the anchor before any reference-conditioned
        (paid) call is attempted.

        Sprint 4 Prompt 13.1: also validates dimensions now, not just
        existence/decodability. The reference workflow retains explicit
        1280x704 output nodes rather than reproducing every UI
        convenience node (e.g. GetImageSize) the official workflow
        has — acceptable specifically because this check exists: if
        the anchor's actual dimensions ever drifted from what the
        reference workflow expects, silently proceeding would either
        fail confusingly deep inside ComfyUI or (worse) succeed with a
        silently mismatched/resized reference. Never resizes — fails
        instead, per this task's explicit "no fallback, no resizing"
        instruction.

        Raises:
            ReferenceAnchorInvalidError: If the anchor is missing,
                undecodable, or its dimensions don't match
                `expected_width`/`expected_height` exactly.
        """
        if not Path(anchor.image_path).exists():
            raise ReferenceAnchorInvalidError(f"Canonical anchor image not found: {anchor.image_path!r}.")
        try:
            with Image.open(anchor.image_path) as image:
                image.verify()
        except Exception as exc:
            raise ReferenceAnchorInvalidError(f"Canonical anchor is not a valid, readable image: {exc}") from exc

        # A second, separate open — Image.verify() invalidates the
        # image object for further use (Pillow's own documented
        # behavior; the same two-open pattern is already established
        # in ComfyUIImageProvider._verify_image() for this reason).
        with Image.open(anchor.image_path) as image:
            actual_width, actual_height = image.size
        if (actual_width, actual_height) != (expected_width, expected_height):
            raise ReferenceAnchorInvalidError(
                f"Canonical anchor is {actual_width}x{actual_height}, expected {expected_width}x{expected_height} "
                "— refusing to submit a reference-conditioned job against a mismatched anchor. No resizing, "
                "no fallback: fix the anchor's own generation resolution instead."
            )

    def _build_single_shot_timeline(self, plan: MultiShotPlan, shot: PromptedShot) -> PromptedTimeline:
        return PromptedTimeline(
            shots=[shot], total_duration_seconds=shot.duration_seconds, shot_count=1,
            scene_continuity=plan.scene_continuity,
        )

    def _build_multi_shot_timeline(self, plan: MultiShotPlan, shots: list[PromptedShot]) -> PromptedTimeline:
        return PromptedTimeline(
            shots=shots, total_duration_seconds=sum(shot.duration_seconds for shot in shots),
            shot_count=len(shots), scene_continuity=plan.scene_continuity,
        )
