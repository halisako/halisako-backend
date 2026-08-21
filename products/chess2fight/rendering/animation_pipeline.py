"""AnimationPipeline: turns RenderPipeline's static per-shot frames
into animated per-shot clips, via AnimationRouter.

    Render Pipeline (static frames) -> Animation Pipeline -> Video Builder (concatenation)

This is the Sprint 4 Prompt 2 insertion point identified in Prompt 1's
final report: `FightVideoPipeline` used to go straight from
`RenderPipeline.render()` (one static PNG per shot) to
`VideoBuilder.build_video()` (holding each frame static for a fixed
duration). This module sits between those two: for every rendered
frame, it builds an `AnimationInstruction` from the frame and its
corresponding `Shot`, calls `AnimationRouter.generate_animation()`
(never a concrete provider directly — see `_build_instruction` and
this module's own test suite for the structural check that confirms
that), and collects the resulting per-shot animated clips in shot
order.

`AnimationPipeline` never touches ImageRouter, MockImageProvider, or
any concrete AnimationProvider. It depends only on `AnimationRouter` —
resolving the configured provider (currently always "mock", per
Sprint 4 Prompt 1's config default) is `AnimationRouter`'s job, not
this module's.

A failure animating any one shot raises `AnimationProviderError`
rather than silently producing a partial or fake-successful result —
`AnimationProviderError` is already `FightVideoPipeline`'s
`FightOrchestrator` dependency's own `Chess2FightError` subclass, so
it propagates through `api/chess2fight.py`'s existing
`except Chess2FightError -> 502` handler unchanged; no new
error-handling code was needed anywhere in the HTTP layer.

Field-mapping notes (Shot -> AnimationInstruction), per this task's
explicit "do not invent information that doesn't exist" instruction:

    Shot's rendered reference image (RenderedFrame.frame_path) -> source_image_path
    PromptedShot.image_prompt                                  -> prompt
    Shot.duration_seconds                                       -> duration_seconds
    Shot.camera_motion.value                                    -> camera_motion
    Shot.description                                            -> subject_motion
    Shot.shot_type / camera_angle / mood / sequence_order        -> metadata

`subject_motion` reuses `Shot.description` directly — Shot has no
separate "what is the subject doing" field, but `description` ("What
happens in this shot, in plain language") already *is* that
information; reusing it is not fabrication, just landing existing data
in a differently-named field. `motion_intensity` has no corresponding
field on Shot at all, so it is deliberately left at
`AnimationInstruction`'s own schema default (see core/animation_router.py)
rather than invented here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.animation_router import AnimationInstruction, AnimationRouter, get_animation_router
from core.exceptions import AnimationProviderError
from products.chess2fight.cinematic.schemas import PromptedShot, PromptedTimeline
from products.chess2fight.rendering.render_pipeline import RenderedFrame, RenderOutput


class AnimatedShot(BaseModel):
    """One shot's animated clip — the animation-pipeline equivalent of
    RenderedFrame."""

    shot_id: str = Field(..., min_length=1, description="ID of the Shot this clip animates.")
    sequence_order: int = Field(..., ge=1, description="1-indexed position of this shot within the timeline.")
    video_path: str = Field(..., min_length=1, description="Path to this shot's animated clip.")
    duration_seconds: float = Field(..., gt=0, description="Duration of this shot's animated clip, in seconds.")


class AnimationPipelineOutput(BaseModel):
    """The AnimationPipeline's complete result for one fight."""

    fight_id: str = Field(..., min_length=1, description="Which fight this animation run belongs to.")
    animated_shots: list[AnimatedShot] = Field(
        ..., min_length=1, description="Every shot's animated clip, in sequence_order."
    )
    shot_count: int = Field(..., ge=1, description="Number of animated shots — len(animated_shots).")


class AnimationPipeline:
    """Turns a RenderOutput's static frames into animated per-shot
    clips, one AnimationRouter call per shot.

    `animation_router` is injected (defaulting to the shared
    `get_animation_router()` singleton) so a test can substitute a
    router wired to a fake provider without any global state — exactly
    the same pattern `RenderPipeline`/`FightVideoPipeline` already use
    for `ImageRouter`/`VideoBuilder`.
    """

    def __init__(self, animation_router: AnimationRouter | None = None) -> None:
        """Initializes the pipeline.

        Args:
            animation_router: Where to send animation instructions.
                Defaults to the shared `get_animation_router()` singleton.
        """
        self._animation_router = animation_router or get_animation_router()

    async def animate(
        self,
        render_output: RenderOutput,
        prompted_timeline: PromptedTimeline,
        width: int = 1024,
        height: int = 1024,
        fps: int | None = None,
    ) -> AnimationPipelineOutput:
        """Animates every shot in `render_output`.

        Args:
            render_output: RenderPipeline's output — one static frame
                per shot, already saved to disk.
            prompted_timeline: The same timeline `render_output` was
                rendered from — supplies each shot's prompt, duration,
                camera, and description for instruction-building.
            width: Output width for every animated clip, in pixels —
                kept uniform across shots so VideoBuilder.concatenate_clips
                can re-encode them into one consistent final video.
            height: Output height for every animated clip, in pixels.
            fps: Output frame rate for every animated clip, if the
                caller wants one specified upfront; left unset
                otherwise (each provider's own default applies).

        Returns:
            An AnimationPipelineOutput with every shot's animated clip,
            in sequence_order.

        Raises:
            AnimationProviderError: If any shot fails to animate — a
                partial render is never silently reported as success.
        """
        shots_by_id = {shot.shot_id: shot for shot in prompted_timeline.shots}

        animated_shots = [
            await self._animate_one_frame(frame, shots_by_id[frame.metadata.shot_id], width, height, fps)
            for frame in render_output.frames
        ]
        animated_shots.sort(key=lambda shot: shot.sequence_order)

        return AnimationPipelineOutput(
            fight_id=render_output.fight_id,
            animated_shots=animated_shots,
            shot_count=len(animated_shots),
        )

    async def _animate_one_frame(
        self, frame: RenderedFrame, shot: PromptedShot, width: int, height: int, fps: int | None,
    ) -> AnimatedShot:
        instruction = self._build_instruction(frame, shot, width, height, fps)
        result = await self._animation_router.generate_animation(instruction)

        if not result.success:
            raise AnimationProviderError(
                f"Failed to animate shot {shot.shot_id!r} (frame {frame.frame_number}): {result.error_message}"
            )

        return AnimatedShot(
            shot_id=shot.shot_id,
            sequence_order=shot.sequence_order,
            video_path=result.video_path,
            duration_seconds=result.duration_seconds,
        )

    def _build_instruction(
        self, frame: RenderedFrame, shot: PromptedShot, width: int, height: int, fps: int | None,
    ) -> AnimationInstruction:
        """Maps one rendered frame + its Shot into an
        AnimationInstruction — see this module's docstring for the
        complete field-by-field mapping and why each choice was made."""
        return AnimationInstruction(
            shot_id=shot.shot_id,
            source_image_path=frame.frame_path,
            prompt=shot.image_prompt,
            duration_seconds=shot.duration_seconds,
            camera_motion=shot.camera_motion.value,
            subject_motion=shot.description,
            width=width,
            height=height,
            fps=fps,
            metadata={
                "shot_type": shot.shot_type.value,
                "camera_angle": shot.camera_angle.value,
                "mood": shot.mood,
                "sequence_order": shot.sequence_order,
            },
        )
