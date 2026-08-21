"""Tests for AnimationPipeline — the Sprint 4 Prompt 2 integration
point between RenderPipeline's static frames and VideoBuilder's final
assembly.

Uses the real pipeline (through RenderPipeline) wherever possible,
since AnimationPipeline's whole job is turning real RenderOutput +
PromptedTimeline data into AnimationRouter calls — a hand-built
fixture would prove much less than testing against what the rest of
the pipeline actually produces.
"""

import asyncio

import pytest

from core.ai_router import TemplateProvider
from core.animation_router import (
    AnimationProvider,
    AnimationProviderRegistry,
    AnimationResult,
    AnimationRouter,
    MockAnimationProvider,
)
from core.exceptions import AnimationProviderError
from core.image_router import ImageProviderRegistry, ImageRouter, MockImageProvider
from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.battle_mode_engine import generate_battle_mode_intelligence
from products.chess2fight.cinematic.prompt_generator import generate_prompts
from products.chess2fight.cinematic.scene_composer import compose_scene
from products.chess2fight.cinematic.timeline_engine import generate_shot_timeline
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.narrative_generator import NarrativeGenerator
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.render_pipeline import RenderPipeline
from products.chess2fight.schemas import BattleMode
from products.chess2fight.style_engine import generate_style_profile

SCHOLARS_MATE_PGN = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)


async def _build_render_output(tmp_path, pgn=SCHOLARS_MATE_PGN, style="anime"):
    """Runs the real pipeline through RenderPipeline, returning
    (render_output, prompted_timeline)."""
    analysis = analyze_game(pgn)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    profile = generate_style_profile(battle, combat, style)
    battle_mode = generate_battle_mode_intelligence(BattleMode.DUEL, combat, battle)
    story = await NarrativeGenerator(TemplateProvider()).generate(analysis, combat, battle, profile, battle_mode)
    timeline = generate_shot_timeline(battle, story)
    composed = compose_scene(timeline, battle, profile, battle_mode)
    prompted = generate_prompts(composed)

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path / "images")))
    image_router = ImageRouter(registry=image_registry)
    asset_manager = AssetManager(storage_root=str(tmp_path / "storage"))
    render_pipeline = RenderPipeline(image_router=image_router, asset_manager=asset_manager)

    render_output = await render_pipeline.render(prompted, "test_fight")
    return render_output, prompted


def _animation_pipeline(tmp_path) -> AnimationPipeline:
    registry = AnimationProviderRegistry()
    registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path / "animations")))
    router = AnimationRouter(registry=registry)
    return AnimationPipeline(animation_router=router)


# --- AnimationInstruction created from a Shot correctly ---------------------


def test_instruction_is_built_correctly_from_a_real_shot(tmp_path):
    render_output, prompted = asyncio.run(_build_render_output(tmp_path))
    pipeline = _animation_pipeline(tmp_path)

    frame = render_output.frames[0]
    shot = next(s for s in prompted.shots if s.shot_id == frame.metadata.shot_id)
    instruction = pipeline._build_instruction(frame, shot, width=512, height=512, fps=24)

    assert instruction.shot_id == shot.shot_id
    assert instruction.source_image_path == frame.frame_path
    assert instruction.prompt == shot.image_prompt
    assert instruction.duration_seconds == shot.duration_seconds
    assert instruction.camera_motion == shot.camera_motion.value
    assert instruction.subject_motion == shot.description
    assert instruction.width == 512
    assert instruction.height == 512
    assert instruction.fps == 24
    assert instruction.metadata["shot_type"] == shot.shot_type.value
    assert instruction.metadata["sequence_order"] == shot.sequence_order


def test_instruction_does_not_fabricate_fields_shot_lacks(tmp_path):
    """motion_intensity has no corresponding field on Shot — must stay
    at its schema default, not an invented value."""
    render_output, prompted = asyncio.run(_build_render_output(tmp_path))
    pipeline = _animation_pipeline(tmp_path)

    frame = render_output.frames[0]
    shot = next(s for s in prompted.shots if s.shot_id == frame.metadata.shot_id)
    instruction = pipeline._build_instruction(frame, shot, width=512, height=512, fps=None)

    from core.animation_router import AnimationInstruction

    default_intensity = AnimationInstruction.model_fields["motion_intensity"].default
    assert instruction.motion_intensity == default_intensity


# --- The configured AnimationRouter is invoked -------------------------------


def test_configured_router_is_actually_invoked(tmp_path, monkeypatch):
    from core import config

    monkeypatch.setattr(config.get_settings(), "animation_provider", "mock")
    render_output, prompted = asyncio.run(_build_render_output(tmp_path))
    pipeline = _animation_pipeline(tmp_path)

    output = asyncio.run(pipeline.animate(render_output, prompted, width=256, height=256))
    assert output.shot_count == render_output.frame_count
    assert all(shot.video_path for shot in output.animated_shots)


# --- MockAnimationProvider produces an actual shot MP4 -----------------------


def test_mock_provider_produces_a_real_mp4_per_shot(tmp_path):
    import os

    render_output, prompted = asyncio.run(_build_render_output(tmp_path))
    pipeline = _animation_pipeline(tmp_path)

    output = asyncio.run(pipeline.animate(render_output, prompted, width=256, height=256))
    for shot in output.animated_shots:
        assert os.path.exists(shot.video_path)
        assert os.path.getsize(shot.video_path) > 0


# --- Shot duration is preserved -----------------------------------------------


def test_each_animated_clips_duration_matches_its_shots_duration(tmp_path):
    render_output, prompted = asyncio.run(_build_render_output(tmp_path))
    pipeline = _animation_pipeline(tmp_path)

    output = asyncio.run(pipeline.animate(render_output, prompted, width=256, height=256))
    shots_by_id = {s.shot_id: s for s in prompted.shots}
    for animated in output.animated_shots:
        original_shot = shots_by_id[animated.shot_id]
        assert animated.duration_seconds == original_shot.duration_seconds


def test_different_shot_types_get_different_durations_not_a_uniform_value(tmp_path):
    """Regression guard: the whole point of this integration is that
    per-shot duration replaces one uniform frame_duration_seconds."""
    render_output, prompted = asyncio.run(_build_render_output(tmp_path))
    pipeline = _animation_pipeline(tmp_path)

    output = asyncio.run(pipeline.animate(render_output, prompted, width=256, height=256))
    durations = {shot.duration_seconds for shot in output.animated_shots}
    assert len(durations) > 1


# --- Animated shots are returned in sequence order ---------------------------


def test_animated_shots_returned_in_sequence_order(tmp_path):
    render_output, prompted = asyncio.run(_build_render_output(tmp_path))
    pipeline = _animation_pipeline(tmp_path)

    output = asyncio.run(pipeline.animate(render_output, prompted, width=256, height=256))
    orders = [shot.sequence_order for shot in output.animated_shots]
    assert orders == sorted(orders)
    assert orders == list(range(1, len(orders) + 1))


# --- Animation failure is handled correctly ----------------------------------


def test_a_failed_shot_raises_animation_provider_error_not_a_fake_success(tmp_path, monkeypatch):
    from core import config

    class _AlwaysFailsProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            return AnimationResult(
                success=False, shot_id=instruction.shot_id, provider="_AlwaysFailsProvider",
                error_message="simulated failure",
            )

    monkeypatch.setattr(config.get_settings(), "animation_provider", "failing")
    registry = AnimationProviderRegistry()
    registry.register("failing", _AlwaysFailsProvider)
    router = AnimationRouter(registry=registry)
    pipeline = AnimationPipeline(animation_router=router)

    render_output, prompted = asyncio.run(_build_render_output(tmp_path))
    with pytest.raises(AnimationProviderError, match="simulated failure"):
        asyncio.run(pipeline.animate(render_output, prompted, width=256, height=256))


def test_partial_failure_does_not_silently_skip_the_failing_shot(tmp_path, monkeypatch):
    """A provider that fails only some shots must still raise — never
    silently return a shorter-than-expected but 'successful' result."""
    from core import config

    calls = {"count": 0}

    class _FailsOnSecondCall(AnimationProvider):
        async def generate_animation(self, instruction):
            calls["count"] += 1
            if calls["count"] == 2:
                return AnimationResult(
                    success=False, shot_id=instruction.shot_id, provider="_FailsOnSecondCall",
                    error_message="shot 2 failed",
                )
            return AnimationResult(
                success=True, shot_id=instruction.shot_id, provider="_FailsOnSecondCall",
                video_path="/tmp/fake.mp4", duration_seconds=instruction.duration_seconds,
            )

    monkeypatch.setattr(config.get_settings(), "animation_provider", "intermittent")
    registry = AnimationProviderRegistry()
    registry.register("intermittent", _FailsOnSecondCall)
    router = AnimationRouter(registry=registry)
    pipeline = AnimationPipeline(animation_router=router)

    render_output, prompted = asyncio.run(_build_render_output(tmp_path))
    with pytest.raises(AnimationProviderError, match="shot 2 failed"):
        asyncio.run(pipeline.animate(render_output, prompted, width=256, height=256))


# --- No direct dependency on MockAnimationProvider inside the pipeline ------


def test_animation_pipeline_never_imports_a_concrete_provider_by_name():
    """Structural check, matching the same guarantee already verified
    for RenderPipeline/ImageRouter and MockAnimationProvider/AnimationRouter:
    this module's own source must never import or reference
    MockAnimationProvider or AnimationProviderRegistry directly."""
    import inspect

    from products.chess2fight.rendering import animation_pipeline as module

    source = inspect.getsource(module)
    assert "MockAnimationProvider" not in source
    assert "AnimationProviderRegistry" not in source


def test_animation_pipeline_depends_only_on_the_router_abstraction():
    """AnimationPipeline's constructor must accept an AnimationRouter
    — the abstraction — not a concrete provider."""
    import inspect

    from products.chess2fight.rendering.animation_pipeline import AnimationPipeline as _AP

    signature = inspect.signature(_AP.__init__)
    assert "animation_router" in signature.parameters
