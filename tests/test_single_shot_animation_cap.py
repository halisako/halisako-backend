"""Tests for the Sprint 4 Prompt 7.1 acceptance-only animation duration
cap (`max_animation_seconds`) on SingleShotAcceptanceRunner.

All mock providers — no GPU/ComfyUI/ffmpeg-independent behavior is
asserted separately from anything that genuinely needs a real video
file; see the FFMPEG_AVAILABLE guard below for the split.
"""

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from core.ai_router import TemplateProvider
from core.animation_providers.comfyui import _duration_to_frame_count
from core.animation_router import (
    AnimationInstruction,
    AnimationProvider,
    AnimationProviderRegistry,
    AnimationResult,
    AnimationRouter,
    MockAnimationProvider,
)
from core.image_router import ImageProvider, ImageProviderRegistry, ImageRouter, MockImageProvider
from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.render_pipeline import RenderPipeline
from products.chess2fight.rendering.single_shot_acceptance import SingleShotAcceptanceRunner
from products.chess2fight.schemas import BattleMode, BattlePreferences

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

SAMPLE_PGN_PATH = Path("products/chess2fight/rendering/fixtures/sample_acceptance_game.pgn")


def _sample_pgn() -> str:
    return SAMPLE_PGN_PATH.read_text(encoding="utf-8")


def _runner(tmp_path, animation_provider=None) -> SingleShotAcceptanceRunner:
    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path / "images")))
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry),
        asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )

    anim_registry = AnimationProviderRegistry()
    anim_registry.register(
        "mock", animation_provider or (lambda: MockAnimationProvider(output_dir=str(tmp_path / "animations")))
    )
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    return SingleShotAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
    )


def _preferences() -> BattlePreferences:
    return BattlePreferences(battle_mode=BattleMode.DUEL, style="anime")


# --- 1. No cap preserves current (Prompt 7) behavior -------------------------


def test_no_cap_effective_duration_equals_real_shot_duration(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=None))
    assert plan.effective_animation_duration_seconds == plan.shot.duration_seconds
    assert plan.max_animation_seconds is None


def test_no_cap_animation_timeline_is_the_same_object_as_render_timeline(tmp_path):
    """Not just numerically equal — structurally the identical object,
    guaranteeing this path is byte-for-byte the same code path Prompt
    7 used, not a new path that happens to produce the same numbers."""
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0))

    captured = {}
    original_animate = runner._animation_pipeline.animate

    async def _spy_animate(render_output, prompted_timeline, **kwargs):
        captured["timeline"] = prompted_timeline
        return await original_animate(render_output, prompted_timeline, **kwargs)

    runner._animation_pipeline.animate = _spy_animate
    asyncio.run(runner.execute(plan, width=256, height=256))
    assert captured["timeline"].shots[0] is plan.shot


# --- 2/6. 7.75s shot + cap 2.0 -> effective duration 2.0, 49 frames @ 24fps --


def test_capped_effective_duration_and_frame_count(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=2.0, fps=24))
    assert plan.shot.duration_seconds == 7.75
    assert plan.effective_animation_duration_seconds == 2.0
    assert plan.calculated_wan_frame_count == 49
    assert plan.calculated_wan_frame_count == _duration_to_frame_count(2.0, 24)


# --- 3. Original PromptedShot.duration_seconds remains 7.75 -----------------


def test_original_shot_duration_unaffected_by_cap(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=2.0))
    assert plan.shot.duration_seconds == 7.75  # never touched, regardless of the cap


def test_original_shot_duration_unaffected_after_execute(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=2.0))
    asyncio.run(runner.execute(plan, width=256, height=256))
    assert plan.shot.duration_seconds == 7.75  # still unmutated after actually executing


# --- 4. Original timeline remains unchanged -----------------------------------


def test_full_timeline_from_orchestrator_is_never_touched(tmp_path):
    """Confirms the cap only ever affects the single-shot acceptance
    plan's own copy — re-running the real orchestrator independently
    still produces the same, unaffected full timeline."""
    runner = _runner(tmp_path)
    asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=2.0))

    from core.ai_router import TemplateProvider as _TP
    from products.chess2fight.orchestrator import FightOrchestrator

    async def _full():
        return await FightOrchestrator(_TP()).generate_fight(_sample_pgn(), _preferences())

    full_response = asyncio.run(_full())
    assert full_response.prompted_timeline.shots[0].duration_seconds == 7.75


# --- 5. AnimationInstruction receives the effective (capped) duration -------


def test_animation_instruction_receives_capped_duration_not_original(tmp_path):
    received_instructions = []

    class _SpyAnimationProvider(AnimationProvider):
        async def generate_animation(self, instruction: AnimationInstruction) -> AnimationResult:
            received_instructions.append(instruction)
            return AnimationResult(
                success=True, shot_id=instruction.shot_id, provider="_SpyAnimationProvider",
                video_path=str(tmp_path / "spy.mp4"), duration_seconds=instruction.duration_seconds,
            )

    (tmp_path / "spy.mp4").write_bytes(b"fake")

    runner = _runner(tmp_path, animation_provider=lambda: _SpyAnimationProvider())
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=2.0))
    asyncio.run(runner.execute(plan, width=256, height=256))

    assert len(received_instructions) == 1
    assert received_instructions[0].duration_seconds == 2.0
    assert received_instructions[0].duration_seconds != plan.shot.duration_seconds


# --- 7. Cap greater than real duration uses real duration --------------------


def test_cap_larger_than_real_duration_uses_real_duration(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=100.0))
    assert plan.effective_animation_duration_seconds == plan.shot.duration_seconds == 7.75


def test_cap_exactly_equal_to_real_duration_is_a_no_op(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=7.75))
    assert plan.effective_animation_duration_seconds == 7.75


# --- 8/9. Zero/negative cap rejected clearly ---------------------------------


def test_zero_cap_rejected_clearly(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(ValueError, match="must be > 0"):
        asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=0))


def test_negative_cap_rejected_clearly(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(ValueError, match="must be > 0"):
        asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=-3.0))


# --- 10. Dry run (prepare only) makes no provider HTTP calls ----------------


def test_prepare_with_cap_still_makes_no_provider_calls(tmp_path):
    class _ExplodingImageProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            raise AssertionError("prepare() must never call the image provider, cap or not")

    class _ExplodingAnimationProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            raise AssertionError("prepare() must never call the animation provider, cap or not")

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _ExplodingImageProvider())
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry),
        asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: _ExplodingAnimationProvider())
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    runner = SingleShotAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
    )
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=2.0))
    assert plan.effective_animation_duration_seconds == 2.0


# --- 11. CLI dry-run prints original and effective duration -----------------


def test_cli_dry_run_prints_both_original_and_effective_duration():
    """Sprint 4 Prompt 8: the frame count here (17 @ 8fps) reflects
    the newer live-validated default that superseded Prompt 4's
    49 @ 24fps — the point being tested (both durations are printed
    clearly) is unaffected by which default is currently active."""
    result = subprocess.run(
        [sys.executable, "scripts/render_single_shot.py", "--sample", "--shot-index", "0",
         "--max-animation-seconds", "2", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "original shot duration:       7.75s" in result.stdout
    assert "effective animation duration: 2.00s" in result.stdout
    assert "wan frame count:     17 frames @ 8fps" in result.stdout


def test_cli_dry_run_without_cap_states_no_cap_clearly():
    result = subprocess.run(
        [sys.executable, "scripts/render_single_shot.py", "--sample", "--shot-index", "0", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "no cap requested" in result.stdout


# --- 12/13. Normal full pipeline and /render remain unchanged ---------------


def test_pipeline_module_still_has_no_knowledge_of_the_animation_cap():
    import inspect

    from products.chess2fight.rendering import pipeline as module

    source = inspect.getsource(module)
    assert "max_animation_seconds" not in source
    assert "SingleShotAcceptanceRunner" not in source


def test_api_module_still_has_no_knowledge_of_the_animation_cap():
    import inspect

    from api import chess2fight as module

    source = inspect.getsource(module)
    assert "max_animation_seconds" not in source


# --- 14. Single-shot execution with no cap behaves exactly as Prompt 7 did --


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="requires ffmpeg, not available on this machine")
def test_no_cap_execution_produces_a_full_duration_clip(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0))
    result = asyncio.run(runner.execute(plan, width=256, height=256))
    assert result.video_duration_seconds == pytest.approx(7.75, abs=0.1)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="requires ffmpeg, not available on this machine")
def test_capped_execution_produces_a_2_second_clip(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, width=256, height=256))
    assert result.video_duration_seconds == pytest.approx(2.0, abs=0.1)


# --- Task 4: representative durations, all must be valid 4n+1 ---------------


@pytest.mark.parametrize("duration", [1.0, 2.0, 3.0, 0.5, 100.0])
def test_representative_durations_all_produce_valid_4n_plus_1_frame_counts(duration):
    frame_count = _duration_to_frame_count(duration, 24)
    assert (frame_count - 1) % 4 == 0


def test_two_seconds_at_24fps_is_exactly_49_frames():
    assert _duration_to_frame_count(2.0, 24) == 49
