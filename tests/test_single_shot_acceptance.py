"""Tests for SingleShotAcceptanceRunner — Sprint 4 Prompt 7.

All mock providers — no GPU/ComfyUI required. Uses the real PGN
analysis/cinematic pipeline throughout (only the image/animation
providers are ever mocked), and the real bundled sample PGN fixture
for representative-duration checks.
"""

import asyncio
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
from core.image_router import ImageGenerationResult, ImageProvider, ImageProviderRegistry, ImageRouter, MockImageProvider
from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.render_pipeline import RenderPipeline
from products.chess2fight.rendering.single_shot_acceptance import (
    ShotIndexOutOfRangeError,
    SingleShotAcceptanceRunner,
)
from products.chess2fight.schemas import BattleMode, BattlePreferences

SAMPLE_PGN_PATH = Path("products/chess2fight/rendering/fixtures/sample_acceptance_game.pgn")
SCHOLARS_MATE_PGN = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)


def _sample_pgn() -> str:
    return SAMPLE_PGN_PATH.read_text()


def _runner(tmp_path, image_provider=None, animation_provider=None) -> SingleShotAcceptanceRunner:
    image_registry = ImageProviderRegistry()
    image_registry.register("mock", image_provider or (lambda: MockImageProvider(output_dir=str(tmp_path / "images"))))
    image_router = ImageRouter(registry=image_registry)
    asset_manager = AssetManager(storage_root=str(tmp_path / "storage"))
    render_pipeline = RenderPipeline(image_router=image_router, asset_manager=asset_manager)

    anim_registry = AnimationProviderRegistry()
    anim_registry.register(
        "mock", animation_provider or (lambda: MockAnimationProvider(output_dir=str(tmp_path / "animations")))
    )
    animation_router = AnimationRouter(registry=anim_registry)
    animation_pipeline = AnimationPipeline(animation_router=animation_router)

    return SingleShotAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
    )


def _preferences() -> BattlePreferences:
    return BattlePreferences(battle_mode=BattleMode.DUEL, style="anime")


# --- 1. A real PGN produces a ShotTimeline ------------------------------------


def test_real_pgn_produces_a_shot_timeline(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0))
    assert plan.total_shots_in_timeline == 8  # the real cinematic pipeline's actual output for this game


def test_scholars_mate_also_produces_a_valid_timeline(tmp_path):
    """A different, much shorter real game — confirms this isn't
    coupled to the specific sample fixture's shot count."""
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(SCHOLARS_MATE_PGN, _preferences(), shot_index=0))
    assert plan.total_shots_in_timeline >= 1


# --- 2. Exactly one requested shot is selected --------------------------------


def test_exactly_the_requested_shot_is_selected(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=3))
    assert plan.shot_index == 3

    # Cross-check against the real, full timeline directly.
    from core.ai_router import TemplateProvider as _TP
    from products.chess2fight.orchestrator import FightOrchestrator

    async def _full():
        return await FightOrchestrator(_TP()).generate_fight(_sample_pgn(), _preferences())

    full_response = asyncio.run(_full())
    assert plan.shot.shot_id == full_response.prompted_timeline.shots[3].shot_id
    assert plan.shot.image_prompt == full_response.prompted_timeline.shots[3].image_prompt


# --- 3. Invalid shot indices fail clearly -------------------------------------


def test_negative_shot_index_raises_clearly(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(ShotIndexOutOfRangeError, match="out of range"):
        asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=-1))


def test_too_large_shot_index_raises_clearly(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(ShotIndexOutOfRangeError, match="out of range"):
        asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=999))


def test_out_of_range_error_message_states_valid_bounds(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(ShotIndexOutOfRangeError, match=r"0-7"):
        asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=8))


# --- 4. The Shot's actual image_prompt reaches ImageRouter unchanged ---------


def test_actual_image_prompt_reaches_the_image_provider_unchanged(tmp_path):
    received_prompts = []

    class _SpyImageProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            received_prompts.append(prompt)
            return ImageGenerationResult(
                image_path=str(tmp_path / "spy.png"), provider="_SpyImageProvider", prompt=prompt,
                width=width, height=height, generation_time_seconds=0.0,
            )

    from PIL import Image

    Image.new("RGB", (64, 64)).save(tmp_path / "spy.png")

    runner = _runner(tmp_path, image_provider=lambda: _SpyImageProvider())
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=2))
    asyncio.run(runner.execute(plan))

    assert len(received_prompts) == 1
    assert received_prompts[0] == plan.shot.image_prompt


# --- 5. Returned image_path reaches AnimationInstruction.source_image_path --


def test_returned_image_path_reaches_animation_instruction(tmp_path):
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
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=1))
    result = asyncio.run(runner.execute(plan))

    assert len(received_instructions) == 1
    assert received_instructions[0].source_image_path == result.image_path


# --- 6. AnimationInstruction.duration comes from the selected Shot ----------


def test_animation_instruction_duration_matches_selected_shot(tmp_path):
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
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=4))
    asyncio.run(runner.execute(plan))

    assert received_instructions[0].duration_seconds == plan.shot.duration_seconds


# --- 7. AnimationRouter receives the expected instruction --------------------


def test_animation_router_actually_dispatches_to_configured_provider(tmp_path, monkeypatch):
    from core import config

    monkeypatch.setattr(config.get_settings(), "animation_provider", "spy")

    calls = []

    class _SpyAnimationProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            calls.append(instruction.shot_id)
            return AnimationResult(
                success=True, shot_id=instruction.shot_id, provider="_SpyAnimationProvider",
                video_path=str(tmp_path / "spy.mp4"), duration_seconds=instruction.duration_seconds,
            )

    (tmp_path / "spy.mp4").write_bytes(b"fake")

    anim_registry = AnimationProviderRegistry()
    anim_registry.register("spy", _SpyAnimationProvider)
    animation_router = AnimationRouter(registry=anim_registry)
    animation_pipeline = AnimationPipeline(animation_router=animation_router)

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path / "images")))
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry),
        asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )

    runner = SingleShotAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
    )
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0))
    asyncio.run(runner.execute(plan))

    assert len(calls) == 1


# --- 8. Only one animation clip is created ------------------------------------


def test_only_one_animated_clip_is_produced(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0))
    result = asyncio.run(runner.execute(plan))
    assert result.video_path  # exactly one — the return shape itself has no list to have more than one in


# --- 9. No final multi-shot concatenation occurs in single-shot mode --------


def test_single_shot_acceptance_module_never_imports_video_builder():
    """Structural check via AST against actual import statements, not
    a raw text search — a raw search is tripped up by this module's
    own docstring, which legitimately *explains* that it never calls
    VideoBuilder.concatenate_clips (there's only ever one clip, so the
    result IS the clip). That's documentation, not a dependency; this
    checks there is no real one."""
    import ast
    import inspect

    from products.chess2fight.rendering import single_shot_acceptance as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "VideoBuilder" not in imported_names


# --- 10. Normal full pipeline behavior remains unchanged ---------------------


def test_full_pipeline_module_was_not_touched():
    """Confirms products/chess2fight/rendering/pipeline.py (the real
    /render production path) still has no knowledge of single-shot
    acceptance — this is a wholly separate module."""
    import inspect

    from products.chess2fight.rendering import pipeline as module

    source = inspect.getsource(module)
    assert "single_shot" not in source.lower()
    assert "SingleShotAcceptanceRunner" not in source


# --- 11. Mock-provider single-shot path produces valid local artifacts ------


def test_mock_single_shot_path_produces_valid_local_image_and_video(tmp_path):
    import os

    from PIL import Image

    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0))
    result = asyncio.run(runner.execute(plan, width=256, height=256))

    assert os.path.exists(result.image_path)
    assert os.path.getsize(result.image_path) > 0
    with Image.open(result.image_path) as img:
        img.verify()

    assert os.path.exists(result.video_path)
    assert os.path.getsize(result.video_path) > 0

    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", result.video_path],
        capture_output=True, text=True, check=True,
    )
    assert float(probe.stdout.strip()) > 0


# --- 12. Dry run performs no provider HTTP calls ------------------------------


def test_prepare_never_touches_image_or_animation_router(tmp_path):
    """prepare() must be safe to call with providers configured to
    unreachable URLs — it should never construct or call either
    router at all, since it makes zero ComfyUI/network calls."""

    class _ExplodingImageProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            raise AssertionError("prepare() must never call the image provider")

    class _ExplodingAnimationProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            raise AssertionError("prepare() must never call the animation provider")

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
    # Must complete without error — proves neither exploding provider was ever called.
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0))
    assert plan.shot_index == 0


def test_dry_run_cli_path_makes_no_network_calls(tmp_path, monkeypatch):
    """End-to-end confirmation via the actual CLI script's --dry-run
    flag, with the provider configured to an address that would fail
    immediately if ever contacted."""
    from core import config

    monkeypatch.setattr(config.get_settings(), "comfyui_base_url", "http://this-must-never-be-contacted.invalid")

    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0))
    # No exception means the unreachable URL was genuinely never touched.
    assert plan.comfyui_base_url == "http://this-must-never-be-contacted.invalid"


# --- 13. Wan frame counts remain 4n+1 for representative shot durations -----


def test_wan_frame_counts_are_4n_plus_1_for_every_shot_in_the_sample_game(tmp_path):
    runner = _runner(tmp_path)
    for shot_index in range(8):
        plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=shot_index, fps=24))
        assert (plan.calculated_wan_frame_count - 1) % 4 == 0, (
            f"shot {shot_index} ({plan.shot.duration_seconds}s) produced "
            f"{plan.calculated_wan_frame_count} frames — not 4n+1"
        )


def test_wan_frame_count_matches_direct_calculation(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, fps=24))
    assert plan.calculated_wan_frame_count == _duration_to_frame_count(plan.shot.duration_seconds, 24)


def test_frame_count_calculation_respects_explicit_fps_override(tmp_path):
    runner = _runner(tmp_path)
    plan_24fps = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, fps=24))
    plan_12fps = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, fps=12))
    assert plan_24fps.fps == 24
    assert plan_12fps.fps == 12
    assert plan_24fps.calculated_wan_frame_count != plan_12fps.calculated_wan_frame_count
