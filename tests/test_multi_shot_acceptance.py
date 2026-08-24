"""Tests for MultiShotAcceptanceRunner — Sprint 4 Prompt 11.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted; every provider call here goes through
mock or spy providers. Gated true live tests, if any, live in a
separate, explicitly-gated file — see
tests/test_multi_shot_live_acceptance.py.

All mock providers use real ffmpeg-backed MockAnimationProvider /
Pillow-backed MockImageProvider — this file intentionally does not
weaken any of that real media validation.
"""

import asyncio

import pytest

from core.ai_router import TemplateProvider
from core.animation_router import (
    AnimationInstruction,
    AnimationProvider,
    AnimationProviderRegistry,
    AnimationResult,
    AnimationRouter,
    AnimationType,
    MockAnimationProvider,
)
from core.image_router import ImageGenerationResult, ImageProvider, ImageProviderRegistry, ImageRouter, MockImageProvider
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.multi_shot_acceptance import (
    MultiShotAcceptanceRunner,
    ShotCountExceedsAcceptanceCapError,
    ShotRangeOutOfRangeError,
)
from products.chess2fight.rendering.render_pipeline import RenderPipeline
from products.chess2fight.rendering.video_builder import VideoBuilder
from tests.test_single_shot_acceptance import _preferences, _sample_pgn


def _runner(tmp_path, image_provider=None, animation_provider=None, video_builder=None) -> MultiShotAcceptanceRunner:
    from products.chess2fight.rendering.animation_pipeline import AnimationPipeline

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", image_provider or (lambda: MockImageProvider(output_dir=str(tmp_path / "images"))))
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry),
        asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register(
        "mock", animation_provider or (lambda: MockAnimationProvider(output_dir=str(tmp_path / "animations")))
    )
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    return MultiShotAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=video_builder or VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )


# --- 1/2. Exactly 3 real shots selected; shots outside range never rendered -


def test_default_plan_selects_exactly_three_real_timeline_shots(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences()))
    assert plan.selected_shot_indices == [0, 1, 2]
    assert len(plan.shots) == 3
    assert plan.shot_count == 3


def test_shots_outside_selected_range_never_reach_render_pipeline(tmp_path):
    """Spy on the image provider — only 3 distinct prompts (matching
    shots 0,1,2's real image_prompt values) may ever be requested."""
    received_prompts = []

    class _SpyImageProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            received_prompts.append(prompt)
            from PIL import Image

            path = tmp_path / f"spy_{len(received_prompts)}.png"
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=str(path), provider="_SpyImageProvider", prompt=prompt,
                width=width, height=height, generation_time_seconds=0.0,
            )

    runner = _runner(tmp_path, image_provider=lambda: _SpyImageProvider())
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), start_shot_index=0, shot_count=3))
    asyncio.run(runner.execute(plan))

    real_prompts_for_selected_shots = {shot.image_prompt for shot in plan.shots}
    assert set(received_prompts) == real_prompts_for_selected_shots
    assert len(received_prompts) == 3


# --- 3/4/5. Exactly 3 image calls, 3 animation calls, 1 concatenation call --


def test_exactly_three_image_provider_calls_and_three_animation_calls(tmp_path):
    image_calls = []
    animation_calls = []

    class _SpyImageProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            image_calls.append(prompt)
            from PIL import Image

            path = tmp_path / f"spy_img_{len(image_calls)}.png"
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=str(path), provider="_SpyImageProvider", prompt=prompt,
                width=width, height=height, generation_time_seconds=0.0,
            )

    class _SpyAnimationProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            animation_calls.append(instruction.shot_id)
            # A genuinely real, ffmpeg-produced tiny MP4 — not a fake
            # byte blob — since this now flows through real
            # concatenation and real ffprobe measurement (Sprint 4
            # Prompt 11.1), not a mocked video builder.
            import subprocess

            from PIL import Image

            frame_dir = tmp_path / f"spy_frame_{len(animation_calls)}"
            frame_dir.mkdir(exist_ok=True)
            Image.new("RGB", (64, 64)).save(frame_dir / "f.png")
            output = tmp_path / f"spy_vid_{len(animation_calls)}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loop", "1", "-i", str(frame_dir / "f.png"), "-t", "1",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)],
                capture_output=True, check=True,
            )
            return AnimationResult(
                success=True, shot_id=instruction.shot_id, provider="_SpyAnimationProvider",
                video_path=str(output), duration_seconds=instruction.duration_seconds,
            )

    runner = _runner(tmp_path, image_provider=lambda: _SpyImageProvider(), animation_provider=lambda: _SpyAnimationProvider())
    # The real VideoBuilder — no mock needed now that the input clips
    # are genuinely real, concatenatable MP4s.

    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences()))
    asyncio.run(runner.execute(plan))

    assert len(image_calls) == 3
    assert len(animation_calls) == 3


def test_exactly_one_local_concatenation_call(tmp_path):
    concat_calls = []

    real_video_builder = VideoBuilder()

    class _CountingVideoBuilder(VideoBuilder):
        async def concatenate_clips(self, *args, **kwargs):
            concat_calls.append(1)
            return await real_video_builder.concatenate_clips(*args, **kwargs)

    runner = _runner(tmp_path, video_builder=_CountingVideoBuilder())
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    asyncio.run(runner.execute(plan, width=256, height=256))

    assert len(concat_calls) == 1


# --- 6/7. Ordering: timeline order preserved regardless of completion timing -


def test_generation_order_is_timeline_order(tmp_path):
    call_order = []

    class _OrderTrackingImageProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            call_order.append(("image", prompt[:30]))
            from PIL import Image

            path = tmp_path / f"order_{len(call_order)}.png"
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=str(path), provider="_OrderTrackingImageProvider", prompt=prompt,
                width=width, height=height, generation_time_seconds=0.0,
            )

    runner = _runner(tmp_path, image_provider=lambda: _OrderTrackingImageProvider())
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences()))
    asyncio.run(runner.execute(plan))

    expected_order = [("image", shot.image_prompt[:30]) for shot in plan.shots]
    assert call_order == expected_order


def test_output_concatenation_order_is_unaffected_by_variable_per_shot_generation_time(tmp_path):
    """Sprint 4 Prompt 11.1 wording correction: the original name/
    docstring here implied this tests concurrent out-of-order
    *completion* — it doesn't, and can't, since
    AnimationPipeline.animate() processes shots sequentially (awaits
    one at a time, never asyncio.gather()), confirmed directly against
    its source in this module's own audit. Shot 2's call cannot
    actually begin, let alone finish, before shot 0's completes.

    What this test genuinely proves: even when different shots take
    different (artificially varied) amounts of wall-clock time to
    generate, sequential processing order — and therefore output
    order — remains timeline order regardless. That's still a real,
    worthwhile property to check explicitly (a latency-dependent bug
    is exactly the kind of thing that could otherwise go unnoticed
    until shots genuinely varied in generation time), just not the
    concurrency-reordering property the old name suggested.
    """
    call_log = []

    class _VariableDelayAnimationProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            # Reverse-magnitude delay: the first shot requested gets
            # the LONGEST artificial delay, the last gets the
            # shortest. Since processing is sequential either way,
            # this just varies each shot's own wall-clock duration —
            # it doesn't (and can't) make a later shot's call begin,
            # let alone finish, before an earlier one's completes.
            delay = 0.05 * (3 - len(call_log))
            call_log.append(instruction.shot_id)
            await asyncio.sleep(delay)
            from PIL import Image
            import subprocess

            frame_dir = tmp_path / f"delay_frame_{instruction.shot_id}"
            frame_dir.mkdir(exist_ok=True)
            Image.new("RGB", (64, 64)).save(frame_dir / "f.png")
            output = tmp_path / f"delay_clip_{instruction.shot_id}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loop", "1", "-i", str(frame_dir / "f.png"), "-t", "1",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)],
                capture_output=True, check=True,
            )
            return AnimationResult(
                success=True, shot_id=instruction.shot_id, provider="_VariableDelayAnimationProvider",
                video_path=str(output), duration_seconds=instruction.duration_seconds,
            )

    runner = _runner(tmp_path, animation_provider=lambda: _VariableDelayAnimationProvider())
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=1.0))
    result = asyncio.run(runner.execute(plan, width=64, height=64))

    expected_shot_ids_in_order = [shot.shot_id for shot in plan.shots]
    # video_paths must correspond to shot_ids in timeline order, not call/delay order.
    for video_path, expected_shot_id in zip(result.video_paths, expected_shot_ids_in_order, strict=True):
        assert expected_shot_id in video_path


# --- 8/9/10/11. Resolution and FPS policy --------------------------------------


def test_flux_keyframes_use_1280x704_production_policy(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, width=256, height=256))

    from PIL import Image

    for image_path in result.image_paths:
        with Image.open(image_path) as img:
            assert img.size == (1280, 704)


def test_wan_animations_use_832x480_by_default(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan))  # no explicit width/height override

    import subprocess

    for video_path in result.video_paths:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, check=True,
        )
        width, height = probe.stdout.strip().split(",")
        assert (int(width), int(height)) == (832, 480)


def test_default_fps_is_8(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences()))
    assert plan.fps == 8


def test_two_second_cap_resolves_to_17_frames_per_shot(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    assert plan.calculated_wan_frame_counts == [17, 17, 17]


# --- 12. Original PromptedShot durations are not mutated --------------------


def test_original_shot_durations_not_mutated(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    original_durations = [shot.duration_seconds for shot in plan.shots]
    asyncio.run(runner.execute(plan, width=256, height=256))
    assert [shot.duration_seconds for shot in plan.shots] == original_durations
    assert original_durations != [2.0, 2.0, 2.0]  # genuinely different from the cap


# --- 13. Shot count cannot silently exceed the acceptance safety limit -----


def test_shot_count_above_default_cap_is_rejected(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(ShotCountExceedsAcceptanceCapError):
        asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_count=4))


def test_shot_count_above_cap_succeeds_with_explicit_override(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(
        runner.prepare(_sample_pgn(), _preferences(), shot_count=5, allow_exceeding_default_cap=True)
    )
    assert plan.shot_count == 5


def test_shot_count_at_exactly_the_cap_needs_no_override(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_count=3))
    assert plan.shot_count == 3


# --- 14. Invalid ranges fail before provider generation ---------------------


def test_invalid_range_raises_before_any_provider_call(tmp_path):
    class _ExplodingImageProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            raise AssertionError("must never be called for an invalid range")

    runner = _runner(tmp_path, image_provider=lambda: _ExplodingImageProvider())
    with pytest.raises(ShotRangeOutOfRangeError):
        asyncio.run(runner.prepare(_sample_pgn(), _preferences(), start_shot_index=6, shot_count=3))


def test_negative_start_index_rejected(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(ShotRangeOutOfRangeError):
        asyncio.run(runner.prepare(_sample_pgn(), _preferences(), start_shot_index=-1, shot_count=3))


# --- 18/19. Partial failure identifies the failing shot, no false success --


def test_partial_failure_raises_clearly_identifying_the_failing_shot(tmp_path):
    call_count = {"n": 0}

    class _FailsOnThirdAnimationProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            call_count["n"] += 1
            if call_count["n"] == 3:
                return AnimationResult(
                    success=False, shot_id=instruction.shot_id, provider="_FailsOnThirdAnimationProvider",
                    error_message="simulated GPU failure on third shot",
                )
            return AnimationResult(
                success=True, shot_id=instruction.shot_id, provider="_FailsOnThirdAnimationProvider",
                video_path=str(tmp_path / f"ok_{call_count['n']}.mp4"), duration_seconds=instruction.duration_seconds,
            )

    for i in (1, 2):
        (tmp_path / f"ok_{i}.mp4").write_bytes(b"fake")

    runner = _runner(tmp_path, animation_provider=lambda: _FailsOnThirdAnimationProvider())
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))

    with pytest.raises(Exception) as exc_info:
        asyncio.run(runner.execute(plan, width=256, height=256))

    assert "simulated GPU failure" in str(exc_info.value)
    # The failing shot's own id is identifiable in the error.
    third_shot_id = plan.shots[2].shot_id
    assert third_shot_id in str(exc_info.value)


def test_partial_failure_never_produces_a_final_video(tmp_path):
    """No fake replacement clip, no final concatenated video, on a
    mid-sequence failure."""
    call_count = {"n": 0}

    class _FailsOnSecondAnimationProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return AnimationResult(
                    success=False, shot_id=instruction.shot_id, provider="_FailsOnSecondAnimationProvider",
                    error_message="simulated failure",
                )
            return AnimationResult(
                success=True, shot_id=instruction.shot_id, provider="_FailsOnSecondAnimationProvider",
                video_path=str(tmp_path / "ok_1.mp4"), duration_seconds=instruction.duration_seconds,
            )

    (tmp_path / "ok_1.mp4").write_bytes(b"fake")

    runner = _runner(tmp_path, animation_provider=lambda: _FailsOnSecondAnimationProvider())
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))

    fight_dir = tmp_path / "storage" / "renders" / plan.fight_id
    with pytest.raises(Exception):
        asyncio.run(runner.execute(plan, width=256, height=256))

    final_path = fight_dir / "multi_shot_acceptance.mp4"
    assert not final_path.exists()


# --- 20. VideoBuilder receives exactly 3 real clip paths, correct order ----


def test_video_builder_receives_exactly_three_clips_in_correct_order(tmp_path):
    received_paths = []
    real_video_builder = VideoBuilder()

    class _CapturingVideoBuilder(VideoBuilder):
        async def concatenate_clips(self, clip_paths, *args, **kwargs):
            received_paths.extend(clip_paths)
            return await real_video_builder.concatenate_clips(clip_paths, *args, **kwargs)

    runner = _runner(tmp_path, video_builder=_CapturingVideoBuilder())
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, width=256, height=256))

    assert len(received_paths) == 3
    assert received_paths == result.video_paths


# --- 21. Dry-run results in zero provider calls -----------------------------


def test_prepare_alone_makes_zero_provider_calls(tmp_path):
    class _ExplodingImageProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            raise AssertionError("prepare() must never call the image provider")

    class _ExplodingAnimationProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            raise AssertionError("prepare() must never call the animation provider")

    runner = _runner(
        tmp_path, image_provider=lambda: _ExplodingImageProvider(), animation_provider=lambda: _ExplodingAnimationProvider()
    )
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    assert plan.shot_count == 3


# --- 22. Manifest contains all 3 selected shots in correct order -----------


def test_manifest_json_contains_all_three_shots_in_correct_order(tmp_path):
    import json
    import subprocess
    import sys

    runner_dir = tmp_path / "manifest_test"
    runner_dir.mkdir()
    manifest_path = runner_dir / "manifest.json"

    subprocess.run(
        [sys.executable, "scripts/render_multi_shot_acceptance.py", "--sample", "--max-animation-seconds", "2",
         "--manifest-path", str(manifest_path)],
        capture_output=True, text=True, check=True,
    )
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert [s["timeline_index"] for s in manifest["shots"]] == [0, 1, 2]
    assert len(manifest["shots"]) == 3

    import shutil

    shutil.rmtree("generated_animations", ignore_errors=True)
    shutil.rmtree("generated_images", ignore_errors=True)
    shutil.rmtree("storage", ignore_errors=True)


# --- 23/24. Generic contracts unchanged --------------------------------------


def test_generic_image_router_defaults_unchanged():
    from core.image_router import MockImageProvider as _MockImageProvider

    result = asyncio.run(_MockImageProvider().generate_image("a prompt"))
    assert (result.width, result.height) == (1024, 1024)


def test_generic_animation_instruction_defaults_unchanged():
    instruction = AnimationInstruction(
        shot_id="s1", source_image_path="/tmp/x.png", prompt="test", duration_seconds=2.0,
        camera_motion="static", subject_motion="test",
    )
    assert instruction.animation_type == AnimationType.IMAGE_TO_VIDEO
    assert (instruction.width, instruction.height) == (1024, 1024)
