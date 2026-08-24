"""Live multi-shot acceptance test — Sprint 4 Prompt 11.

This is NOT part of the ordinary test suite. It requires:

1. `COMFYUI_MULTI_SHOT_LIVE_TEST=1` set in the environment.
2. A real, running ComfyUI server with both the validated FLUX.2 Klein
   and Wan 2.2 TI2V models installed (see
   products/chess2fight/rendering/workflows/VALIDATED-SETTINGS.md).
3. `image_provider=comfyui` and `animation_provider=comfyui` configured.

Without all three, this test is skipped — not faked, not silently
passed. Run it explicitly with:

    COMFYUI_MULTI_SHOT_LIVE_TEST=1 \\
    IMAGE_PROVIDER=comfyui ANIMATION_PROVIDER=comfyui \\
    COMFYUI_BASE_URL=http://<your-comfyui-host>:8188 \\
    pytest tests/test_multi_shot_live_acceptance.py -v -s

This drives the real production rendering classes
(RenderPipeline/AnimationPipeline/VideoBuilder via
MultiShotAcceptanceRunner) for exactly 3 real shots from the bundled
sample PGN — proving the same production wiring already GPU-validated
for one shot (Sprint 4 Prompt 10) scales correctly to 3, with correct
ordering and local concatenation. Per this task's explicit instruction,
this file's own existence and content do not claim any live result —
only an actual run with the environment variables above produces one.
This is deliberately capped at 3 shots (this module's own default
safety cap) — it does not attempt the full 8-shot timeline.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import httpx
import pytest
from PIL import Image

from core.ai_router import get_ai_provider
from core.config import get_settings
from products.chess2fight.rendering.multi_shot_acceptance import MultiShotAcceptanceRunner
from products.chess2fight.schemas import BattleMode, BattlePreferences

LIVE_TEST_ENABLED = os.environ.get("COMFYUI_MULTI_SHOT_LIVE_TEST") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_TEST_ENABLED,
    reason=(
        "Live multi-shot acceptance test skipped — set COMFYUI_MULTI_SHOT_LIVE_TEST=1 "
        "to run it (and see this file's own module docstring for the other requirements: "
        "a running ComfyUI server with both validated models, and "
        "image_provider=comfyui / animation_provider=comfyui configured)."
    ),
)

SAMPLE_PGN_PATH = Path("products/chess2fight/rendering/fixtures/sample_acceptance_game.pgn")


def test_comfyui_server_is_reachable_before_attempting_generation():
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.comfyui_base_url.rstrip('/')}/system_stats", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.fail(f"ComfyUI is not reachable at {settings.comfyui_base_url!r}: {exc}.")


def test_both_provider_workflows_exist():
    settings = get_settings()
    image_path = Path(settings.comfyui_image_workflow_path)
    animation_path = Path(settings.comfyui_workflow_path)
    if not image_path.exists():
        pytest.fail(f"FLUX workflow missing at {image_path!r}.")
    if not animation_path.exists():
        pytest.fail(f"Wan workflow missing at {animation_path!r}.")


def test_providers_are_configured_for_comfyui():
    settings = get_settings()
    if settings.image_provider != "comfyui":
        pytest.fail(f"image_provider is {settings.image_provider!r}, not 'comfyui' — set IMAGE_PROVIDER=comfyui.")
    if settings.animation_provider != "comfyui":
        pytest.fail(
            f"animation_provider is {settings.animation_provider!r}, not 'comfyui' — "
            "set ANIMATION_PROVIDER=comfyui."
        )


def test_three_real_shots_render_animate_and_concatenate_through_the_production_pipeline():
    """The full 3-shot acceptance path: three real shots, from a real
    PGN, through the real RenderPipeline/AnimationPipeline/VideoBuilder,
    via the real, externally-validated FLUX and Wan providers, capped
    to ~2s animation per shot — the first paid multi-shot milestone."""
    runner = MultiShotAcceptanceRunner(get_ai_provider())
    preferences = BattlePreferences(battle_mode=BattleMode.DUEL, style="anime")
    pgn = SAMPLE_PGN_PATH.read_text(encoding="utf-8")

    plan = asyncio.run(
        runner.prepare(pgn, preferences, start_shot_index=0, shot_count=3, max_animation_seconds=2.0)
    )
    print(f"\nSelected shots: {plan.selected_shot_indices}")
    for i, shot in enumerate(plan.shots):
        print(f"  shot[{plan.selected_shot_indices[i]}]: {shot.shot_type.value}, "
              f"real={shot.duration_seconds:.2f}s, effective={plan.effective_animation_durations_seconds[i]:.2f}s, "
              f"frames={plan.calculated_wan_frame_counts[i]}")
    print(f"Expected ComfyUI job count: {plan.expected_comfyui_job_count}")
    print(f"Expected assembled duration: ~{plan.expected_assembled_duration_seconds:.3f}s")

    result = asyncio.run(runner.execute(plan))

    # --- Per-shot image checks: exactly 3, all valid 1280x704 -----------
    assert len(result.image_paths) == 3
    for image_path in result.image_paths:
        assert os.path.exists(image_path), f"Image not found: {image_path}"
        assert os.path.getsize(image_path) > 0
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            assert image.size == (1280, 704), f"{image_path} is {image.size}, expected (1280, 704)"
    print(f"\nAll 3 FLUX keyframes valid at 1280x704: {result.image_paths}")

    # --- Per-shot clip checks: exactly 3, all valid 832x480/8fps/~2.125s
    assert len(result.video_paths) == 3
    for video_path in result.video_paths:
        assert os.path.exists(video_path), f"Video not found: {video_path}"
        assert os.path.getsize(video_path) > 0
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", video_path],
            capture_output=True, text=True, check=True,
        )
        probe_data = json.loads(probe.stdout)
        video_stream = next(s for s in probe_data["streams"] if s["codec_type"] == "video")
        duration = float(probe_data["format"]["duration"])
        print(f"  {video_path}: {video_stream['width']}x{video_stream['height']} "
              f"@ {video_stream.get('r_frame_rate')}, {duration:.3f}s")
        assert (video_stream["width"], video_stream["height"]) == (832, 480)
        assert duration > 0

    # --- Final concatenated video ----------------------------------------
    assert os.path.exists(result.final_video_path), f"Final video not found: {result.final_video_path}"
    assert os.path.getsize(result.final_video_path) > 0

    final_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", result.final_video_path],
        capture_output=True, text=True, check=True,
    )
    final_probe_data = json.loads(final_probe.stdout)
    final_stream = next(s for s in final_probe_data["streams"] if s["codec_type"] == "video")
    final_duration = float(final_probe_data["format"]["duration"])

    print(f"\nFinal video: {result.final_video_path}")
    print(f"Final resolution: {final_stream['width']}x{final_stream['height']}")
    print(f"Final fps: {final_stream.get('r_frame_rate')}")
    print(f"Final duration: {final_duration:.3f}s (expected ~{plan.expected_assembled_duration_seconds:.3f}s)")

    assert (final_stream["width"], final_stream["height"]) == (832, 480)
    assert final_duration > 0
