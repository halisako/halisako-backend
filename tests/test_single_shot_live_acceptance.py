"""Live single-shot acceptance test — Sprint 4 Prompt 7.

This is NOT part of the ordinary test suite. It requires:

1. `HALISAKO_SINGLE_SHOT_LIVE_TEST=1` set in the environment.
2. A real, running ComfyUI server with both the validated FLUX.2 Klein
   and Wan 2.2 TI2V models installed (see
   products/chess2fight/rendering/workflows/VALIDATED-SETTINGS.md).
3. `image_provider=comfyui` and `animation_provider=comfyui` configured.

Without all three, this test is skipped — not faked, not silently
passed. Run it explicitly with:

    HALISAKO_SINGLE_SHOT_LIVE_TEST=1 \\
    IMAGE_PROVIDER=comfyui ANIMATION_PROVIDER=comfyui \\
    COMFYUI_BASE_URL=http://<your-comfyui-host>:8188 \\
    pytest tests/test_single_shot_live_acceptance.py -v -s

This drives the real production rendering classes
(RenderPipeline/AnimationPipeline via SingleShotAcceptanceRunner) for
exactly one real shot from the bundled sample PGN — proving the
externally-validated FLUX -> Wan workflows work not just standalone
(Prompt 6) but when actually invoked by Chess2Fight's own cinematic
pipeline. Per this task's explicit instruction, this file's own
existence and content do not claim any live result — only an actual
run with the environment variables above produces one.
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
from products.chess2fight.rendering.single_shot_acceptance import SingleShotAcceptanceRunner
from products.chess2fight.schemas import BattleMode, BattlePreferences

LIVE_TEST_ENABLED = os.environ.get("HALISAKO_SINGLE_SHOT_LIVE_TEST") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_TEST_ENABLED,
    reason=(
        "Live single-shot acceptance test skipped — set HALISAKO_SINGLE_SHOT_LIVE_TEST=1 "
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


def test_real_shot_renders_and_animates_through_the_production_pipeline():
    """The full acceptance path: one real shot, from a real PGN,
    through the real RenderPipeline/AnimationPipeline, via the real,
    externally-validated FLUX and Wan providers."""
    runner = SingleShotAcceptanceRunner(get_ai_provider())
    preferences = BattlePreferences(battle_mode=BattleMode.DUEL, style="anime")
    pgn = SAMPLE_PGN_PATH.read_text()

    plan = asyncio.run(runner.prepare(pgn, preferences, shot_index=0))
    print(f"\nSelected shot: {plan.shot.shot_type.value}, {plan.shot.duration_seconds:.2f}s")
    print(f"Image prompt: {plan.shot.image_prompt[:200]}")
    print(f"Calculated Wan frame count: {plan.calculated_wan_frame_count}")

    result = asyncio.run(runner.execute(plan))

    # --- Image checks ---
    assert os.path.exists(result.image_path), f"Image not found: {result.image_path}"
    assert os.path.getsize(result.image_path) > 0
    with Image.open(result.image_path) as image:
        image.verify()
    print(f"\nImage path: {result.image_path}")

    # --- Video checks ---
    assert os.path.exists(result.video_path), f"Video not found: {result.video_path}"
    assert os.path.getsize(result.video_path) > 0

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", result.video_path],
        capture_output=True, text=True, check=True,
    )
    probe_data = json.loads(probe.stdout)
    video_stream = next(s for s in probe_data["streams"] if s["codec_type"] == "video")

    duration = float(probe_data["format"]["duration"])
    print(f"Video path: {result.video_path}")
    print(f"Video duration: {duration:.2f}s")
    print(f"Video resolution: {video_stream['width']}x{video_stream['height']}")
    print(f"Video fps: {video_stream.get('r_frame_rate')}")

    assert duration > 0
    assert video_stream["width"] > 0
    assert video_stream["height"] > 0
