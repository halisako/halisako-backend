"""Live integration test for ComfyUIAnimationProvider against a real
ComfyUI + Wan 2.2 TI2V-5B installation, using the real, supplied
wan22_i2v_5b.json workflow (Sprint 4 Prompt 4).

This is NOT part of the ordinary test suite. It requires:

1. `COMFYUI_LIVE_TEST=1` set in the environment.
2. A real, running ComfyUI server (default: http://localhost:8188,
   override via settings.comfyui_base_url / environment).
3. The Wan 2.2 TI2V-5B model files installed where ComfyUI expects
   them: wan2.2_ti2v_5B_fp16.safetensors, umt5_xxl_fp8_e4m3fn_scaled.safetensors,
   wan2.2_vae.safetensors. The workflow JSON itself IS present in this
   repository (products/chess2fight/rendering/workflows/wan22_i2v_5b.json)
   — unlike Prompt 3, that requirement is now satisfied; what's still
   missing in the environment this was written in is the actual GPU
   server and model weights to run it against.

Without all three, this test is skipped — not faked, not silently
passed. Run it explicitly with:

    COMFYUI_LIVE_TEST=1 pytest tests/test_comfyui_live_integration.py -v -s

An ordinary `pytest` / `pytest tests/` run never triggers this file's
tests — see the module-level skip condition below, evaluated at
collection time before any ComfyUI connection is attempted.
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from PIL import Image

from core.animation_providers.comfyui import ComfyUIAnimationProvider
from core.animation_router import AnimationInstruction
from core.config import get_settings

LIVE_TEST_ENABLED = os.environ.get("COMFYUI_LIVE_TEST") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE_TEST_ENABLED,
    reason=(
        "Live ComfyUI integration test skipped — set COMFYUI_LIVE_TEST=1 to run it "
        "(and see this file's own module docstring: a running ComfyUI server with "
        "the Wan 2.2 TI2V-5B model files installed is also required; the workflow "
        "JSON itself is already present in this repository)."
    ),
)

# The validated workflow's own proven, proof-quality configuration —
# using these exact values means a passing run is directly comparable
# to the original manual validation (640x352, 24fps, 49 frames, 2.04s).
_VALIDATED_WIDTH = 640
_VALIDATED_HEIGHT = 352
_VALIDATED_DURATION_SECONDS = 2.0416667


def _make_reference_image(path: Path) -> str:
    """A simple, real reference image — genuinely readable image
    content, not a placeholder pretending to be one."""
    Image.new("RGB", (_VALIDATED_WIDTH, _VALIDATED_HEIGHT), color=(180, 60, 60)).save(path)
    return str(path)


def test_comfyui_server_is_reachable_before_attempting_generation():
    """A focused pre-check, separate from the full generation test
    below, so a missing server produces a clear, specific failure
    message rather than an opaque one buried inside a longer test."""
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.comfyui_base_url.rstrip('/')}/system_stats", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.fail(
            f"ComfyUI is not reachable at {settings.comfyui_base_url!r}: {exc}. "
            "COMFYUI_LIVE_TEST=1 was set, but no real ComfyUI server is running — "
            "start one before running this test."
        )


def test_workflow_file_exists_before_attempting_generation():
    """This should now always pass — the real workflow file is
    committed to the repository as of Prompt 4. Kept as an explicit
    guard so a future accidental deletion fails clearly here rather
    than as a confusing failure deeper in the real generation test."""
    settings = get_settings()
    path = Path(settings.comfyui_workflow_path)
    if not path.exists():
        pytest.fail(
            f"No workflow file at {path!r} — expected the committed "
            "products/chess2fight/rendering/workflows/wan22_i2v_5b.json. "
            "Has it been moved or deleted?"
        )


def test_real_image_to_video_generation_produces_actual_motion(tmp_path):
    """The full proof-of-life: one real reference image, through the
    real provider, through the real validated Wan 2.2 TI2V-5B workflow,
    verified with ffprobe and frame-sampling — not merely an MP4
    container, but confirmed generated motion. Uses the workflow's own
    proven configuration (640x352, ~2.04s) so a passing result is
    directly comparable to the original manual validation."""
    provider = ComfyUIAnimationProvider(output_dir=str(tmp_path / "out"))

    image_path = _make_reference_image(tmp_path / "reference.png")
    instruction = AnimationInstruction(
        shot_id="live_test_shot",
        source_image_path=image_path,
        prompt=(
            "Cinematic fantasy battle. The warrior shifts into an attacking stance, "
            "raises his weapon and prepares to strike. Smooth, coherent cinematic motion."
        ),
        duration_seconds=_VALIDATED_DURATION_SECONDS,
        camera_motion="static",
        subject_motion="attacking stance",
        width=_VALIDATED_WIDTH,
        height=_VALIDATED_HEIGHT,
    )

    start = time.monotonic()
    result = asyncio.run(provider.generate_animation(instruction))
    generation_time = time.monotonic() - start

    print(f"\ngeneration_time_seconds: {generation_time:.1f}")
    print(f"result.success: {result.success}")
    print(f"result.error_message: {result.error_message}")
    print(f"result.video_path: {result.video_path}")
    print(f"result.metadata: {result.metadata}")

    assert result.success, f"Generation failed: {result.error_message}"
    assert result.video_path is not None
    assert Path(result.video_path).exists()

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", result.video_path],
        capture_output=True, text=True, check=True,
    )
    probe_data = json.loads(probe.stdout)
    video_stream = next(s for s in probe_data["streams"] if s["codec_type"] == "video")

    duration = float(probe_data["format"]["duration"])
    width, height = int(video_stream["width"]), int(video_stream["height"])
    print(f"probed duration: {duration} (validated reference: {_VALIDATED_DURATION_SECONDS})")
    print(f"probed resolution: {width}x{height} (validated reference: {_VALIDATED_WIDTH}x{_VALIDATED_HEIGHT})")
    print(f"probed fps: {video_stream.get('r_frame_rate')} (validated reference: 24/1)")

    assert duration > 0
    assert width > 0
    assert height > 0

    # The actual proof-of-life requirement: confirm generated motion,
    # not a static container — sample several frames and check they
    # are not identical to each other (a real animation changes pixel
    # content over time; a broken/static "generation" would not).
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    subprocess.run(
        ["ffmpeg", "-y", "-i", result.video_path, "-vf", "fps=4", str(extracted_dir / "f%03d.png")],
        capture_output=True, check=True,
    )
    frames = sorted(extracted_dir.iterdir())
    assert len(frames) >= 2, "Not enough frames extracted to verify motion."

    frame_signatures = []
    for frame_path in frames:
        img = Image.open(frame_path).convert("RGB").resize((32, 32))
        frame_signatures.append(tuple(img.getdata()))

    distinct_frames = len(set(frame_signatures))
    print(f"distinct frame signatures out of {len(frames)} sampled: {distinct_frames}")
    assert distinct_frames > 1, (
        "Every sampled frame is pixel-identical — this looks like a static image, "
        "not real generated motion."
    )
