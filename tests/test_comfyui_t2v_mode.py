"""Tests for ComfyUIAnimationProvider's T2V (text-to-video) mode —
Sprint 4 Prompt 8.

TEST CATEGORY (Sprint 4 Prompt 9's explicit distinction): every test in
this file is a MOCKED UNIT TEST or a LOCAL CONTRACT/INTEGRATION TEST —
no real ComfyUI server or GPU is ever contacted. The gated TRUE
GPU/COMFYUI LIVE TEST for T2V lives in
tests/test_comfyui_live_integration.py's
`test_real_text_to_video_generation_produces_actual_motion`.

Against the real, supplied wan22_t2v_5b.json — the file this task's
own artifacts labeled as "the I2V API workflow" but which is
structurally text-to-video (no LoadImage, no start_image wiring at
all on Wan22ImageToVideoLatent). See core/animation_providers/comfyui.py's
module docstring for the full discrepancy discovered between that
label and the file's actual structure. This test file exists
specifically to make sure T2V is exercised meaningfully — not just
"a mock returns success" — matching this task's explicit instruction
not to over-mock the important structure.

I2V's own tests remain in tests/test_comfyui_animation_provider.py,
unchanged in spirit; this file is additive, not a replacement.
"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from core.animation_providers.comfyui import ComfyUIAnimationProvider, _duration_to_frame_count
from core.animation_router import AnimationInstruction, AnimationType
from core.config import get_settings

T2V_WORKFLOW_PATH = "products/chess2fight/rendering/workflows/wan22_t2v_5b.json"
I2V_WORKFLOW_PATH = "products/chess2fight/rendering/workflows/wan22_i2v_5b.json"

_REAL_ASYNC_CLIENT = httpx.AsyncClient  # captured once, before any test can monkeypatch it


def _t2v_instruction(**overrides) -> AnimationInstruction:
    defaults = dict(
        shot_id="shot_t2v_001",
        prompt="a knight chess piece rotating on a chessboard",
        duration_seconds=2.0,
        camera_motion="static",
        subject_motion="rotate",
        animation_type=AnimationType.TEXT_TO_VIDEO,
    )
    defaults.update(overrides)
    return AnimationInstruction(**defaults)


def _provider(tmp_path, **overrides) -> ComfyUIAnimationProvider:
    defaults = dict(
        base_url="http://fake-comfyui:8188",
        workflow_path=I2V_WORKFLOW_PATH,
        t2v_workflow_path=T2V_WORKFLOW_PATH,
        timeout_seconds=5.0,
        output_dir=str(tmp_path / "out"),
    )
    defaults.update(overrides)
    return ComfyUIAnimationProvider(**defaults)


def _make_real_mp4(tmp_path, width=832, height=480, duration=2.125, fps=8) -> bytes:
    """A genuinely valid, ffprobe-readable MP4 — matches the T2V
    workflow's own validated 832x480/8fps settings."""
    import subprocess

    frame_dir = tmp_path / "t2v_mp4_frame"
    frame_dir.mkdir(exist_ok=True)
    Image.new("RGB", (width, height), color=(40, 40, 200)).save(frame_dir / "frame.png")
    output = tmp_path / "t2v_generated.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loop", "1", "-framerate", str(fps), "-i", str(frame_dir / "frame.png"),
            "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
        ],
        capture_output=True, check=True,
    )
    return output.read_bytes()


class _MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handlers: dict):
        self._handlers = handlers
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        handler = self._handlers.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": f"no handler for {key}"})
        return handler(request)


def _patch_httpx_client(monkeypatch, transport: "_MockTransport") -> None:
    """Always wraps the true original client, not whatever
    httpx.AsyncClient currently is — see the same fix already applied
    in test_comfyui_animation_provider.py and test_flux_to_wan_handoff.py."""
    import core.animation_providers.comfyui as module

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", _factory)


def _t2v_success_handlers(video_bytes: bytes, prompt_id: str = "t2v-test-123"):
    def queue(request):
        return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})

    def history(request):
        return httpx.Response(
            200,
            json={
                prompt_id: {
                    "status": {"status_str": "success"},
                    "outputs": {"58": {"images": [{"filename": "t2v_out.mp4", "subfolder": "video", "type": "output"}]}},
                }
            },
        )

    def view(request):
        return httpx.Response(200, content=video_bytes)

    return {"POST /prompt": queue, f"GET /history/{prompt_id}": history, "GET /view": view}


# --- T2V workflow file itself is the real, supplied artifact ----------------


def test_t2v_workflow_file_exists_in_the_repository():
    assert Path(T2V_WORKFLOW_PATH).exists()


def test_t2v_workflow_has_no_load_image_node():
    """The structural fact this whole T2V mode depends on: unlike I2V,
    this workflow genuinely has no image input at all."""
    workflow = json.loads(Path(T2V_WORKFLOW_PATH).read_text(encoding="utf-8"))
    assert "56" not in workflow
    assert not any(n.get("class_type") == "LoadImage" for n in workflow.values())
    assert "start_image" not in workflow["55"]["inputs"]


def test_t2v_workflow_uses_the_same_validated_model_names_as_i2v():
    t2v = json.loads(Path(T2V_WORKFLOW_PATH).read_text(encoding="utf-8"))
    i2v = json.loads(Path(I2V_WORKFLOW_PATH).read_text(encoding="utf-8"))
    assert t2v["37"]["inputs"]["unet_name"] == i2v["37"]["inputs"]["unet_name"] == "wan2.2_ti2v_5B_fp16.safetensors"
    assert t2v["38"]["inputs"]["clip_name"] == i2v["38"]["inputs"]["clip_name"]
    assert t2v["39"]["inputs"]["vae_name"] == i2v["39"]["inputs"]["vae_name"]


def test_t2v_workflow_uses_the_new_ground_truth_settings():
    workflow = json.loads(Path(T2V_WORKFLOW_PATH).read_text(encoding="utf-8"))
    assert workflow["55"]["inputs"]["width"] == 832
    assert workflow["55"]["inputs"]["height"] == 480
    assert workflow["55"]["inputs"]["length"] == 17
    assert workflow["3"]["inputs"]["steps"] == 8
    assert workflow["3"]["inputs"]["cfg"] == 5
    assert workflow["3"]["inputs"]["sampler_name"] == "uni_pc"
    assert workflow["57"]["inputs"]["fps"] == 8


# --- AnimationInstruction validation: T2V vs I2V requirements ---------------


def test_t2v_instruction_does_not_require_source_image_path():
    instruction = _t2v_instruction()
    assert instruction.source_image_path is None


def test_i2v_instruction_without_source_image_path_rejected():
    with pytest.raises(ValueError, match="source_image_path is required"):
        AnimationInstruction(
            shot_id="s1", prompt="test", duration_seconds=2.0, camera_motion="static",
            subject_motion="test", animation_type=AnimationType.IMAGE_TO_VIDEO,
        )


def test_default_animation_type_is_still_image_to_video():
    """Backward compatibility: every existing caller that doesn't set
    animation_type explicitly still gets I2V behavior, unchanged."""
    instruction = AnimationInstruction(
        shot_id="s1", source_image_path="/tmp/x.png", prompt="test", duration_seconds=2.0,
        camera_motion="static", subject_motion="test",
    )
    assert instruction.animation_type == AnimationType.IMAGE_TO_VIDEO


# --- T2V parameter injection: workflow parameter injection for T2V ---------


def test_t2v_injection_sets_prompt_seed_dimensions_fps():
    provider = _provider(Path("/tmp"))
    workflow = provider._load_workflow(provider._t2v_workflow_path)
    instruction = _t2v_instruction(seed=99999, width=832, height=480, fps=8)
    prepared = provider._inject_t2v_parameters(workflow, instruction)

    assert prepared["6"]["inputs"]["text"] == "a knight chess piece rotating on a chessboard"
    assert prepared["3"]["inputs"]["seed"] == 99999
    assert prepared["55"]["inputs"]["width"] == 832
    assert prepared["55"]["inputs"]["height"] == 480
    assert prepared["57"]["inputs"]["fps"] == 8


def test_t2v_injection_never_touches_an_image_node():
    """The key correctness property for T2V: there is no image node to
    accidentally set, and injection must not fabricate one."""
    provider = _provider(Path("/tmp"))
    workflow = provider._load_workflow(provider._t2v_workflow_path)
    prepared = provider._inject_t2v_parameters(workflow, _t2v_instruction())
    assert "56" not in prepared
    assert "start_image" not in prepared["55"]["inputs"]


def test_t2v_injection_frame_count_uses_the_new_default_fps():
    provider = _provider(Path("/tmp"))
    workflow = provider._load_workflow(provider._t2v_workflow_path)
    prepared = provider._inject_t2v_parameters(workflow, _t2v_instruction(duration_seconds=2.0, fps=None))
    assert prepared["55"]["inputs"]["length"] == _duration_to_frame_count(2.0, get_settings().comfyui_default_fps)
    assert prepared["55"]["inputs"]["length"] == 17


def test_t2v_injection_preserves_negative_prompt_when_unset():
    provider = _provider(Path("/tmp"))
    workflow = provider._load_workflow(provider._t2v_workflow_path)
    original_negative = workflow["7"]["inputs"]["text"]
    prepared = provider._inject_t2v_parameters(workflow, _t2v_instruction(negative_prompt=None))
    assert prepared["7"]["inputs"]["text"] == original_negative
    assert original_negative  # genuinely non-empty in the real file


def test_t2v_injection_overrides_negative_prompt_when_provided():
    provider = _provider(Path("/tmp"))
    workflow = provider._load_workflow(provider._t2v_workflow_path)
    prepared = provider._inject_t2v_parameters(workflow, _t2v_instruction(negative_prompt="a custom negative"))
    assert prepared["7"]["inputs"]["text"] == "a custom negative"


def test_t2v_injection_does_not_mutate_the_loaded_workflow():
    provider = _provider(Path("/tmp"))
    workflow = provider._load_workflow(provider._t2v_workflow_path)
    original_prompt = workflow["6"]["inputs"]["text"]
    provider._inject_t2v_parameters(workflow, _t2v_instruction(prompt="a different prompt entirely"))
    assert workflow["6"]["inputs"]["text"] == original_prompt


# --- Full mocked T2V generation flow: no upload call, real MP4 out ---------


def test_full_t2v_generation_flow_produces_a_real_valid_mp4(tmp_path, monkeypatch):
    video_bytes = _make_real_mp4(tmp_path)
    provider = _provider(tmp_path)
    transport = _MockTransport(_t2v_success_handlers(video_bytes))
    _patch_httpx_client(monkeypatch, transport)

    result = asyncio.run(provider.generate_animation(_t2v_instruction()))

    assert result.success is True
    assert result.provider == "ComfyUIAnimationProvider"
    assert result.metadata["mode"] == "t2v"

    # No upload call was ever made — the core T2V-specific property.
    upload_requests = [r for r in transport.requests if r.url.path == "/upload/image"]
    assert len(upload_requests) == 0

    from pathlib import Path as _Path

    assert _Path(result.video_path).exists()
    assert _Path(result.video_path).stat().st_size > 0


def test_i2v_generation_still_calls_upload_exactly_once(tmp_path, monkeypatch):
    """Regression guard, run alongside the T2V no-upload test above so
    the contrast is explicit: I2V mode must still upload exactly once."""
    from tests.test_comfyui_animation_provider import _instruction, _make_reference_image, _success_handlers

    video_bytes = _make_real_mp4(tmp_path, width=832, height=480)
    provider = _provider(tmp_path)
    transport = _MockTransport(_success_handlers(video_bytes))
    _patch_httpx_client(monkeypatch, transport)

    image_path = _make_reference_image(tmp_path)
    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))

    assert result.success is True
    assert result.metadata["mode"] == "i2v"
    upload_requests = [r for r in transport.requests if r.url.path == "/upload/image"]
    assert len(upload_requests) == 1


def test_t2v_result_carries_correct_metadata(tmp_path, monkeypatch):
    video_bytes = _make_real_mp4(tmp_path)
    provider = _provider(tmp_path)
    transport = _MockTransport(_t2v_success_handlers(video_bytes))
    _patch_httpx_client(monkeypatch, transport)

    result = asyncio.run(provider.generate_animation(_t2v_instruction()))
    assert result.metadata["mode"] == "t2v"
    assert result.metadata["model"] == "wan2.2_ti2v_5B_fp16.safetensors"


def test_missing_t2v_workflow_file_returns_failed_result(tmp_path):
    provider = _provider(tmp_path, t2v_workflow_path=str(tmp_path / "does_not_exist.json"))
    result = asyncio.run(provider.generate_animation(_t2v_instruction()))
    assert result.success is False
    assert "not found" in result.error_message.lower()


# --- MockAnimationProvider correctly rejects T2V-style instructions --------


def test_mock_animation_provider_rejects_t2v_instruction_clearly():
    from core.animation_router import MockAnimationProvider

    provider = MockAnimationProvider()
    result = asyncio.run(provider.generate_animation(_t2v_instruction()))
    assert result.success is False
    assert "text-to-video" in result.error_message.lower() or "no text-to-video" in result.error_message.lower()
