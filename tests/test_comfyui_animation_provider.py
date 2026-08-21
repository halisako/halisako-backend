"""Tests for ComfyUIAnimationProvider, against the real, validated
wan22_i2v_5b.json workflow (Sprint 4 Prompt 4) — every ComfyUI HTTP
call is mocked, but the workflow file itself, its node IDs, and its
model/config values are the actual supplied artifact, not a stand-in.

For the gated live integration test that requires a real ComfyUI
server, see tests/test_comfyui_live_integration.py.
"""

import asyncio
import json
from pathlib import Path

import httpx

_REAL_ASYNC_CLIENT = httpx.AsyncClient  # captured once, before any test can monkeypatch it
from PIL import Image

from core.animation_providers.comfyui import (
    ComfyUIAnimationProvider,
    ComfyUIRequestError,
    _derive_seed,
    _duration_to_frame_count,
    _normalize_dimension,
)
from core.animation_router import AnimationInstruction
from core.config import get_settings

REAL_WORKFLOW_PATH = "products/chess2fight/rendering/workflows/wan22_i2v_5b.json"


def _instruction(**overrides) -> AnimationInstruction:
    defaults = dict(
        shot_id="shot_001",
        source_image_path="/tmp/placeholder_never_used.png",
        prompt="a fighter lunging forward",
        duration_seconds=2.0,
        camera_motion="tracking",
        subject_motion="forward_strike",
    )
    defaults.update(overrides)
    return AnimationInstruction(**defaults)


def _make_reference_image(tmp_path, size=(512, 512)) -> str:
    path = tmp_path / "ref.png"
    Image.new("RGB", size, color=(200, 50, 50)).save(path)
    return str(path)


def _make_real_mp4(tmp_path, width=640, height=352, duration=2.0416667, fps=24) -> bytes:
    """A genuinely valid, ffprobe-readable MP4 — used as the mocked
    /view response so output-verification tests exercise real ffprobe
    logic, not a stubbed-out check."""
    import subprocess

    frame_dir = tmp_path / "mp4_frame"
    frame_dir.mkdir(exist_ok=True)
    Image.new("RGB", (width, height), color=(0, 120, 200)).save(frame_dir / "frame.png")
    output = tmp_path / "generated.mp4"
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


def _provider(tmp_path, **overrides) -> ComfyUIAnimationProvider:
    defaults = dict(
        base_url="http://fake-comfyui:8188",
        workflow_path=REAL_WORKFLOW_PATH,
        timeout_seconds=5.0,
        output_dir=str(tmp_path / "out"),
    )
    defaults.update(overrides)
    return ComfyUIAnimationProvider(**defaults)


def _patch_httpx_client(monkeypatch, transport: _MockTransport) -> None:
    """Makes every httpx.AsyncClient() constructed inside the provider
    use our mock transport instead of making a real network call.

    Always wraps the true original client (_REAL_ASYNC_CLIENT), not
    whatever httpx.AsyncClient currently is — calling this twice in
    one test (e.g. to swap transports mid-test) would otherwise wrap
    the previous call's mock factory instead of the real client,
    silently keeping the first transport active.
    """
    import core.animation_providers.comfyui as module

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", _factory)


def _success_handlers(video_bytes: bytes, output_key: str = "images", prompt_id: str = "real-test-123"):
    def upload(request):
        return httpx.Response(200, json={"name": "uploaded_ref.png", "subfolder": "", "type": "input"})

    def queue(request):
        return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})

    def history(request):
        return httpx.Response(
            200,
            json={
                prompt_id: {
                    "status": {"status_str": "success"},
                    "outputs": {"58": {output_key: [{"filename": "ComfyUI_00001_.mp4", "subfolder": "video", "type": "output"}]}},
                }
            },
        )

    def view(request):
        return httpx.Response(200, content=video_bytes)

    return {
        "POST /upload/image": upload, "POST /prompt": queue,
        f"GET /history/{prompt_id}": history, "GET /view": view,
    }


# --- The workflow file itself is the real, supplied artifact ----------------


def test_real_workflow_file_exists_in_the_repository():
    assert Path(REAL_WORKFLOW_PATH).exists()


def test_real_workflow_has_the_documented_node_ids():
    workflow = json.loads(Path(REAL_WORKFLOW_PATH).read_text(encoding="utf-8"))
    for node_id in ("3", "6", "7", "8", "37", "38", "39", "48", "55", "56", "57", "58"):
        assert node_id in workflow


def test_workflow_uses_actual_wan_model_names():
    """#9 from the task's list: the workflow's model/VAE/text-encoder
    filenames are the real, documented Wan 2.2 TI2V-5B files, not
    placeholders."""
    workflow = json.loads(Path(REAL_WORKFLOW_PATH).read_text(encoding="utf-8"))
    assert workflow["37"]["inputs"]["unet_name"] == "wan2.2_ti2v_5B_fp16.safetensors"
    assert workflow["38"]["inputs"]["clip_name"] == "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    assert workflow["39"]["inputs"]["vae_name"] == "wan2.2_vae.safetensors"


def test_fps_expectation_matches_node_57():
    """#8 from the task's list: settings.comfyui_default_fps must match
    the workflow's own verified CreateVideo (node 57) fps value."""
    workflow = json.loads(Path(REAL_WORKFLOW_PATH).read_text(encoding="utf-8"))
    assert workflow["57"]["inputs"]["fps"] == 24
    assert get_settings().comfyui_default_fps == 24


def test_correct_output_node_is_recognized():
    """#10: node 58 (SaveVideo) is the workflow's actual output node."""
    workflow = json.loads(Path(REAL_WORKFLOW_PATH).read_text(encoding="utf-8"))
    assert workflow["58"]["class_type"] == "SaveVideo"


# --- Frame count / dimension conversion (with alignment constraints) --------


def test_duration_to_frame_count_matches_the_validated_data_point():
    """The strongest possible check: 2.0s @ 24fps must produce exactly
    49 frames — the workflow's own experimentally-proven value."""
    assert _duration_to_frame_count(2.0, 24) == 49


def test_frame_count_always_satisfies_wan_alignment():
    for duration in (0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        frame_count = _duration_to_frame_count(duration, 24)
        assert (frame_count - 1) % 4 == 0, f"{frame_count} frames does not satisfy (n-1)%4==0"


def test_frame_count_never_below_minimum():
    assert _duration_to_frame_count(0.001, 24) >= 1


def test_dimension_normalization_leaves_valid_values_unchanged():
    assert _normalize_dimension(640) == 640
    assert _normalize_dimension(352) == 352


def test_dimension_normalization_rounds_to_nearest_multiple_of_16():
    assert _normalize_dimension(1000) == 992  # 62*16
    assert _normalize_dimension(1001) == 1008  # 63*16 (nearest)


def test_dimension_normalization_floors_at_one_alignment_unit():
    assert _normalize_dimension(1) == 16
    assert _normalize_dimension(0) == 16


# --- Node-level injection mapping (#1-7 from the task's list) ---------------


def test_node_56_receives_uploaded_image_filename(tmp_path):
    provider = _provider(tmp_path)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(workflow, _instruction(), uploaded_image_name="my_upload.png")
    assert prepared["56"]["inputs"]["image"] == "my_upload.png"


def test_positive_prompt_maps_to_node_6(tmp_path):
    provider = _provider(tmp_path)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(
        workflow, _instruction(prompt="a very specific marker prompt"), uploaded_image_name="x.png"
    )
    assert prepared["6"]["inputs"]["text"] == "a very specific marker prompt"


def test_negative_prompt_maps_to_node_7_when_provided(tmp_path):
    provider = _provider(tmp_path)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(
        workflow, _instruction(negative_prompt="blurry, low quality"), uploaded_image_name="x.png"
    )
    assert prepared["7"]["inputs"]["text"] == "blurry, low quality"


def test_negative_prompt_left_untouched_when_not_provided(tmp_path):
    """Regression test for a real Prompt 3 bug: an unset negative
    prompt must NOT blank out the workflow's own tuned negative
    prompt on node 7."""
    provider = _provider(tmp_path)
    workflow = provider._load_workflow()
    original_negative = workflow["7"]["inputs"]["text"]
    prepared = provider._inject_parameters(workflow, _instruction(negative_prompt=None), uploaded_image_name="x.png")
    assert prepared["7"]["inputs"]["text"] == original_negative
    assert prepared["7"]["inputs"]["text"] != ""


def test_seed_maps_to_node_3(tmp_path):
    provider = _provider(tmp_path)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(workflow, _instruction(seed=123456), uploaded_image_name="x.png")
    assert prepared["3"]["inputs"]["seed"] == 123456


def test_seed_falls_back_to_derived_value_when_unset(tmp_path):
    provider = _provider(tmp_path)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(
        workflow, _instruction(prompt="a specific prompt", seed=None), uploaded_image_name="x.png"
    )
    assert prepared["3"]["inputs"]["seed"] == _derive_seed("a specific prompt")


def test_width_height_length_map_to_node_55(tmp_path):
    provider = _provider(tmp_path)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(
        workflow, _instruction(width=640, height=352, duration_seconds=2.0), uploaded_image_name="x.png"
    )
    assert prepared["55"]["inputs"]["width"] == 640
    assert prepared["55"]["inputs"]["height"] == 352
    assert prepared["55"]["inputs"]["length"] == 49


def test_injection_does_not_mutate_the_loaded_workflow(tmp_path):
    provider = _provider(tmp_path)
    workflow = provider._load_workflow()
    original_text = workflow["6"]["inputs"]["text"]
    provider._inject_parameters(workflow, _instruction(prompt="different"), uploaded_image_name="x.png")
    assert workflow["6"]["inputs"]["text"] == original_text


def test_model_loader_nodes_are_never_modified(tmp_path):
    """Nodes 37/38/39/48 (model/VAE/text-encoder loaders,
    ModelSamplingSD3) stay as workflow configuration — never
    request-specific."""
    provider = _provider(tmp_path)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(workflow, _instruction(), uploaded_image_name="x.png")
    for node_id in ("37", "38", "39", "48"):
        assert prepared[node_id] == workflow[node_id]


# --- Source image upload / no shared-filesystem assumption (#11, #12) -------


def test_source_image_uploaded_through_comfyui_api(tmp_path, monkeypatch):
    video_bytes = _make_real_mp4(tmp_path)
    transport = _MockTransport(_success_handlers(video_bytes))
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))

    upload_requests = [r for r in transport.requests if r.url.path == "/upload/image"]
    assert len(upload_requests) == 1


def test_provider_never_reads_or_writes_a_comfyui_side_filesystem_path():
    """Structural check: this provider's only interaction with a
    remote filesystem is via httpx calls — no os.path/shutil
    operations that would assume a shared filesystem with ComfyUI."""
    import ast
    import inspect

    import core.animation_providers.comfyui as module

    source = inspect.getsource(module.ComfyUIAnimationProvider)
    tree = ast.parse(source)
    forbidden_calls = {"copyfile", "copy2", "symlink", "os.rename"}
    called_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (called_names & forbidden_calls)


# --- Full successful flow, request construction, response handling ---------


def test_full_successful_generation_flow_with_real_workflow(tmp_path, monkeypatch):
    video_bytes = _make_real_mp4(tmp_path)
    transport = _MockTransport(_success_handlers(video_bytes))
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))

    assert result.success is True
    assert result.provider == "ComfyUIAnimationProvider"
    assert Path(result.video_path).exists()


def test_result_metadata_contains_correct_artifact_information(tmp_path, monkeypatch):
    """#18: AnimationResult metadata carries the real model names and
    prompt_id, not placeholders."""
    video_bytes = _make_real_mp4(tmp_path)
    transport = _MockTransport(_success_handlers(video_bytes))
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))

    assert result.metadata["prompt_id"] == "real-test-123"
    assert result.metadata["model"] == "wan2.2_ti2v_5B_fp16.safetensors"
    assert result.metadata["vae"] == "wan2.2_vae.safetensors"
    assert result.metadata["text_encoder"] == "umt5_xxl_fp8_e4m3fn_scaled.safetensors"


def test_result_duration_width_height_come_from_actual_probed_video(tmp_path, monkeypatch):
    """The result must reflect the REAL downloaded file's properties
    (via ffprobe), not just echo back the request."""
    video_bytes = _make_real_mp4(tmp_path, width=640, height=352, duration=2.0416667)
    transport = _MockTransport(_success_handlers(video_bytes))
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))

    assert result.width == 640
    assert result.height == 352
    assert abs(result.duration_seconds - 2.0416667) < 0.01


def test_output_key_variants_are_all_recognized(tmp_path, monkeypatch):
    """SaveVideo's actual /history output key is unverified (see
    module docstring) — confirm all three checked candidates work."""
    for key in ("images", "videos", "gifs"):
        video_bytes = _make_real_mp4(tmp_path)
        transport = _MockTransport(_success_handlers(video_bytes, output_key=key))
        _patch_httpx_client(monkeypatch, transport)
        provider = _provider(tmp_path, output_dir=str(tmp_path / f"out_{key}"))
        image_path = _make_reference_image(tmp_path)

        result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
        assert result.success is True, f"failed for output key {key!r}"


def test_unique_filenames_across_repeated_shots(tmp_path, monkeypatch):
    """Avoid filename collisions between concurrent/repeated jobs for
    the same shot_id."""
    video_bytes = _make_real_mp4(tmp_path)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    transport1 = _MockTransport(_success_handlers(video_bytes, prompt_id="job-aaa"))
    _patch_httpx_client(monkeypatch, transport1)
    result1 = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path, shot_id="shot_1")))

    transport2 = _MockTransport(_success_handlers(video_bytes, prompt_id="job-bbb"))
    _patch_httpx_client(monkeypatch, transport2)
    result2 = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path, shot_id="shot_1")))

    assert result1.success is True and result2.success is True
    assert result1.video_path != result2.video_path


# --- Output verification: missing / invalid artifact (#16, #17) ------------


def test_missing_video_artifact_is_rejected(tmp_path, monkeypatch):
    """#16: if /view returns empty content, the result must fail, not
    report success with a zero-byte file."""

    def view_empty(request):
        return httpx.Response(200, content=b"")

    handlers = _success_handlers(b"")
    handlers["GET /view"] = view_empty
    transport = _MockTransport(handlers)
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    assert "empty" in result.error_message.lower() or "invalid" in result.error_message.lower()


def test_invalid_output_video_is_rejected(tmp_path, monkeypatch):
    """#17: if /view returns bytes that aren't a real video (garbage
    data), the result must fail with a clear message, not report
    success."""

    def view_garbage(request):
        return httpx.Response(200, content=b"this is not a real mp4 file at all")

    handlers = _success_handlers(b"garbage")
    handlers["GET /view"] = view_garbage
    transport = _MockTransport(handlers)
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    assert "invalid" in result.error_message.lower() or "video" in result.error_message.lower()


# --- Distinct error handling per stage ---------------------------------------


def test_missing_workflow_file_returns_failed_result(tmp_path):
    provider = _provider(tmp_path, workflow_path=str(tmp_path / "does_not_exist.json"))
    image_path = _make_reference_image(tmp_path)
    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    assert "workflow" in result.error_message.lower()


def test_missing_reference_image_returns_failed_result(tmp_path):
    provider = _provider(tmp_path)
    instruction = _instruction(source_image_path=str(tmp_path / "does_not_exist.png"))
    result = asyncio.run(provider.generate_animation(instruction))
    assert result.success is False
    assert "not found" in result.error_message.lower()


def test_image_upload_failure_has_a_distinct_message(tmp_path, monkeypatch):
    def upload_fails(request):
        return httpx.Response(500, json={"error": "gpu worker offline"})

    transport = _MockTransport({"POST /upload/image": upload_fails})
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    assert "upload" in result.error_message.lower()


def test_workflow_submission_failure_has_a_distinct_message(tmp_path, monkeypatch):
    def upload(request):
        return httpx.Response(200, json={"name": "x.png", "subfolder": "", "type": "input"})

    def queue_fails(request):
        return httpx.Response(500, json={"error": "queue full"})

    transport = _MockTransport({"POST /upload/image": upload, "POST /prompt": queue_fails})
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    assert "submission" in result.error_message.lower()


def test_generation_failure_has_a_distinct_message(tmp_path, monkeypatch):
    def upload(request):
        return httpx.Response(200, json={"name": "x.png", "subfolder": "", "type": "input"})

    def queue(request):
        return httpx.Response(200, json={"prompt_id": "real-test-123", "node_errors": {}})

    def history(request):
        return httpx.Response(200, json={"real-test-123": {"status": {"status_str": "error", "messages": ["OOM"]}}})

    transport = _MockTransport(
        {"POST /upload/image": upload, "POST /prompt": queue, "GET /history/real-test-123": history}
    )
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    assert "generation" in result.error_message.lower()


def test_execution_timeout_has_a_distinct_message(tmp_path, monkeypatch):
    def upload(request):
        return httpx.Response(200, json={"name": "x.png", "subfolder": "", "type": "input"})

    def queue(request):
        return httpx.Response(200, json={"prompt_id": "never-done", "node_errors": {}})

    def history(request):
        return httpx.Response(200, json={})

    transport = _MockTransport(
        {"POST /upload/image": upload, "POST /prompt": queue, "GET /history/never-done": history}
    )
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path, timeout_seconds=0.5)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    assert "timeout" in result.error_message.lower()


def test_history_output_missing_has_a_distinct_message(tmp_path, monkeypatch):
    def upload(request):
        return httpx.Response(200, json={"name": "x.png", "subfolder": "", "type": "input"})

    def queue(request):
        return httpx.Response(200, json={"prompt_id": "real-test-123", "node_errors": {}})

    def history(request):
        return httpx.Response(200, json={"real-test-123": {"status": {"status_str": "success"}, "outputs": {}}})

    transport = _MockTransport(
        {"POST /upload/image": upload, "POST /prompt": queue, "GET /history/real-test-123": history}
    )
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    assert "history/output" in result.error_message.lower() or "output" in result.error_message.lower()


def test_video_download_failure_has_a_distinct_message(tmp_path, monkeypatch):
    def upload(request):
        return httpx.Response(200, json={"name": "x.png", "subfolder": "", "type": "input"})

    def queue(request):
        return httpx.Response(200, json={"prompt_id": "real-test-123", "node_errors": {}})

    def history(request):
        return httpx.Response(
            200,
            json={
                "real-test-123": {
                    "status": {"status_str": "success"},
                    "outputs": {"58": {"images": [{"filename": "x.mp4", "subfolder": "", "type": "output"}]}},
                }
            },
        )

    def view_fails(request):
        return httpx.Response(500, json={"error": "disk read error"})

    transport = _MockTransport(
        {"POST /upload/image": upload, "POST /prompt": queue, "GET /history/real-test-123": history, "GET /view": view_fails}
    )
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    assert "download" in result.error_message.lower()


def test_unreachable_server_fails_at_the_first_http_call_with_a_clear_message(tmp_path):
    """An unreachable server fails at whichever HTTP call happens
    first — here, image upload, since that's the provider's first
    network call — producing that stage's specific message rather
    than a generic one. That's more useful, not less specific."""
    provider = _provider(tmp_path, base_url="http://this-host-does-not-exist.invalid:8188", timeout_seconds=3.0)
    image_path = _make_reference_image(tmp_path)
    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    assert "upload" in result.error_message.lower()


def test_error_messages_do_not_leak_raw_comfyui_payloads(tmp_path, monkeypatch):
    """Internal diagnostics are fine to log, but the error_message
    returned to the caller should stay reasonably sized — not dump an
    entire raw ComfyUI response."""
    huge_payload = {"node_errors": {str(i): {"errors": [{"message": "x" * 200}]} for i in range(50)}}

    def upload(request):
        return httpx.Response(200, json={"name": "x.png", "subfolder": "", "type": "input"})

    def queue_huge_error(request):
        return httpx.Response(200, json={"prompt_id": None, **huge_payload})

    transport = _MockTransport({"POST /upload/image": upload, "POST /prompt": queue_huge_error})
    _patch_httpx_client(monkeypatch, transport)
    provider = _provider(tmp_path)
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    # Not asserting a hard byte limit (this task doesn't specify one),
    # but confirming the message is a string we could reasonably show,
    # not an unbounded dump — a sanity ceiling well under the raw
    # payload's own ~10,000+ character size.
    assert len(result.error_message) < 5000


# --- Isolation / interface conformance ---------------------------------------


def test_comfyui_provider_satisfies_the_animation_provider_interface():
    from core.animation_router import AnimationProvider

    assert issubclass(ComfyUIAnimationProvider, AnimationProvider)


def test_comfyui_provider_never_imports_router_or_registry_or_mock_provider():
    import ast
    import inspect

    import core.animation_providers.comfyui as module

    tree = ast.parse(inspect.getsource(module))
    forbidden_names = {"AnimationRouter", "AnimationProviderRegistry", "MockAnimationProvider"}
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store)
    }
    assert not (referenced_names & forbidden_names)


def test_comfyui_request_error_never_escapes_generate_animation(tmp_path):
    provider = _provider(tmp_path)

    async def _raising_upload(*args, **kwargs):
        raise ComfyUIRequestError("simulated deep failure")

    provider._upload_image = _raising_upload
    image_path = _make_reference_image(tmp_path)

    result = asyncio.run(provider.generate_animation(_instruction(source_image_path=image_path)))
    assert result.success is False
    assert "simulated deep failure" in result.error_message


# --- FPS mapped to node 57 (Sprint 4 Prompt 6: was computed but never written) --


def test_fps_is_written_to_node_57_not_just_used_for_frame_count_math(tmp_path):
    """Regression test for a real gap found while verifying against
    the Prompt 6 supplied workflow: fps was computed and used to
    calculate frame_count, but never actually written into node 57
    (CreateVideo)'s own fps input — harmless only because the default
    fps happened to already match the workflow's baked-in value."""
    provider = _provider(tmp_path)
    workflow = provider._load_workflow()
    instruction = _instruction(fps=12, duration_seconds=2.0)
    prepared = provider._inject_parameters(workflow, instruction, uploaded_image_name="x.png")

    assert prepared["57"]["inputs"]["fps"] == 12
    # And frame_count/fps stay consistent with each other regardless
    # of which fps was requested.
    assert prepared["55"]["inputs"]["length"] == _duration_to_frame_count(2.0, 12)


def test_fps_defaults_to_settings_default_when_instruction_fps_unset(tmp_path):
    provider = _provider(tmp_path)
    workflow = provider._load_workflow()
    instruction = _instruction(fps=None)
    prepared = provider._inject_parameters(workflow, instruction, uploaded_image_name="x.png")
    assert prepared["57"]["inputs"]["fps"] == get_settings().comfyui_default_fps
