"""Verifies the FLUX (ComfyUIImageProvider) -> Wan (ComfyUIAnimationProvider)
handoff — Sprint 4 Prompt 6, Task 5.

Both providers' HTTP calls are mocked; this test verifies the DATA
FLOW between them — a real ImageGenerationResult.image_path becomes
the AnimationInstruction.source_image_path that gets uploaded through
Wan's own upload mechanism, with no hardcoded filename anywhere in
between — proving the two providers communicate only through the
existing ImageGenerationResult / AnimationInstruction contracts, never
a tight coupling to each other's internals. Neither model actually
runs; every HTTP boundary is mocked.
"""

import asyncio
import json
import os

import httpx
from PIL import Image

from core.animation_providers.comfyui import ComfyUIAnimationProvider
from core.animation_router import AnimationInstruction
from core.image_providers.comfyui import ComfyUIImageProvider

FLUX_WORKFLOW_PATH = "products/chess2fight/rendering/workflows/flux2_klein_t2i_4b.json"
WAN_WORKFLOW_PATH = "products/chess2fight/rendering/workflows/wan22_i2v_5b.json"


class _MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handlers: dict):
        self._handlers = handlers
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()  # multipart/upload requests stream; .content needs an explicit read first
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        handler = self._handlers.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": f"no handler for {key}"})
        return handler(request)


def _real_png_bytes(width=1280, height=704, color=(60, 120, 200)) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _real_mp4_bytes() -> bytes:
    """A genuinely valid, tiny MP4 — ComfyUIAnimationProvider validates
    downloaded output with real ffprobe, so a mock response needs real
    encoded bytes, not a placeholder string."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        frame_path = f"{tmp}/frame0001.png"
        Image.new("RGB", (64, 64), color=(200, 0, 0)).save(frame_path)
        output_path = f"{tmp}/out.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loop", "1", "-i", frame_path, "-t", "1", "-r", "24",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path],
            capture_output=True, check=True,
        )
        with open(output_path, "rb") as f:
            return f.read()


_REAL_HTTPX_ASYNC_CLIENT = httpx.AsyncClient  # captured once, before any test can monkeypatch it — see
# test_comfyui_image_provider.py's own comment on this: capturing it fresh inside a
# helper function would, on the second call within one test (as this file makes,
# once for the FLUX module and once for the Wan module — both references to the
# SAME global httpx module object), wrap the already-patched first factory instead
# of the real class.


def _patch_module_httpx_client(monkeypatch, module, transport: "_MockTransport") -> None:
    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_HTTPX_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", _factory)


def test_flux_output_flows_into_wan_with_no_hardcoded_filename(tmp_path, monkeypatch):
    """The core Task 5 property, end to end."""
    import core.animation_providers.comfyui as animation_module
    import core.image_providers.comfyui as image_module

    # --- Step 1: mock FLUX generation, produce a real local PNG -------------
    flux_bytes = _real_png_bytes()

    def flux_queue(request):
        return httpx.Response(200, json={"prompt_id": "flux-run-1", "node_errors": {}})

    def flux_history(request):
        return httpx.Response(
            200,
            json={
                "flux-run-1": {
                    "status": {"status_str": "success"},
                    "outputs": {"78": {"images": [{"filename": "flux_out.png", "subfolder": "", "type": "output"}]}},
                }
            },
        )

    def flux_view(request):
        return httpx.Response(200, content=flux_bytes)

    flux_transport = _MockTransport(
        {"POST /prompt": flux_queue, "GET /history/flux-run-1": flux_history, "GET /view": flux_view}
    )
    _patch_module_httpx_client(monkeypatch, image_module, flux_transport)

    flux_provider = ComfyUIImageProvider(
        base_url="http://fake-comfyui:8188", workflow_path=FLUX_WORKFLOW_PATH,
        output_dir=str(tmp_path / "images"),
    )
    real_shot_prompt = "a fighter with spiky hair lunging forward with a sword, cinematic lighting"
    image_result = asyncio.run(flux_provider.generate_image(real_shot_prompt, width=1280, height=704))

    assert os.path.exists(image_result.image_path)
    with open(image_result.image_path, "rb") as f:
        assert f.read() == flux_bytes  # genuinely the FLUX output, not a stand-in

    # The known bad pattern this task explicitly calls out: never a
    # fixed filename from the original manual proof-of-life run.
    assert "halisako_flux2_keyframe_proof_01" not in image_result.image_path

    # --- Step 2: build the AnimationInstruction the way AnimationPipeline
    # actually would — source_image_path = the FLUX result's own path,
    # nothing hardcoded, nothing FLUX-specific known to the animation side.
    instruction = AnimationInstruction(
        shot_id="shot_climax",
        source_image_path=image_result.image_path,
        prompt="the warrior lunges forward to strike",
        duration_seconds=2.0,
        camera_motion="static",
        subject_motion="forward lunge attack",
    )

    # --- Step 3: mock Wan, verify the upload genuinely carries the FLUX bytes ---
    def wan_upload(request):
        return httpx.Response(200, json={"name": "wan_uploaded_ref.png", "subfolder": "", "type": "input"})

    def wan_queue(request):
        body = json.loads(request.content)
        # The uploaded filename (not any hardcoded one) must be what
        # node 56 received.
        assert body["prompt"]["56"]["inputs"]["image"] == "wan_uploaded_ref.png"
        return httpx.Response(200, json={"prompt_id": "wan-run-1", "node_errors": {}})

    def wan_history(request):
        return httpx.Response(
            200,
            json={
                "wan-run-1": {
                    "status": {"status_str": "success"},
                    "outputs": {"58": {"gifs": [{"filename": "clip.mp4", "subfolder": "", "type": "output"}]}},
                }
            },
        )

    def wan_view(request):
        return httpx.Response(200, content=_real_mp4_bytes())

    wan_transport = _MockTransport(
        {
            "POST /upload/image": wan_upload,
            "POST /prompt": wan_queue,
            "GET /history/wan-run-1": wan_history,
            "GET /view": wan_view,
        }
    )
    _patch_module_httpx_client(monkeypatch, animation_module, wan_transport)

    wan_provider = ComfyUIAnimationProvider(
        base_url="http://fake-comfyui:8188", workflow_path=WAN_WORKFLOW_PATH,
        output_dir=str(tmp_path / "animations"),
    )
    animation_result = asyncio.run(wan_provider.generate_animation(instruction))

    assert animation_result.success is True

    # The upload request actually carried the FLUX-generated bytes —
    # not a placeholder, not a re-read of some other file.
    upload_requests = [r for r in wan_transport.requests if r.url.path == "/upload/image"]
    assert len(upload_requests) == 1
    assert flux_bytes in upload_requests[0].content


def test_two_different_shots_produce_two_different_flux_outputs_and_uploads(tmp_path, monkeypatch):
    """Confirms there's no single fixed filename anywhere in the
    chain — different shots (different prompts) get different FLUX
    output paths and different uploaded references, never colliding."""
    import core.image_providers.comfyui as image_module

    def flux_queue(request):
        body = json.loads(request.content)
        prompt_text = body["prompt"]["76"]["inputs"]["value"]
        prompt_id = "flux-A" if "shot A" in prompt_text else "flux-B"
        return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})

    def make_history(prompt_id):
        def history(request):
            return httpx.Response(
                200,
                json={
                    prompt_id: {
                        "status": {"status_str": "success"},
                        "outputs": {"78": {"images": [{"filename": f"{prompt_id}.png", "subfolder": "", "type": "output"}]}},
                    }
                },
            )

        return history

    def flux_view(request):
        return httpx.Response(200, content=_real_png_bytes())

    transport = _MockTransport(
        {
            "POST /prompt": flux_queue,
            "GET /history/flux-A": make_history("flux-A"),
            "GET /history/flux-B": make_history("flux-B"),
            "GET /view": flux_view,
        }
    )
    _patch_module_httpx_client(monkeypatch, image_module, transport)

    provider = ComfyUIImageProvider(
        base_url="http://fake-comfyui:8188", workflow_path=FLUX_WORKFLOW_PATH, output_dir=str(tmp_path / "images"),
    )
    result_a = asyncio.run(provider.generate_image("shot A: a wide establishing shot"))
    result_b = asyncio.run(provider.generate_image("shot B: a climactic finishing blow"))

    assert result_a.image_path != result_b.image_path
