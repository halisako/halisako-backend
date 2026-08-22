"""Tests for scripts/comfyui_single_shot_smoke.py.

TEST CATEGORY: MOCKED UNIT TEST — ComfyUI's HTTP boundary is mocked;
no real server or GPU involved. Imports the script's own `_main()`
directly rather than running it as a subprocess, so httpx can be
patched the same way every other provider test in this suite already
does.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import httpx
from PIL import Image

_SCRIPT_PATH = Path("scripts/comfyui_single_shot_smoke.py")
_REAL_ASYNC_CLIENT = httpx.AsyncClient  # captured once, before any test can monkeypatch it


def _load_script_module():
    """Imports the script as a module so its _main()/_parse_args() are
    directly callable and testable — the script itself guards its
    entry point behind `if __name__ == "__main__"`, so importing it
    doesn't execute anything."""
    spec = importlib.util.spec_from_file_location("comfyui_single_shot_smoke", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_script_module()


def _make_real_mp4(tmp_path, width=832, height=480, duration=2.125, fps=8) -> bytes:
    import subprocess

    frame_dir = tmp_path / "script_mp4_frame"
    frame_dir.mkdir(exist_ok=True)
    Image.new("RGB", (width, height), color=(60, 180, 60)).save(frame_dir / "frame.png")
    output = tmp_path / "script_generated.mp4"
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
        await request.aread()
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        handler = self._handlers.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": f"no handler for {key}"})
        return handler(request)


def _patch_httpx_client(monkeypatch, transport: "_MockTransport") -> None:
    import core.animation_providers.comfyui as comfyui_module

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(comfyui_module.httpx, "AsyncClient", _factory)


def _success_handlers(video_bytes: bytes, prompt_id: str = "smoke-script-123"):
    def upload(request):
        return httpx.Response(200, json={"name": "uploaded.png", "subfolder": "", "type": "input"})

    def queue(request):
        return httpx.Response(200, json={"prompt_id": prompt_id, "node_errors": {}})

    def history(request):
        return httpx.Response(
            200,
            json={
                prompt_id: {
                    "status": {"status_str": "success"},
                    "outputs": {"58": {"images": [{"filename": "out.mp4", "subfolder": "video", "type": "output"}]}},
                }
            },
        )

    def view(request):
        return httpx.Response(200, content=video_bytes)

    return {"POST /upload/image": upload, "POST /prompt": queue, f"GET /history/{prompt_id}": history, "GET /view": view}


def _make_reference_image(tmp_path) -> str:
    path = tmp_path / "ref.png"
    Image.new("RGB", (832, 480), color=(200, 50, 50)).save(path)
    return str(path)


def test_missing_image_fails_fast_with_no_network_call(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv",
        ["comfyui_single_shot_smoke.py", "--image", str(tmp_path / "does_not_exist.png"), "--prompt", "test"],
    )
    exit_code = asyncio.run(smoke._main())
    assert exit_code == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_resolved_defaults_match_the_validated_baseline(tmp_path, monkeypatch, capsys):
    """Confirms the script's own default resolution logic — 832x480,
    8fps, 17 frames — without needing a real server: this only
    exercises argument parsing and default resolution, which happens
    before any network call."""
    image_path = _make_reference_image(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["comfyui_single_shot_smoke.py", "--base-url", "http://unreachable.invalid:1",
         "--image", image_path, "--prompt", "test prompt", "--timeout-seconds", "1"],
    )
    asyncio.run(smoke._main())
    stdout = capsys.readouterr().out
    assert "resolved width:      832" in stdout
    assert "resolved height:     480" in stdout
    assert "resolved fps:        8" in stdout
    assert "resolved frame count: 17" in stdout


def test_successful_generation_via_mocked_comfyui(tmp_path, monkeypatch, capsys):
    video_bytes = _make_real_mp4(tmp_path)
    transport = _MockTransport(_success_handlers(video_bytes))
    _patch_httpx_client(monkeypatch, transport)

    image_path = _make_reference_image(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["comfyui_single_shot_smoke.py", "--base-url", "http://fake-comfyui:8188",
         "--image", image_path, "--prompt", "a warrior attacks",
         "--output-dir", str(tmp_path / "out")],
    )
    exit_code = asyncio.run(smoke._main())
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "SUCCESS" in stdout
    assert "prompt_id:           smoke-script-123" in stdout

    import re

    match = re.search(r"video_path:\s+(\S+)", stdout)
    assert match is not None
    video_path = Path(match.group(1))
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_failed_generation_exits_nonzero_with_real_error_no_fake_output(tmp_path, monkeypatch, capsys):
    """No fake MP4 fallback — a failure must produce no output file and
    a non-zero exit, per this task's explicit instruction."""

    def failing_upload(request):
        return httpx.Response(500, json={"error": "internal error"})

    transport = _MockTransport({"POST /upload/image": failing_upload})
    _patch_httpx_client(monkeypatch, transport)

    image_path = _make_reference_image(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv",
        ["comfyui_single_shot_smoke.py", "--base-url", "http://fake-comfyui:8188",
         "--image", image_path, "--prompt", "test", "--output-dir", str(output_dir)],
    )
    exit_code = asyncio.run(smoke._main())
    stderr = capsys.readouterr().err

    assert exit_code == 1
    assert "FAILED" in stderr
    # No output file was ever written.
    if output_dir.exists():
        assert list(output_dir.iterdir()) == []


def test_negative_prompt_flows_through_to_the_instruction(tmp_path, monkeypatch, capsys):
    video_bytes = _make_real_mp4(tmp_path)
    transport = _MockTransport(_success_handlers(video_bytes))
    _patch_httpx_client(monkeypatch, transport)

    image_path = _make_reference_image(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["comfyui_single_shot_smoke.py", "--base-url", "http://fake-comfyui:8188",
         "--image", image_path, "--prompt", "test", "--negative-prompt", "blurry, low quality",
         "--output-dir", str(tmp_path / "out")],
    )
    asyncio.run(smoke._main())

    import json

    queue_requests = [r for r in transport.requests if r.url.path == "/prompt"]
    body = json.loads(queue_requests[0].content)
    assert body["prompt"]["7"]["inputs"]["text"] == "blurry, low quality"


def test_explicit_duration_and_fps_override_the_baseline_default(tmp_path, monkeypatch, capsys):
    video_bytes = _make_real_mp4(tmp_path, fps=24)
    transport = _MockTransport(_success_handlers(video_bytes))
    _patch_httpx_client(monkeypatch, transport)

    image_path = _make_reference_image(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["comfyui_single_shot_smoke.py", "--base-url", "http://fake-comfyui:8188",
         "--image", image_path, "--prompt", "test", "--duration-seconds", "3.0", "--fps", "24",
         "--output-dir", str(tmp_path / "out")],
    )
    asyncio.run(smoke._main())
    stdout = capsys.readouterr().out
    assert "resolved fps:        24" in stdout
    # 3.0s * 24fps = 72 raw frames -> nearest 4n+1 is 73.
    assert "resolved frame count: 73" in stdout
