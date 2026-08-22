"""Verifies the actual Chess2Fight production wiring end to end —
Sprint 4 Prompt 9, Task 9:

    rendered shot still -> AnimationPipeline -> AnimationInstruction ->
    AnimationRouter -> ComfyUIAnimationProvider -> AnimatedShot

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TEST — ComfyUI's HTTP
boundary is mocked; no real server or GPU involved. What makes this
different from tests/test_animation_pipeline.py (which only exercises
MockAnimationProvider) and tests/test_flux_to_wan_handoff.py /
tests/test_comfyui_animation_provider.py (which call
ComfyUIAnimationProvider directly, bypassing AnimationPipeline) is
that this file runs the REAL AnimationPipeline -> AnimationRouter ->
ComfyUIAnimationProvider chain together, against a REAL rendered shot
from the REAL cinematic pipeline — proving the actual production
wiring, not each piece in isolation.
"""

import asyncio
import json

import httpx

from core.ai_router import TemplateProvider
from core.animation_providers.comfyui import ComfyUIAnimationProvider, _duration_to_frame_count
from core.animation_router import AnimationProviderRegistry, AnimationRouter
from core.config import get_settings
from core.image_router import ImageProviderRegistry, ImageRouter, MockImageProvider
from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.battle_mode_engine import generate_battle_mode_intelligence
from products.chess2fight.cinematic.prompt_generator import generate_prompts
from products.chess2fight.cinematic.scene_composer import compose_scene
from products.chess2fight.cinematic.timeline_engine import generate_shot_timeline
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.narrative_generator import NarrativeGenerator
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.render_pipeline import RenderPipeline
from products.chess2fight.schemas import BattleMode
from products.chess2fight.style_engine import generate_style_profile

SCHOLARS_MATE_PGN = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)

I2V_WORKFLOW_PATH = "products/chess2fight/rendering/workflows/wan22_i2v_5b.json"

_REAL_ASYNC_CLIENT = httpx.AsyncClient  # captured once, before any test can monkeypatch it


async def _build_render_output(tmp_path, pgn=SCHOLARS_MATE_PGN, style="anime"):
    """Runs the real pipeline through RenderPipeline, returning
    (render_output, prompted_timeline) — same pattern already
    established in tests/test_animation_pipeline.py."""
    analysis = analyze_game(pgn)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    profile = generate_style_profile(battle, combat, style)
    battle_mode = generate_battle_mode_intelligence(BattleMode.DUEL, combat, battle)
    story = await NarrativeGenerator(TemplateProvider()).generate(analysis, combat, battle, profile, battle_mode)
    timeline = generate_shot_timeline(battle, story)
    composed = compose_scene(timeline, battle, profile, battle_mode)
    prompted = generate_prompts(composed)

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path / "images")))
    image_router = ImageRouter(registry=image_registry)
    asset_manager = AssetManager(storage_root=str(tmp_path / "storage"))
    render_pipeline = RenderPipeline(image_router=image_router, asset_manager=asset_manager)

    render_output = await render_pipeline.render(prompted, "test_fight")
    return render_output, prompted


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
    import core.animation_providers.comfyui as module

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", _factory)


def _make_real_mp4(tmp_path, width=832, height=480, duration=2.125, fps=8) -> bytes:
    import subprocess

    from PIL import Image

    frame_dir = tmp_path / "wiring_mp4_frame"
    frame_dir.mkdir(exist_ok=True)
    Image.new("RGB", (width, height), color=(80, 20, 150)).save(frame_dir / "frame.png")
    output = tmp_path / "wiring_generated.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loop", "1", "-framerate", str(fps), "-i", str(frame_dir / "frame.png"),
            "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
        ],
        capture_output=True, check=True,
    )
    return output.read_bytes()


def _comfyui_success_handlers(video_bytes: bytes, prompt_id: str = "wiring-test-123"):
    def upload(request):
        return httpx.Response(200, json={"name": "uploaded_shot.png", "subfolder": "", "type": "input"})

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

    return {
        "POST /upload/image": upload, "POST /prompt": queue,
        f"GET /history/{prompt_id}": history, "GET /view": view,
    }


def test_rendered_shot_becomes_a_correct_i2v_instruction_through_the_real_wiring(tmp_path, monkeypatch):
    """The core Task 9 property, checked in one place: a real rendered
    shot, driven through the real AnimationPipeline -> AnimationRouter
    -> ComfyUIAnimationProvider chain, must produce a request that
    carries the correct source-image path, Wan-compatible geometry,
    the correctly-resolved frame count, and confirms the ComfyUI
    provider (not mock) was actually the one dispatched to."""
    video_bytes = _make_real_mp4(tmp_path)
    transport = _MockTransport(_comfyui_success_handlers(video_bytes))
    _patch_httpx_client(monkeypatch, transport)

    render_output, prompted = asyncio.run(_build_render_output(tmp_path))
    first_frame = render_output.frames[0]
    first_shot = next(s for s in prompted.shots if s.shot_id == first_frame.metadata.shot_id)

    anim_registry = AnimationProviderRegistry()
    anim_registry.register(
        "comfyui",
        lambda: ComfyUIAnimationProvider(
            base_url="http://fake-comfyui:8188", workflow_path=I2V_WORKFLOW_PATH,
            output_dir=str(tmp_path / "animations"),
        ),
    )
    animation_router = AnimationRouter(registry=anim_registry)
    monkeypatch.setattr(get_settings(), "animation_provider", "comfyui")

    animation_pipeline = AnimationPipeline(animation_router=animation_router)

    settings = get_settings()
    animation_output = asyncio.run(
        animation_pipeline.animate(
            render_output, prompted,
            width=settings.comfyui_animation_default_width,
            height=settings.comfyui_animation_default_height,
        )
    )

    # 1. Correct provider selection — the real ComfyUI provider ran,
    # not a fallback to mock.
    assert animation_router.active_provider_name() == "comfyui"
    animated_shot = animation_output.animated_shots[0]
    assert animated_shot.shot_id == first_shot.shot_id

    # 2. Correct source-image path reached the provider — inspect the
    # actual /prompt request body, not just the final result.
    queue_requests = [r for r in transport.requests if r.url.path == "/prompt"]
    assert len(queue_requests) >= 1
    submitted_workflow = json.loads(queue_requests[0].content)["prompt"]
    assert submitted_workflow["56"]["inputs"]["image"] == "uploaded_shot.png"  # the uploaded name, not a local path
    upload_requests = [r for r in transport.requests if r.url.path == "/upload/image"]
    assert len(upload_requests) >= 1

    # 3. Wan-compatible geometry — 832x480, not the generic 1024x1024.
    assert submitted_workflow["55"]["inputs"]["width"] == 832
    assert submitted_workflow["55"]["inputs"]["height"] == 480

    # 4. Duration/frame resolution — the shot's own real duration,
    # converted to a valid 4n+1 frame count at the configured fps.
    expected_frames = _duration_to_frame_count(first_shot.duration_seconds, settings.comfyui_default_fps)
    assert submitted_workflow["55"]["inputs"]["length"] == expected_frames
    assert (expected_frames - 1) % 4 == 0

    # 5. The motion/positive prompt reaching the provider is the
    # shot's own real image_prompt, not a placeholder.
    assert submitted_workflow["6"]["inputs"]["text"] == first_shot.image_prompt

    # 6. The resulting clip is a real, valid artifact.
    assert animated_shot.video_path is not None


def test_multiple_shots_each_get_correctly_wired_independently(tmp_path, monkeypatch):
    """A second, independent check across more than one shot — proves
    the wiring isn't coincidentally correct only for the first shot in
    the timeline."""
    video_bytes = _make_real_mp4(tmp_path)
    call_count = {"n": 0}

    def counting_queue(request):
        call_count["n"] += 1
        return httpx.Response(200, json={"prompt_id": f"wiring-multi-{call_count['n']}", "node_errors": {}})

    handlers = _comfyui_success_handlers(video_bytes)

    def make_history(prompt_id):
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

        return history

    handlers["POST /prompt"] = counting_queue
    for i in range(1, 9):
        handlers[f"GET /history/wiring-multi-{i}"] = make_history(f"wiring-multi-{i}")

    transport = _MockTransport(handlers)
    _patch_httpx_client(monkeypatch, transport)

    render_output, prompted = asyncio.run(_build_render_output(tmp_path))

    anim_registry = AnimationProviderRegistry()
    anim_registry.register(
        "comfyui",
        lambda: ComfyUIAnimationProvider(
            base_url="http://fake-comfyui:8188", workflow_path=I2V_WORKFLOW_PATH,
            output_dir=str(tmp_path / "animations"),
        ),
    )
    monkeypatch.setattr(get_settings(), "animation_provider", "comfyui")
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    settings = get_settings()
    animation_output = asyncio.run(
        animation_pipeline.animate(
            render_output, prompted,
            width=settings.comfyui_animation_default_width,
            height=settings.comfyui_animation_default_height,
        )
    )

    assert animation_output.shot_count == len(prompted.shots)
    queue_requests = [r for r in transport.requests if r.url.path == "/prompt"]
    assert len(queue_requests) == len(prompted.shots)

    # Each shot's own duration produced its own (possibly different) frame count.
    submitted_lengths = [json.loads(r.content)["prompt"]["55"]["inputs"]["length"] for r in queue_requests]
    expected_lengths = [
        _duration_to_frame_count(shot.duration_seconds, settings.comfyui_default_fps) for shot in prompted.shots
    ]
    assert submitted_lengths == expected_lengths
