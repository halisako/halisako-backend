"""Regression tests for Sprint 4 Prompt 10.1's two fixes:

1. A confirmed-missing required model (via a successfully fetched and
   parsed /object_info response) is now a hard preflight problem, not
   a warning — a paid acceptance run must not proceed into a doomed
   generation. A genuinely unparseable/unreachable /object_info
   response remains a warning, not a false confirmed-missing failure.
2. Production Chess2Fight FPS (RenderVideoRequest.fps default,
   FightVideoPipeline.run()'s own fps default) now resolves from
   settings.comfyui_default_fps (8) instead of a stale literal 24 that
   had drifted out of sync with it since Sprint 4 Prompt 8.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted. Mock /object_info responses below use
exactly the shape the implementation's own code comment documents
(combo/dropdown values nested under input.required.<field>[0] as a
list) — not an invented structure.
"""

import asyncio

import httpx

from core.config import get_settings
from scripts import render_single_shot as script

_REAL_ASYNC_CLIENT = httpx.AsyncClient  # captured once, before any test can monkeypatch it


def _patch_client(monkeypatch, transport: httpx.AsyncBaseTransport) -> None:
    monkeypatch.setattr(
        script.httpx, "AsyncClient",
        lambda *a, **kw: _REAL_ASYNC_CLIENT(*a, **{**kw, "transport": transport}),
    )


class _MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handlers: dict):
        self._handlers = handlers

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        key = f"{request.method} {request.url.path}"
        handler = self._handlers.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": f"no handler for {key}"})
        return handler(request)


def _object_info_response(node_type: str, available_filenames: list[str]) -> dict:
    """Exactly the shape core/scripts/render_single_shot.py's own
    _preflight_check() comment documents: combo (dropdown) values
    nested under input.required.<field>[0] as a list."""
    return {node_type: {"input": {"required": {"some_field": [available_filenames, {}]}}}}


def _handlers_with_all_models_present() -> dict:
    def system_stats(request):
        return httpx.Response(200, json={"system": {}})

    return {
        "GET /system_stats": system_stats,
        "GET /object_info/UNETLoader": lambda r: httpx.Response(
            200, json=_object_info_response("UNETLoader", ["flux-2-klein-4b.safetensors", "wan2.2_ti2v_5B_fp16.safetensors"])
        ),
        "GET /object_info/CLIPLoader": lambda r: httpx.Response(
            200, json=_object_info_response("CLIPLoader", ["qwen_3_4b.safetensors", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"])
        ),
        "GET /object_info/VAELoader": lambda r: httpx.Response(
            200, json=_object_info_response("VAELoader", ["flux2-vae.safetensors", "wan2.2_vae.safetensors"])
        ),
    }


# --- Fix 1: confirmed-missing models are hard problems ----------------------


def test_confirmed_missing_flux_diffusion_model_is_a_hard_problem(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _handlers_with_all_models_present()
    handlers["GET /object_info/UNETLoader"] = lambda r: httpx.Response(
        200, json=_object_info_response("UNETLoader", ["some_other_model.safetensors"])
    )
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(script._preflight_check(settings))
    assert any("flux-2-klein-4b.safetensors" in p for p in problems)
    assert warnings == []


def test_confirmed_missing_wan_diffusion_model_is_a_hard_problem(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "mock")
    monkeypatch.setattr(settings, "animation_provider", "comfyui")

    handlers = _handlers_with_all_models_present()
    handlers["GET /object_info/UNETLoader"] = lambda r: httpx.Response(
        200, json=_object_info_response("UNETLoader", ["some_other_model.safetensors"])
    )
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(script._preflight_check(settings))
    assert any("wan2.2_ti2v_5B_fp16.safetensors" in p for p in problems)
    assert warnings == []


def test_confirmed_missing_text_encoder_is_a_hard_problem(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _handlers_with_all_models_present()
    handlers["GET /object_info/CLIPLoader"] = lambda r: httpx.Response(
        200, json=_object_info_response("CLIPLoader", ["some_other_encoder.safetensors"])
    )
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(script._preflight_check(settings))
    assert any("qwen_3_4b.safetensors" in p for p in problems)
    assert warnings == []


def test_confirmed_missing_vae_is_a_hard_problem(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "mock")
    monkeypatch.setattr(settings, "animation_provider", "comfyui")

    handlers = _handlers_with_all_models_present()
    handlers["GET /object_info/VAELoader"] = lambda r: httpx.Response(
        200, json=_object_info_response("VAELoader", ["some_other_vae.safetensors"])
    )
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(script._preflight_check(settings))
    assert any("wan2.2_vae.safetensors" in p for p in problems)
    assert warnings == []


def test_malformed_object_info_produces_warning_not_false_confirmed_missing(monkeypatch):
    """A response missing the expected node_type key entirely (a
    genuinely unparseable shape, not an empty combo list) must be a
    warning, never misread as a confirmed-missing model."""
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _handlers_with_all_models_present()
    handlers["GET /object_info/UNETLoader"] = lambda r: httpx.Response(200, json={"totally": "unexpected shape"})
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(script._preflight_check(settings))
    assert problems == []
    assert any("could not verify" in w.lower() for w in warnings)


def test_object_info_with_no_recognizable_combo_field_is_a_warning(monkeypatch):
    """A response that has the node_type key but no list-shaped combo
    field at all — also genuinely unparseable, also a warning."""
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _handlers_with_all_models_present()
    handlers["GET /object_info/UNETLoader"] = lambda r: httpx.Response(
        200, json={"UNETLoader": {"input": {"required": {"some_field": ["not_a_combo_list"]}}}}
    )
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(script._preflight_check(settings))
    assert problems == []
    assert any("could not verify" in w.lower() for w in warnings)


def test_all_models_present_produces_no_problems_and_no_warnings(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "comfyui")
    _patch_client(monkeypatch, _MockTransport(_handlers_with_all_models_present()))

    problems, warnings = asyncio.run(script._preflight_check(settings))
    assert problems == []
    assert warnings == []


def test_cli_does_not_call_execute_when_a_model_is_confirmed_missing(monkeypatch, tmp_path):
    """End-to-end: the CLI's own generation call must never happen
    when a model is confirmed missing."""
    import sys

    settings = get_settings()
    monkeypatch.setattr(sys, "argv", [
        "render_single_shot.py", "--sample", "--shot-index", "0", "--max-animation-seconds", "2",
    ])
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _handlers_with_all_models_present()
    handlers["GET /object_info/UNETLoader"] = lambda r: httpx.Response(
        200, json=_object_info_response("UNETLoader", ["some_other_model.safetensors"])
    )
    _patch_client(monkeypatch, _MockTransport(handlers))

    execute_called = {"value": False}

    async def _exploding_execute(*args, **kwargs):
        execute_called["value"] = True
        raise AssertionError("execute() must never be called when a model is confirmed missing")

    from products.chess2fight.rendering.single_shot_acceptance import SingleShotAcceptanceRunner

    monkeypatch.setattr(SingleShotAcceptanceRunner, "execute", _exploding_execute)

    exit_code = asyncio.run(script._main())
    assert exit_code == 1
    assert execute_called["value"] is False


# --- Fix 2: production FPS default matches validated policy -----------------


def test_render_video_request_default_fps_resolves_to_current_settings():
    from products.chess2fight.rendering.pipeline import RenderVideoRequest

    request = RenderVideoRequest(pgn="dummy")
    assert request.fps == get_settings().comfyui_default_fps
    assert request.fps == 8


def test_fight_video_pipeline_run_with_no_fps_override_uses_8(tmp_path):
    from core.ai_router import TemplateProvider
    from core.animation_router import AnimationProviderRegistry, AnimationRouter, MockAnimationProvider
    from core.image_router import ImageProviderRegistry, ImageRouter, MockImageProvider
    from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
    from products.chess2fight.rendering.asset_manager import AssetManager
    from products.chess2fight.rendering.pipeline import FightVideoPipeline
    from products.chess2fight.rendering.render_pipeline import RenderPipeline
    from products.chess2fight.rendering.video_builder import VideoBuilder
    from tests.test_single_shot_acceptance import _preferences, _sample_pgn

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path / "images")))
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry),
        asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path / "animations")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    pipeline = FightVideoPipeline(
        ai_provider=TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )

    response = asyncio.run(pipeline.run(_sample_pgn(), _preferences()))  # no fps override at all

    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", response.video_path],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "8/1"


def test_explicit_fps_override_still_works(tmp_path):
    from core.ai_router import TemplateProvider
    from core.animation_router import AnimationProviderRegistry, AnimationRouter, MockAnimationProvider
    from core.image_router import ImageProviderRegistry, ImageRouter, MockImageProvider
    from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
    from products.chess2fight.rendering.asset_manager import AssetManager
    from products.chess2fight.rendering.pipeline import FightVideoPipeline
    from products.chess2fight.rendering.render_pipeline import RenderPipeline
    from products.chess2fight.rendering.video_builder import VideoBuilder
    from tests.test_single_shot_acceptance import _preferences, _sample_pgn

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path / "images")))
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry),
        asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path / "animations")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    pipeline = FightVideoPipeline(
        ai_provider=TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )

    response = asyncio.run(pipeline.run(_sample_pgn(), _preferences(), fps=24))

    import subprocess

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", response.video_path],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "24/1"


def test_single_shot_acceptance_and_production_pipeline_resolve_the_same_default_fps(tmp_path):
    from tests.test_single_shot_acceptance import _preferences, _runner, _sample_pgn

    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0))
    assert plan.fps == get_settings().comfyui_default_fps == 8


def test_wan_frame_count_remains_17_for_the_2_second_acceptance_baseline():
    from core.animation_providers.comfyui import _duration_to_frame_count

    assert _duration_to_frame_count(2.0, get_settings().comfyui_default_fps) == 17
