"""Regression tests for Sprint 4 Prompt 10's four audited issues:

1. render_single_shot CLI no longer silently overrides the validated
   Wan animation resolution (832x480) with a literal 1280x704 default.
2. RenderPipeline can receive explicit FLUX image dimensions, and
   FightVideoPipeline/SingleShotAcceptanceRunner both resolve and pass
   through the same settings.comfyui_image_default_width/height policy
   — while the generic ImageRouter/ImageProvider/RenderPipeline
   contract remains unchanged when no override is given.
3. (Documentation-only; no dedicated runtime test — see the audit
   report for exactly what was corrected in workflows/README.md,
   VALIDATED-SETTINGS.md, and single_shot_acceptance.py's docstring.)
4. scripts/render_single_shot.py preflights real (comfyui) acceptance
   runs before the first expensive generation call, and is a complete
   no-op for mock runs.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real
ComfyUI server or GPU is ever contacted.
"""

import asyncio
import subprocess
import sys

import httpx

from core.animation_providers.comfyui import _duration_to_frame_count
from core.config import get_settings
from core.image_router import ImageProviderRegistry, ImageRouter, MockImageProvider
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.render_pipeline import RenderPipeline
from tests.test_single_shot_acceptance import _preferences, _runner, _sample_pgn

RENDER_SINGLE_SHOT = "scripts/render_single_shot.py"


# --- Issue 1: CLI no longer forces Wan to 1280x704 by default --------------


def test_cli_help_no_longer_offers_a_width_flag_defaulting_to_1280():
    """The old --width/--height flags (literal 1280/704 defaults) are
    gone, replaced by --animation-width/--animation-height defaulting
    to None (letting the Wan policy resolve)."""
    result = subprocess.run([sys.executable, RENDER_SINGLE_SHOT, "--help"], capture_output=True, text=True, check=True)
    assert "--width " not in result.stdout  # the old flag name itself is gone
    assert "--animation-width" in result.stdout
    assert "--animation-height" in result.stdout
    assert "default: 1280" not in result.stdout.lower()
    assert "default: 704" not in result.stdout.lower()


def test_cli_default_animation_resolution_is_832x480_not_1280x704(tmp_path):
    """The core Issue 1 regression: running with no explicit
    --animation-width/--animation-height must produce an 832x480 clip,
    not 1280x704."""
    result = subprocess.run(
        [sys.executable, RENDER_SINGLE_SHOT, "--sample", "--shot-index", "0",
         "--max-animation-seconds", "2"],
        capture_output=True, text=True, check=True, cwd=".",
    )
    assert result.returncode == 0
    import re

    match = re.search(r"video path:\s*(\S+)", result.stdout)
    assert match is not None
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", match.group(1)],
        capture_output=True, text=True, check=True,
    )
    width, height = probe.stdout.strip().split(",")
    assert (int(width), int(height)) == (832, 480)

    # Clean up the artifacts this subprocess run created.
    import shutil

    shutil.rmtree("generated_animations", ignore_errors=True)
    shutil.rmtree("generated_images", ignore_errors=True)
    shutil.rmtree("storage", ignore_errors=True)


def test_cli_explicit_animation_resolution_override_still_works(tmp_path):
    """An explicit override must still take effect — the fix removes
    the silent default, not the ability to override intentionally."""
    result = subprocess.run(
        [sys.executable, RENDER_SINGLE_SHOT, "--sample", "--shot-index", "0",
         "--max-animation-seconds", "2", "--animation-width", "640", "--animation-height", "480"],
        capture_output=True, text=True, check=True,
    )
    assert result.returncode == 0
    import re

    match = re.search(r"video path:\s*(\S+)", result.stdout)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", match.group(1)],
        capture_output=True, text=True, check=True,
    )
    width, height = probe.stdout.strip().split(",")
    assert (int(width), int(height)) == (640, 480)

    import shutil

    shutil.rmtree("generated_animations", ignore_errors=True)
    shutil.rmtree("generated_images", ignore_errors=True)
    shutil.rmtree("storage", ignore_errors=True)


# --- Issue 2: FLUX image resolution policy reaches RenderPipeline ----------


def test_render_pipeline_accepts_explicit_width_height(tmp_path):
    from core.ai_router import TemplateProvider
    from products.chess2fight.orchestrator import FightOrchestrator

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path / "images")))
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry),
        asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )

    generate_response = asyncio.run(
        FightOrchestrator(TemplateProvider()).generate_fight(_sample_pgn(), _preferences())
    )

    output = asyncio.run(
        render_pipeline.render(generate_response.prompted_timeline, "test_fight", width=1280, height=704)
    )
    from PIL import Image

    with Image.open(output.frames[0].frame_path) as img:
        assert img.size == (1280, 704)


def test_render_pipeline_default_unchanged_when_no_width_height_given(tmp_path):
    """The generic ImageRouter/ImageProvider/RenderPipeline contract
    must remain exactly as before when no override is given — this is
    the explicit design constraint the task calls out."""
    from core.ai_router import TemplateProvider
    from products.chess2fight.orchestrator import FightOrchestrator

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path / "images")))
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry),
        asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )

    generate_response = asyncio.run(
        FightOrchestrator(TemplateProvider()).generate_fight(_sample_pgn(), _preferences())
    )

    output = asyncio.run(
        render_pipeline.render(generate_response.prompted_timeline, "test_fight")
    )  # no width/height at all
    from PIL import Image

    with Image.open(output.frames[0].frame_path) as img:
        assert img.size == (1024, 1024)  # ImageProvider's own generic default, unchanged


def test_full_pipeline_flux_keyframe_uses_configured_image_resolution(tmp_path, monkeypatch):
    """FightVideoPipeline.run() must resolve and pass through
    settings.comfyui_image_default_width/height — the actual generated
    keyframe must reflect it, not the generic 1024x1024."""
    from core.ai_router import TemplateProvider
    from core.animation_router import AnimationProviderRegistry, AnimationRouter, MockAnimationProvider
    from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
    from products.chess2fight.rendering.pipeline import FightVideoPipeline
    from products.chess2fight.rendering.video_builder import VideoBuilder

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

    response = asyncio.run(pipeline.run(_sample_pgn(), _preferences()))
    from PIL import Image

    settings = get_settings()
    with Image.open(response.frames[0].frame_path) as img:
        assert img.size == (settings.comfyui_image_default_width, settings.comfyui_image_default_height)


def test_single_shot_acceptance_and_full_pipeline_use_the_same_image_policy(tmp_path):
    """Explicit compatibility check the task calls out by name:
    FightVideoPipeline and SingleShotAcceptanceRunner must resolve the
    identical image policy, not two different special-cased values."""
    from core.ai_router import TemplateProvider
    from core.animation_router import AnimationProviderRegistry, AnimationRouter, MockAnimationProvider
    from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
    from products.chess2fight.rendering.pipeline import FightVideoPipeline
    from products.chess2fight.rendering.single_shot_acceptance import SingleShotAcceptanceRunner
    from products.chess2fight.rendering.video_builder import VideoBuilder

    def make_pipelines(subdir):
        image_registry = ImageProviderRegistry()
        image_registry.register("mock", lambda: MockImageProvider(output_dir=str(tmp_path / subdir / "images")))
        render_pipeline = RenderPipeline(
            image_router=ImageRouter(registry=image_registry),
            asset_manager=AssetManager(storage_root=str(tmp_path / subdir / "storage")),
        )
        anim_registry = AnimationProviderRegistry()
        anim_registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path / subdir / "animations")))
        animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))
        return render_pipeline, animation_pipeline

    full_render, full_animation = make_pipelines("full")
    full_pipeline = FightVideoPipeline(
        ai_provider=None, render_pipeline=full_render, animation_pipeline=full_animation,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "full" / "storage")),
    )
    from core.ai_router import TemplateProvider as _TP

    full_pipeline._orchestrator._ai_provider = _TP()  # ensure a working provider regardless of constructor default
    full_response = asyncio.run(full_pipeline.run(_sample_pgn(), _preferences()))

    single_render, single_animation = make_pipelines("single")
    runner = SingleShotAcceptanceRunner(TemplateProvider(), render_pipeline=single_render, animation_pipeline=single_animation)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0))
    single_result = asyncio.run(runner.execute(plan))

    from PIL import Image

    with Image.open(full_response.frames[0].frame_path) as img_full:
        with Image.open(single_result.image_path) as img_single:
            assert img_full.size == img_single.size


# --- Issue 4: preflight ------------------------------------------------------


def test_preflight_is_a_complete_noop_for_mock(monkeypatch):
    from scripts import render_single_shot as script

    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "mock")
    monkeypatch.setattr(settings, "animation_provider", "mock")
    problems, warnings = asyncio.run(script._preflight_check(settings))
    assert problems == []
    assert warnings == []


def test_preflight_fails_on_unreachable_comfyui(monkeypatch):
    from scripts import render_single_shot as script

    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "comfyui_base_url", "http://this-host-does-not-exist.invalid:1")
    problems, warnings = asyncio.run(script._preflight_check(settings))
    assert any("not reachable" in p.lower() for p in problems)


def test_preflight_fails_on_missing_workflow_file(monkeypatch, tmp_path):
    from scripts import render_single_shot as script

    settings = get_settings()
    monkeypatch.setattr(settings, "animation_provider", "comfyui")
    monkeypatch.setattr(settings, "comfyui_workflow_path", str(tmp_path / "does_not_exist.json"))

    class _MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, json={"system": {}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        script.httpx, "AsyncClient",
        lambda *a, **kw: real_client(*a, **{**kw, "transport": _MockTransport()}),
    )
    problems, warnings = asyncio.run(script._preflight_check(settings))
    assert any("workflow file not found" in p.lower() for p in problems)


def test_preflight_never_blocks_generation_that_never_gets_attempted_when_ffprobe_missing(monkeypatch):
    from scripts import render_single_shot as script

    settings = get_settings()
    monkeypatch.setattr(settings, "animation_provider", "comfyui")
    monkeypatch.setattr(script.shutil, "which", lambda name: None)
    problems, warnings = asyncio.run(script._preflight_check(settings))
    assert any("ffprobe" in p.lower() for p in problems)
    assert any("ffmpeg" in p.lower() for p in problems)


def test_cli_fails_before_generation_when_preflight_fails(monkeypatch):
    """End-to-end confirmation via the actual CLI: a real (comfyui)
    run against an unreachable server must exit non-zero WITHOUT ever
    reaching runner.execute() — i.e. before generation."""
    result = subprocess.run(
        [sys.executable, RENDER_SINGLE_SHOT, "--sample", "--shot-index", "0", "--max-animation-seconds", "2"],
        capture_output=True, text=True,
        env={
            **__import__("os").environ, "IMAGE_PROVIDER": "comfyui", "ANIMATION_PROVIDER": "comfyui",
            "COMFYUI_BASE_URL": "http://localhost:1",
        },
    )
    assert result.returncode == 1
    assert "preflight check failed" in result.stderr.lower()
    assert "rendering via" not in result.stdout.lower()  # never reached the generation step


def test_skip_preflight_flag_bypasses_the_check(monkeypatch):
    """--skip-preflight must actually bypass the check — confirmed by
    the run reaching (and failing inside) the real execute() call
    instead of the preflight step."""
    result = subprocess.run(
        [sys.executable, RENDER_SINGLE_SHOT, "--sample", "--shot-index", "0", "--max-animation-seconds", "2",
         "--skip-preflight"],
        capture_output=True, text=True,
        env={
            **__import__("os").environ, "IMAGE_PROVIDER": "comfyui", "ANIMATION_PROVIDER": "comfyui",
            "COMFYUI_BASE_URL": "http://localhost:1",
        },
    )
    assert result.returncode == 1
    assert "preflight check failed" not in result.stderr.lower()
    assert "error during rendering" in result.stderr.lower()  # failed later, inside execute()


# --- General regressions the task explicitly asks for ------------------------


def test_generic_animation_instruction_semantics_unchanged():
    """AnimationInstruction's own generic contract (default
    animation_type, default width/height) must remain exactly as
    established in Sprint 4 Prompt 1/8 — untouched by any of this
    task's Chess2Fight-specific policy fixes."""
    from core.animation_router import AnimationInstruction, AnimationType

    instruction = AnimationInstruction(
        shot_id="s1", source_image_path="/tmp/x.png", prompt="test",
        duration_seconds=2.0, camera_motion="static", subject_motion="test",
    )
    assert instruction.animation_type == AnimationType.IMAGE_TO_VIDEO
    assert instruction.width == 1024
    assert instruction.height == 1024


def test_max_animation_seconds_still_does_not_mutate_original_shot(tmp_path):
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0, max_animation_seconds=2.0))
    asyncio.run(runner.execute(plan, width=832, height=480))
    assert plan.shot.duration_seconds == 7.75


def test_eight_fps_frame_count_matches_current_wan_runtime():
    assert _duration_to_frame_count(2.0, 8) == 17


def test_exactly_one_image_and_one_animation_call_in_acceptance_execution(tmp_path):
    image_calls = []
    animation_calls = []

    from core.animation_router import AnimationProvider, AnimationProviderRegistry, AnimationResult, AnimationRouter
    from core.image_router import ImageGenerationResult, ImageProvider
    from products.chess2fight.rendering.animation_pipeline import AnimationPipeline

    class _SpyImageProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            image_calls.append(prompt)
            from PIL import Image

            path = tmp_path / "spy.png"
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=str(path), provider="_SpyImageProvider", prompt=prompt,
                width=width, height=height, generation_time_seconds=0.0,
            )

    class _SpyAnimationProvider(AnimationProvider):
        async def generate_animation(self, instruction):
            animation_calls.append(instruction.shot_id)
            return AnimationResult(
                success=True, shot_id=instruction.shot_id, provider="_SpyAnimationProvider",
                video_path=str(tmp_path / "spy.mp4"), duration_seconds=instruction.duration_seconds,
            )

    (tmp_path / "spy.mp4").write_bytes(b"fake")

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _SpyImageProvider())
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry),
        asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: _SpyAnimationProvider())
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))

    from core.ai_router import TemplateProvider
    from products.chess2fight.rendering.single_shot_acceptance import SingleShotAcceptanceRunner

    runner = SingleShotAcceptanceRunner(TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), shot_index=0))
    asyncio.run(runner.execute(plan))

    assert len(image_calls) == 1
    assert len(animation_calls) == 1
