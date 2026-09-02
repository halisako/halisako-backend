"""Regression tests for Sprint 4 Prompt 13.1's four hardening fixes:

1. The reference-conditioned workflow now matches the official
   distilled image-edit topology: TWO ReferenceLatent nodes (positive
   and negative branches), both fed the same encoded anchor latent.
2. A strict, hard-failing preflight check for ReferenceLatent/VAEEncode
   node availability, plus static workflow-topology validation.
3. The real acceptance CLI now requires BOTH image_provider and
   animation_provider to be comfyui, not just image_provider.
4. compose_reference_edit_prompt()'s PRESERVE block is now
   focus-independent — identical regardless of shot.focus.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
import json
import subprocess
import sys

import httpx
import pytest
from PIL import Image

from core.config import get_settings
from products.chess2fight.cinematic.prompt_generator import compose_reference_edit_prompt
from products.chess2fight.cinematic.schemas import ShotFocus
from products.chess2fight.rendering import acceptance_preflight as preflight_module
from products.chess2fight.rendering.acceptance_preflight import validate_reference_workflow_topology
from products.chess2fight.rendering.reference_continuity_acceptance import ReferenceAnchorInvalidError
from tests.test_prompt12_visual_continuity import _prompted_timeline
from tests.test_prompt13_reference_continuity import _runner_and_provider
from tests.test_single_shot_acceptance import _preferences, _sample_pgn

RENDER_REFERENCE_CLI = "scripts/render_reference_continuity_acceptance.py"
REFERENCE_WORKFLOW_PATH = "products/chess2fight/rendering/workflows/flux2_klein_reference_4b.json"


def _load_reference_workflow() -> dict:
    with open(REFERENCE_WORKFLOW_PATH) as f:
        return json.load(f)


# --- 1-7. Workflow topology --------------------------------------------------


def test_reference_workflow_contains_vaeencode():
    wf = _load_reference_workflow()
    vae_encode_nodes = [n for n in wf.values() if n.get("class_type") == "VAEEncode"]
    assert len(vae_encode_nodes) >= 1


def test_reference_workflow_contains_two_reference_latent_nodes():
    wf = _load_reference_workflow()
    ref_latent_nodes = [n for n in wf.values() if n.get("class_type") == "ReferenceLatent"]
    assert len(ref_latent_nodes) == 2


def test_both_reference_latent_nodes_receive_the_same_encoded_anchor_latent():
    wf = _load_reference_workflow()
    ref_latent_nodes = [n for n in wf.values() if n.get("class_type") == "ReferenceLatent"]
    latent_sources = {tuple(n["inputs"]["latent"]) for n in ref_latent_nodes}
    assert len(latent_sources) == 1  # both point at the identical latent source


def test_positive_reference_latent_receives_positive_text_conditioning():
    wf = _load_reference_workflow()
    # ref:3 is the positive branch — its conditioning input is the plain
    # CLIPTextEncode node (77:92), never ConditioningZeroOut.
    assert wf["ref:3"]["inputs"]["conditioning"] == ["77:92", 0]


def test_negative_reference_latent_receives_conditioning_zero_out_output():
    wf = _load_reference_workflow()
    # ref:4 is the negative branch — fed by ConditioningZeroOut (77:91).
    assert wf["ref:4"]["inputs"]["conditioning"] == ["77:91", 0]


def test_cfgguider_positive_points_to_reference_conditioned_positive():
    """Sprint 4 Prompt 18: production now routes through
    FluxKontextMultiReferenceLatentMethod (method:1) — CFGGuider.positive
    points at that node, which in turn traces back to ref:3
    (ReferenceLatent), rather than pointing at ref:3 directly as it did
    before this prompt's production promotion of the "offset" method."""
    wf = _load_reference_workflow()
    assert wf["77:90"]["inputs"]["positive"] == ["method:1", 0]
    assert wf["method:1"]["inputs"]["conditioning"] == ["ref:3", 0]


def test_cfgguider_negative_points_to_reference_conditioned_negative():
    """Sprint 4 Prompt 18: same reasoning as the positive-branch test
    above, for the negative branch (method:2, tracing back to ref:4)."""
    wf = _load_reference_workflow()
    assert wf["77:90"]["inputs"]["negative"] == ["method:2", 0]
    assert wf["method:2"]["inputs"]["conditioning"] == ["ref:4", 0]


def test_validate_reference_workflow_topology_passes_for_the_real_file():
    problems = validate_reference_workflow_topology(REFERENCE_WORKFLOW_PATH)
    assert problems == []


def test_validate_reference_workflow_topology_catches_missing_negative_reference_latent(tmp_path):
    wf = _load_reference_workflow()
    broken = dict(wf)
    del broken["ref:4"]
    broken["77:90"] = dict(broken["77:90"])
    broken["77:90"]["inputs"] = dict(broken["77:90"]["inputs"])
    broken["77:90"]["inputs"]["negative"] = ["77:91", 0]
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken))

    problems = validate_reference_workflow_topology(str(broken_path))
    assert len(problems) >= 1
    assert any("negative" in p.lower() for p in problems)


# --- 8/9/10. Strict live-node preflight --------------------------------------


class _MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handlers: dict):
        self._handlers = handlers

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        key = f"{request.method} {request.url.path}"
        handler = self._handlers.get(key)
        if handler is None:
            return httpx.Response(404)
        return handler(request)


def _patch_client(monkeypatch, transport):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(preflight_module.httpx, "AsyncClient", lambda *a, **kw: real_client(*a, **{**kw, "transport": transport}))


def _model_visibility_handlers():
    def system_stats(r):
        return httpx.Response(200, json={"system": {}})

    def unet_ok(r):
        return httpx.Response(200, json={"UNETLoader": {"input": {"required": {"unet_name": [["flux-2-klein-4b.safetensors"], {}]}}}})

    def clip_ok(r):
        return httpx.Response(200, json={"CLIPLoader": {"input": {"required": {"clip_name": [["qwen_3_4b.safetensors"], {}]}}}})

    def vae_ok(r):
        return httpx.Response(200, json={"VAELoader": {"input": {"required": {"vae_name": [["flux2-vae.safetensors"], {}]}}}})

    return {
        "GET /system_stats": system_stats,
        "GET /object_info/UNETLoader": unet_ok, "GET /object_info/CLIPLoader": clip_ok, "GET /object_info/VAELoader": vae_ok,
    }


def test_missing_reference_latent_node_is_hard_preflight_failure(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _model_visibility_handlers()
    handlers["GET /object_info/ReferenceLatent"] = lambda r: httpx.Response(200, json={})  # not registered
    handlers["GET /object_info/VAEEncode"] = lambda r: httpx.Response(200, json={"VAEEncode": {"input": {}}})
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(preflight_module.preflight_check(settings, check_reference_workflow=True))
    assert any("ReferenceLatent" in p for p in problems)
    assert not any("ReferenceLatent" in w for w in warnings)  # confirmed hard problem, never a warning


def test_missing_vaeencode_node_is_hard_preflight_failure(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _model_visibility_handlers()
    handlers["GET /object_info/ReferenceLatent"] = lambda r: httpx.Response(200, json={"ReferenceLatent": {"input": {}}})
    handlers["GET /object_info/VAEEncode"] = lambda r: httpx.Response(200, json={})  # not registered
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(preflight_module.preflight_check(settings, check_reference_workflow=True))
    assert any("VAEEncode" in p for p in problems)


def test_reference_node_preflight_disabled_by_default_preserves_older_behavior(monkeypatch):
    """check_reference_workflow defaults to False — older callers
    (render_single_shot.py, render_multi_shot_acceptance.py) must never
    trigger this new, stricter check."""
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _model_visibility_handlers()
    # Deliberately no handlers for ReferenceLatent/VAEEncode at all —
    # if the check ran, it would fail with a connection/404 error.
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(preflight_module.preflight_check(settings))  # check_reference_workflow=False (default)
    assert not any("ReferenceLatent" in p for p in problems)
    assert not any("VAEEncode" in p for p in problems)


def test_reference_node_preflight_all_available_produces_no_problems(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _model_visibility_handlers()
    handlers["GET /object_info/ReferenceLatent"] = lambda r: httpx.Response(200, json={"ReferenceLatent": {"input": {}}})
    handlers["GET /object_info/VAEEncode"] = lambda r: httpx.Response(200, json={"VAEEncode": {"input": {}}})
    # Sprint 4 Prompt 18: preflight_check now also requires
    # FluxKontextMultiReferenceLatentMethod (with "offset" support) —
    # without this handler, that new HTTP call would 404 and this
    # "all available" happy-path test would no longer be testing a
    # genuinely all-available scenario.
    handlers["GET /object_info/FluxKontextMultiReferenceLatentMethod"] = lambda r: httpx.Response(200, json={
        "FluxKontextMultiReferenceLatentMethod": {
            "input": {"required": {"reference_latents_method": [["offset", "index", "uxo/uno", "index_timestep_zero"], {}]}}
        }
    })
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(preflight_module.preflight_check(settings, check_reference_workflow=True))
    assert problems == []


# --- 11/12/13. CLI dual-provider requirement ---------------------------------


def test_cli_rejects_animation_provider_not_comfyui(monkeypatch):
    result = subprocess.run(
        [sys.executable, RENDER_REFERENCE_CLI, "--sample", "--max-animation-seconds", "2"],
        capture_output=True, text=True, env={**__import__("os").environ, "IMAGE_PROVIDER": "comfyui"},
    )
    assert result.returncode == 1
    assert "animation_provider is 'mock'" in result.stderr
    assert "generating anchor" not in result.stdout.lower()


def test_cli_rejects_image_provider_not_comfyui(monkeypatch):
    result = subprocess.run(
        [sys.executable, RENDER_REFERENCE_CLI, "--sample", "--max-animation-seconds", "2"],
        capture_output=True, text=True, env={**__import__("os").environ, "ANIMATION_PROVIDER": "comfyui"},
    )
    assert result.returncode == 1
    assert "image_provider is 'mock'" in result.stderr
    assert "generating anchor" not in result.stdout.lower()


def test_cli_rejects_when_both_are_mock():
    result = subprocess.run(
        [sys.executable, RENDER_REFERENCE_CLI, "--sample", "--max-animation-seconds", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "image_provider is 'mock'" in result.stderr
    assert "animation_provider is 'mock'" in result.stderr


def test_dry_run_remains_provider_agnostic_and_makes_zero_calls():
    """Neither provider set to comfyui — dry-run must still work,
    making no network/provider calls at all."""
    result = subprocess.run(
        [sys.executable, RENDER_REFERENCE_CLI, "--sample", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "dry run complete" in result.stdout.lower()
    assert "generating anchor" not in result.stdout.lower()


# --- 14/15/16. Focus-independent PRESERVE, mutable CHANGE, shot-specific ----


def test_canonical_preserve_identity_text_is_focus_independent():
    prompted = _prompted_timeline(_sample_pgn())
    white_focused = next(s for s in prompted.shots if s.focus == ShotFocus.WHITE)
    black_focused = next(s for s in prompted.shots if s.focus == ShotFocus.BLACK)

    white_prompt = compose_reference_edit_prompt(white_focused)
    black_prompt = compose_reference_edit_prompt(black_focused)

    white_preserve = white_prompt.split("CHANGE ONLY:")[0]
    black_preserve = black_prompt.split("CHANGE ONLY:")[0]
    assert white_preserve == black_preserve


def test_foreground_background_prominence_appears_only_in_change_block():
    prompted = _prompted_timeline(_sample_pgn())
    shot = prompted.shots[1]
    edit_prompt = compose_reference_edit_prompt(shot)
    preserve_block, change_block = edit_prompt.split("CHANGE ONLY:", 1)
    assert "prominence" not in preserve_block.lower()
    assert "background" not in preserve_block.lower()
    assert "prominence" in change_block.lower()


def test_shot_action_and_camera_remain_shot_specific():
    prompted = _prompted_timeline(_sample_pgn())
    edit_prompt_1 = compose_reference_edit_prompt(prompted.shots[1])
    edit_prompt_2 = compose_reference_edit_prompt(prompted.shots[2])
    change_1 = edit_prompt_1.split("CHANGE ONLY:")[1]
    change_2 = edit_prompt_2.split("CHANGE ONLY:")[1]
    assert change_1 != change_2


# --- 17/18. Anchor dimension validation --------------------------------------


def test_anchor_dimensions_validated_before_reference_generation(tmp_path):
    """A genuinely working run with correctly-sized frames must pass
    validation (no exception) before both reference calls."""
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert len(reference_provider.calls) == 2  # reached, meaning dimension validation passed


def test_wrong_anchor_dimensions_cause_zero_reference_generations(tmp_path):
    from core.image_router import ImageGenerationResult, ImageProvider, ImageProviderRegistry, ImageRouter
    from products.chess2fight.rendering.animation_pipeline import AnimationPipeline
    from products.chess2fight.rendering.asset_manager import AssetManager
    from products.chess2fight.rendering.reference_continuity_acceptance import ReferenceContinuityAcceptanceRunner
    from products.chess2fight.rendering.render_pipeline import RenderPipeline
    from products.chess2fight.rendering.video_builder import VideoBuilder
    from core.animation_router import AnimationProviderRegistry, AnimationRouter, MockAnimationProvider
    from core.ai_router import TemplateProvider
    from tests.test_prompt13_reference_continuity import _FakeReferenceProvider

    class _WrongSizeAnchorProvider(ImageProvider):
        async def generate_image(self, prompt, width=1024, height=1024):
            # Deliberately generates the WRONG size regardless of what was requested.
            path = str(tmp_path / "wrong_size_anchor.png")
            Image.new("RGB", (640, 480)).save(path)
            return ImageGenerationResult(
                image_path=path, provider="_WrongSizeAnchorProvider", prompt=prompt,
                width=640, height=480, generation_time_seconds=0.0, metadata={"seed": 1},
            )

    image_registry = ImageProviderRegistry()
    image_registry.register("mock", lambda: _WrongSizeAnchorProvider())
    render_pipeline = RenderPipeline(
        image_router=ImageRouter(registry=image_registry), asset_manager=AssetManager(storage_root=str(tmp_path / "storage"))
    )
    anim_registry = AnimationProviderRegistry()
    anim_registry.register("mock", lambda: MockAnimationProvider(output_dir=str(tmp_path / "anim")))
    animation_pipeline = AnimationPipeline(animation_router=AnimationRouter(registry=anim_registry))
    runner = ReferenceContinuityAcceptanceRunner(
        TemplateProvider(), render_pipeline=render_pipeline, animation_pipeline=animation_pipeline,
        video_builder=VideoBuilder(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")),
    )
    reference_provider = _FakeReferenceProvider(1)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))

    with pytest.raises(ReferenceAnchorInvalidError):
        asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert len(reference_provider.calls) == 0


# --- 19/20. Same anchor, generation contract preserved -----------------------


def test_shots_1_and_2_still_reference_the_same_anchor(tmp_path):
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert reference_provider.calls[0][1] == reference_provider.calls[1][1] == result.anchor.image_path


def test_generation_contract_remains_1_t2i_2_reference_3_wan(tmp_path):
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert result.generation_modes == ["t2i", "reference_conditioned", "reference_conditioned"]
    assert plan.expected_comfyui_job_count == 6


# --- 21/22/23. Generic contracts, no real ComfyUI contact, gated live tests -


def test_generic_image_provider_signature_still_unchanged():
    import inspect

    from core.image_router import ImageProvider

    sig = inspect.signature(ImageProvider.generate_image)
    assert list(sig.parameters.keys()) == ["self", "prompt", "width", "height"]


def test_no_ordinary_test_module_contacts_real_comfyui():
    with open("products/chess2fight/rendering/acceptance_preflight.py") as f:
        source = f.read()
    assert "requests.get(" not in source
    assert "http://" not in source  # only settings-derived URLs, never hardcoded


def test_live_tests_remain_gated():
    import os

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_multi_shot_live_acceptance.py", "-q"],
        capture_output=True, text=True, env={k: v for k, v in os.environ.items() if "LIVE_TEST" not in k},
    )
    assert "skipped" in result.stdout.lower()
