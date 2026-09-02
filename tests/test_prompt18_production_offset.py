"""Tests for Sprint 4 Prompt 18's promotion of reference_latents_method
"offset" from Sprint 4 Prompt 16 calibration-only behavior into
Halisako's production FLUX.2 reference workflow.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
import io
import json
import subprocess
import sys

import httpx
from PIL import Image

from core.config import get_settings
from products.chess2fight.rendering.acceptance_preflight import preflight_check
from tests.test_prompt13_1_workflow_hardening import _MockTransport, _model_visibility_handlers, _patch_client
from tests.test_prompt13_reference_continuity import _runner_and_provider
from tests.test_single_shot_acceptance import _preferences, _sample_pgn

PRODUCTION_WORKFLOW = "products/chess2fight/rendering/workflows/flux2_klein_reference_4b.json"
_PGN = _sample_pgn()

# Captured once, at module import time — same established pattern as
# tests/test_comfyui_image_provider.py's own _patch_httpx_client.
import core.image_providers.comfyui as _comfyui_module  # noqa: E402

_REAL_HTTPX_ASYNC_CLIENT = httpx.AsyncClient


def _load_production_workflow() -> dict:
    with open(PRODUCTION_WORKFLOW) as f:
        return json.load(f)


# --- 1/2/3/4/5. Production topology ------------------------------------------


def test_production_workflow_contains_exactly_two_method_nodes():
    wf = _load_production_workflow()
    method_nodes = [n for n in wf.values() if n.get("class_type") == "FluxKontextMultiReferenceLatentMethod"]
    assert len(method_nodes) == 2


def test_positive_method_is_offset():
    wf = _load_production_workflow()
    assert wf["method:1"]["inputs"]["reference_latents_method"] == "offset"


def test_negative_method_is_offset():
    wf = _load_production_workflow()
    assert wf["method:2"]["inputs"]["reference_latents_method"] == "offset"


def test_positive_branch_topology_is_referencelatent_then_method_then_cfgguider():
    wf = _load_production_workflow()
    assert wf["method:1"]["inputs"]["conditioning"] == ["ref:3", 0]
    assert wf["ref:3"]["class_type"] == "ReferenceLatent"
    assert wf["77:90"]["inputs"]["positive"] == ["method:1", 0]


def test_negative_branch_topology_is_referencelatent_then_method_then_cfgguider():
    wf = _load_production_workflow()
    assert wf["method:2"]["inputs"]["conditioning"] == ["ref:4", 0]
    assert wf["ref:4"]["class_type"] == "ReferenceLatent"
    assert wf["77:90"]["inputs"]["negative"] == ["method:2", 0]


# --- 6-13. All other generation parameters unchanged -------------------------


def test_production_model_unchanged():
    wf = _load_production_workflow()
    assert wf["77:87"]["inputs"]["unet_name"] == "flux-2-klein-4b.safetensors"


def test_production_text_encoder_unchanged():
    wf = _load_production_workflow()
    assert wf["77:88"]["inputs"]["clip_name"] == "qwen_3_4b.safetensors"


def test_production_vae_unchanged():
    wf = _load_production_workflow()
    assert wf["77:89"]["inputs"]["vae_name"] == "flux2-vae.safetensors"


def test_production_dimensions_unchanged():
    wf = _load_production_workflow()
    assert wf["77:84"]["inputs"]["value"] == 1280
    assert wf["77:85"]["inputs"]["value"] == 704


def test_production_steps_unchanged():
    wf = _load_production_workflow()
    assert wf["77:93"]["inputs"]["steps"] == 4


def test_production_cfg_unchanged():
    wf = _load_production_workflow()
    assert wf["77:90"]["inputs"]["cfg"] == 1


def test_production_sampler_unchanged():
    wf = _load_production_workflow()
    assert wf["77:80"]["inputs"]["sampler_name"] == "euler"


def test_production_saveimage_node_78_unchanged():
    wf = _load_production_workflow()
    assert wf["78"]["class_type"] == "SaveImage"
    assert wf["78"]["inputs"]["images"] == ["77:82", 0]


# --- 14/15. Prompt/seed behavior unchanged ------------------------------------


def test_prompt15_1_prompt_behavior_unchanged():
    from products.chess2fight.cinematic.prompt_generator import compose_reference_edit_prompt
    from tests.test_prompt12_visual_continuity import _prompted_timeline

    prompted = _prompted_timeline(_PGN)
    edit_prompt = compose_reference_edit_prompt(prompted.shots[1])
    assert "PRESERVE EXACTLY" in edit_prompt
    assert "CHANGE ONLY" in edit_prompt
    assert "same facial identity" in edit_prompt


def test_derived_per_shot_seed_behavior_unchanged():
    from products.chess2fight.rendering.visual_continuity import derive_fight_base_visual_seed, derive_shot_seed

    base_seed = derive_fight_base_visual_seed(_PGN, "anime", "duel")
    assert base_seed == 1697950441
    assert derive_shot_seed(123, "test prompt") == 2860405817


# --- 16/17/18. Production continuity path audit ------------------------------


def test_canonical_anchor_generation_remains_t2i(tmp_path):
    """Shot 0 (the anchor) goes through the ordinary T2I workflow/path
    — never through the reference-method nodes at all."""
    runner, reference_provider = _runner_and_provider(tmp_path)
    plan = asyncio.run(runner.prepare(_PGN, _preferences(), max_animation_seconds=2.0))
    result = asyncio.run(runner.execute(plan, reference_provider, width=256, height=256))
    assert result.generation_modes[0] == "t2i"
    assert result.anchor.provenance == "t2i"


def test_reference_conditioned_submission_uses_production_offset_workflow(tmp_path, monkeypatch):
    """End-to-end: the real ComfyUIImageProvider, constructed with no
    explicit reference_workflow_path override (exactly how production
    FightVideoPipeline/ReferenceContinuityAcceptanceRunner construct
    it), must load and submit the PRODUCTION workflow file — which now
    carries method:1/method:2 = offset."""
    monkeypatch.setattr(get_settings(), "image_provider", "comfyui")

    submitted_workflows = []

    class RoutedTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            await request.aread()
            if request.method == "POST" and request.url.path == "/upload/image":
                return httpx.Response(200, json={"name": "anchor.png", "subfolder": "", "type": "input"})
            if request.method == "POST" and request.url.path == "/prompt":
                body = json.loads(request.content)
                submitted_workflows.append(body["prompt"])
                return httpx.Response(200, json={"prompt_id": "job-1", "node_errors": {}})
            if request.method == "GET" and request.url.path.startswith("/history/"):
                pid = request.url.path.split("/")[-1]
                return httpx.Response(200, json={pid: {"status": {"status_str": "success"},
                    "outputs": {"78": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}}})
            if request.method == "GET" and request.url.path == "/view":
                img_bytes = io.BytesIO()
                Image.new("RGB", (1280, 704)).save(img_bytes, format="PNG")
                return httpx.Response(200, content=img_bytes.getvalue())
            return httpx.Response(404)

    def _factory(*a, **kw):
        kw["transport"] = RoutedTransport()
        return _REAL_HTTPX_ASYNC_CLIENT(*a, **kw)

    monkeypatch.setattr(_comfyui_module.httpx, "AsyncClient", _factory)

    from core.image_providers.comfyui import ComfyUIImageProvider

    anchor_path = str(tmp_path / "anchor.png")
    Image.new("RGB", (1280, 704)).save(anchor_path)

    # No reference_workflow_path override — exactly how production code
    # constructs this provider; it must default to the (now offset-carrying)
    # production file, per settings.comfyui_reference_workflow_path.
    provider = ComfyUIImageProvider()
    asyncio.run(provider.generate_reference_conditioned_image("a test prompt", anchor_path, width=1280, height=704))

    assert len(submitted_workflows) == 1
    assert submitted_workflows[0]["method:1"]["inputs"]["reference_latents_method"] == "offset"
    assert submitted_workflows[0]["method:2"]["inputs"]["reference_latents_method"] == "offset"


# --- 19/20. Live production preflight requires the method node -------------


def test_live_reference_preflight_requires_flux_kontext_method_node(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _model_visibility_handlers()
    handlers["GET /object_info/ReferenceLatent"] = lambda r: httpx.Response(200, json={"ReferenceLatent": {"input": {}}})
    handlers["GET /object_info/VAEEncode"] = lambda r: httpx.Response(200, json={"VAEEncode": {"input": {}}})
    handlers["GET /object_info/FluxKontextMultiReferenceLatentMethod"] = lambda r: httpx.Response(200, json={
        "FluxKontextMultiReferenceLatentMethod": {"input": {"required": {"reference_latents_method": [["offset", "index", "uxo/uno", "index_timestep_zero"], {}]}}}
    })
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(preflight_check(settings, check_reference_workflow=True))
    assert problems == []


def test_missing_offset_capability_fails_before_generation(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _model_visibility_handlers()
    handlers["GET /object_info/ReferenceLatent"] = lambda r: httpx.Response(200, json={"ReferenceLatent": {"input": {}}})
    handlers["GET /object_info/VAEEncode"] = lambda r: httpx.Response(200, json={"VAEEncode": {"input": {}}})
    # Node exists, but this older install's own advertised choices don't include "offset".
    handlers["GET /object_info/FluxKontextMultiReferenceLatentMethod"] = lambda r: httpx.Response(200, json={
        "FluxKontextMultiReferenceLatentMethod": {"input": {"required": {"reference_latents_method": [["index"], {}]}}}
    })
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(preflight_check(settings, check_reference_workflow=True))
    assert len(problems) == 1
    assert "offset" in problems[0]


# --- 21/22. Calibration regression preservation ------------------------------


def test_prompt16_calibration_workflows_remain_available():
    import os

    for filename in [
        "flux2_klein_reference_method_offset_4b.json",
        "flux2_klein_reference_method_uxo_4b.json",
        "flux2_klein_reference_method_index_timestep_zero_4b.json",
    ]:
        assert os.path.exists(f"products/chess2fight/rendering/workflows/{filename}")


def test_prompt17_single_method_calibration_behavior_unchanged():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_prompt17_method_sweep_generalization.py", "-q"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "9 passed" in result.stdout


# --- 23/24. No Wan/VideoBuilder behavior changes ------------------------------


def test_no_wan_behavior_changes():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_animation_pipeline.py", "tests/test_comfyui_animation_provider.py",
         "tests/test_flux_to_wan_handoff.py", "-q"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


def test_no_videobuilder_behavior_changes():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_video_builder.py", "-q"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


# --- 25. Ordinary tests never contact real ComfyUI ---------------------------


def test_no_hardcoded_network_target_in_production_workflow_source():
    with open(PRODUCTION_WORKFLOW) as f:
        content = f.read()
    assert "http://" not in content
    assert "https://" not in content
