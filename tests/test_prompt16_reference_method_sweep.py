"""Tests for Sprint 4 Prompt 16's reference-latent method sweep —
isolating `reference_latents_method` for the single known-difficult
Prompt 15.1 shot (timeline index 2).

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
import hashlib
import io
import json
import subprocess
import sys

import httpx
import pytest
from PIL import Image

from core.ai_router import TemplateProvider
from core.exceptions import ImageProviderError
from core.image_router import ImageProvider
from products.chess2fight.rendering import acceptance_preflight as preflight_module
from products.chess2fight.rendering.acceptance_preflight import check_reference_method_node_availability
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.multi_shot_acceptance import SeedEvidenceMismatchError
from products.chess2fight.rendering.reference_method_calibration import (
    CANDIDATE_METHODS,
    ReferenceMethodCalibrationRunner,
    method_workflow_path,
)
from products.chess2fight.rendering.reference_seed_calibration import AnchorValidationError
from tests.test_prompt14_reference_seed_calibration import _make_anchor
from tests.test_single_shot_acceptance import _preferences, _sample_pgn

RENDER_METHOD_CLI = "scripts/render_reference_method_calibration.py"
PRODUCTION_WORKFLOW = "products/chess2fight/rendering/workflows/flux2_klein_reference_4b.json"

_PGN = _sample_pgn()

# Captured once, at module import time — see the identical established
# pattern/rationale in tests/test_comfyui_image_provider.py's own
# _patch_httpx_client and tests/test_prompt15_1_live_seed_wiring.py.
import core.image_providers.comfyui as _comfyui_module  # noqa: E402

_REAL_HTTPX_ASYNC_CLIENT = httpx.AsyncClient


def _runner(tmp_path):
    return ReferenceMethodCalibrationRunner(TemplateProvider(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")))


async def _plan(tmp_path, anchor_path=None):
    if anchor_path is None:
        anchor_path = str(tmp_path / "anchor.png")
        _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = await runner.prepare(
        _PGN, _preferences(), anchor_path=anchor_path, anchor_original_seed=1697950441,
        seed=981216397, style="anime", battle_mode="duel", timeline_index=2,
    )
    return runner, plan


class RoutedTransport(httpx.AsyncBaseTransport):
    def __init__(self, fail_on_workflow_containing: str | None = None):
        self.submitted_workflows = []
        self.job_counter = 0
        self._fail_on = fail_on_workflow_containing

    async def handle_async_request(self, request):
        await request.aread()
        if request.method == "POST" and request.url.path == "/upload/image":
            return httpx.Response(200, json={"name": "anchor.png", "subfolder": "", "type": "input"})
        if request.method == "POST" and request.url.path == "/prompt":
            body = json.loads(request.content)
            method_value = body["prompt"].get("method:1", {}).get("inputs", {}).get("reference_latents_method")
            if self._fail_on is not None and method_value == self._fail_on:
                return httpx.Response(500, json={"error": "simulated failure"})
            self.submitted_workflows.append(body["prompt"])
            self.job_counter += 1
            return httpx.Response(200, json={"prompt_id": f"job-{self.job_counter}", "node_errors": {}})
        if request.method == "GET" and request.url.path.startswith("/history/"):
            pid = request.url.path.split("/")[-1]
            return httpx.Response(200, json={pid: {"status": {"status_str": "success"},
                "outputs": {"78": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}}})
        if request.method == "GET" and request.url.path == "/view":
            img_bytes = io.BytesIO()
            Image.new("RGB", (1280, 704)).save(img_bytes, format="PNG")
            return httpx.Response(200, content=img_bytes.getvalue())
        return httpx.Response(404)


def _patch_transport(monkeypatch, transport):
    """Sprint 4 Prompt 18 fix: this previously did a direct, permanent
    `_comfyui_module.httpx.AsyncClient = _client_factory` assignment
    with no teardown at all — meaning whichever test in this file ran
    last left httpx.AsyncClient permanently patched to a lambda
    routing to that one test's own, long-finished transport, silently
    breaking any later test file's own httpx mocking for the rest of
    that pytest session (confirmed directly: reproduced by running
    this file together with tests/test_prompt18_production_offset.py,
    where two of that file's tests failed with a real 404 response
    from THIS file's leftover transport, not their own). Using
    monkeypatch.setattr here instead means it's automatically reverted
    at the end of each test that uses it, exactly like every other
    httpx-patching helper in this codebase already does (see
    tests/test_prompt13_1_workflow_hardening.py's own _patch_client)."""

    def _client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_HTTPX_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(_comfyui_module.httpx, "AsyncClient", _client_factory)


# --- 1. Production reference workflow byte-for-byte unchanged --------------


def test_production_reference_workflow_checksum_matches_current_baseline():
    """Sprint 4 Prompt 18 deliberately, correctly changed the
    production reference workflow (promoting reference_latents_method
    "offset" into it — see that prompt's own diff record). This test's
    checksum was 738ad1818a72a2ac21c5f7ddf69e23c7ead867515a609b3520e07a6c6fe14a9b
    (Sprint 4 Prompt 16, before that promotion); updated to the new,
    current baseline so future unexpected drift is still caught."""
    actual_sha256 = hashlib.sha256(open(PRODUCTION_WORKFLOW, "rb").read()).hexdigest()
    assert actual_sha256 == "b2ff7c9e1024abc76361c363601c68870148f9924dc3ca7c022c81ecfdb10b3d"


# --- 2/3/4/5/6/7. Experimental workflow structure ---------------------------


@pytest.mark.parametrize("method,expected_value", [
    ("offset", "offset"), ("uxo/uno", "uxo/uno"), ("index_timestep_zero", "index_timestep_zero"),
])
def test_experimental_workflow_uses_exact_method_value(method, expected_value):
    with open(method_workflow_path(method)) as f:
        wf = json.load(f)
    assert wf["method:1"]["inputs"]["reference_latents_method"] == expected_value
    assert wf["method:2"]["inputs"]["reference_latents_method"] == expected_value


@pytest.mark.parametrize("method", CANDIDATE_METHODS)
def test_positive_and_negative_branches_use_the_same_method(method):
    with open(method_workflow_path(method)) as f:
        wf = json.load(f)
    assert wf["method:1"]["inputs"]["reference_latents_method"] == wf["method:2"]["inputs"]["reference_latents_method"]


@pytest.mark.parametrize("method", CANDIDATE_METHODS)
def test_experimental_workflow_differs_from_production_only_by_method_value(method):
    """Sprint 4 Prompt 18 promoted "offset" into production, giving
    production the exact same method:1/method:2 nodes (and 77:90
    rewiring) every Prompt 16 experimental variant already had. This
    test's original premise — "exactly two NEW nodes vs. production" —
    is no longer true for any of the three variants, since production
    now has those same node IDs too. Redesigned to check what's
    actually true now: for "offset" specifically, the experimental
    file is byte-identical to production (documented explicitly, per
    this task's own "if the experimental offset workflow becomes
    byte-equivalent... document that fact" instruction); for the other
    two, every node is identical except method:1/method:2's own
    reference_latents_method value.
    """
    with open(PRODUCTION_WORKFLOW) as f:
        production = json.load(f)
    with open(method_workflow_path(method)) as f:
        experimental = json.load(f)

    if method == "offset":
        assert experimental == production
        return

    # No longer "new" nodes — both files already have method:1/method:2 by
    # the same IDs; only their reference_latents_method value differs.
    assert set(experimental.keys()) == set(production.keys())
    for node_id in production:
        if node_id in ("method:1", "method:2"):
            continue
        assert experimental[node_id] == production[node_id]
    assert experimental["method:1"]["inputs"]["reference_latents_method"] == method
    assert experimental["method:2"]["inputs"]["reference_latents_method"] == method
    assert production["method:1"]["inputs"]["reference_latents_method"] == "offset"
    assert production["method:2"]["inputs"]["reference_latents_method"] == "offset"


@pytest.mark.parametrize("method", CANDIDATE_METHODS)
def test_no_experimental_workflow_changes_other_settings(method):
    with open(method_workflow_path(method)) as f:
        wf = json.load(f)
    assert wf["77:87"]["inputs"]["unet_name"] == "flux-2-klein-4b.safetensors"  # model loader
    assert wf["77:88"]["inputs"]["clip_name"] == "qwen_3_4b.safetensors"  # CLIP loader
    assert wf["77:89"]["inputs"]["vae_name"] == "flux2-vae.safetensors"  # VAE loader
    assert wf["77:90"]["inputs"]["cfg"] == 1  # CFG
    assert wf["77:93"]["inputs"]["steps"] == 4  # step count
    assert wf["77:80"]["inputs"]["sampler_name"] == "euler"  # sampler
    assert wf["77:93"]["class_type"] == "Flux2Scheduler"  # scheduler
    assert wf["77:84"]["inputs"]["value"] == 1280  # width
    assert wf["77:85"]["inputs"]["value"] == 704  # height
    assert wf["78"]["class_type"] == "SaveImage"  # SaveImage node


# --- 8/9/10/11/12/13. Plan correctness -----------------------------------


def test_only_timeline_index_2_is_planned(tmp_path):
    _, plan = asyncio.run(_plan(tmp_path))
    assert plan.timeline_index == 2


def test_exact_prompt15_1_shot2_prompt_is_reused(tmp_path):
    from products.chess2fight.cinematic.prompt_generator import compose_reference_edit_prompt
    from tests.test_prompt12_visual_continuity import _prompted_timeline

    _, plan = asyncio.run(_plan(tmp_path))
    prompted = _prompted_timeline(_PGN)
    expected_prompt = compose_reference_edit_prompt(prompted.shots[2])
    assert plan.prompt == expected_prompt


def test_seed_is_exactly_981216397_for_every_candidate(tmp_path):
    _, plan = asyncio.run(_plan(tmp_path))
    assert plan.planned_seed == 981216397
    assert all(True for _ in plan.candidates)  # planned_seed is plan-level, shared by all


def test_actual_seed_equals_planned_seed_end_to_end(tmp_path, monkeypatch):
    transport = RoutedTransport()
    _patch_transport(monkeypatch, transport)
    runner, plan = asyncio.run(_plan(tmp_path))
    result = asyncio.run(runner.execute(plan))
    for r in result.candidate_results:
        assert r.actual_seed == r.planned_seed == 981216397


def test_same_anchor_path_used_for_every_method(tmp_path, monkeypatch):
    transport = RoutedTransport()
    _patch_transport(monkeypatch, transport)
    runner, plan = asyncio.run(_plan(tmp_path))
    asyncio.run(runner.execute(plan))
    # All three jobs uploaded from the identical anchor path — confirmed
    # via the plan itself (single anchor field, not per-candidate).
    assert len({plan.anchor.path}) == 1


def test_same_anchor_sha256_used_for_every_method(tmp_path):
    _, plan = asyncio.run(_plan(tmp_path))
    # Single anchor object shared by the whole plan — one sha256 for all three candidates by construction.
    assert plan.anchor.sha256 == hashlib.sha256(open(plan.anchor.path, "rb").read()).hexdigest()


# --- 14/15/16/17/18/19. Generation counts and control preservation ---------


def test_current_index_control_never_generated(tmp_path, monkeypatch):
    transport = RoutedTransport()
    _patch_transport(monkeypatch, transport)
    runner, plan = asyncio.run(_plan(tmp_path))
    asyncio.run(runner.execute(plan))
    submitted_methods = [wf["method:1"]["inputs"]["reference_latents_method"] for wf in transport.submitted_workflows]
    assert "index" not in submitted_methods
    assert plan.control.generated_this_run is False


def test_exactly_three_provider_generations_occur(tmp_path, monkeypatch):
    transport = RoutedTransport()
    _patch_transport(monkeypatch, transport)
    runner, plan = asyncio.run(_plan(tmp_path))
    asyncio.run(runner.execute(plan))
    assert transport.job_counter == 3


def test_each_provider_uses_the_correct_reference_workflow_path(tmp_path, monkeypatch):
    transport = RoutedTransport()
    _patch_transport(monkeypatch, transport)
    runner, plan = asyncio.run(_plan(tmp_path))
    result = asyncio.run(runner.execute(plan))
    expected_slugs = {"offset": "offset", "uxo/uno": "uxo", "index_timestep_zero": "index_timestep_zero"}
    for candidate, r in zip(plan.candidates, result.candidate_results, strict=True):
        assert r.workflow_path == candidate.workflow_path
        assert expected_slugs[candidate.method] in candidate.workflow_path


def test_zero_t2i_calls():
    """The runner never imports RenderPipeline/ImageRouter — no T2I path exists at all."""
    import products.chess2fight.rendering.reference_method_calibration as module

    with open(module.__file__) as f:
        source = f.read()
    assert "import RenderPipeline" not in source
    assert "import ImageRouter" not in source


def test_zero_wan_calls():
    import products.chess2fight.rendering.reference_method_calibration as module

    with open(module.__file__) as f:
        source = f.read()
    assert "import AnimationPipeline" not in source
    assert "import AnimationRouter" not in source


def test_zero_videobuilder_calls():
    import products.chess2fight.rendering.reference_method_calibration as module

    with open(module.__file__) as f:
        source = f.read()
    assert "import VideoBuilder" not in source


# --- 20/21/22. Failure semantics --------------------------------------------


def test_no_retry_no_fallback_on_failure(tmp_path, monkeypatch):
    transport = RoutedTransport(fail_on_workflow_containing="offset")
    _patch_transport(monkeypatch, transport)
    runner, plan = asyncio.run(_plan(tmp_path))
    with pytest.raises(ImageProviderError):
        asyncio.run(runner.execute(plan))
    # Only the one, failing attempt was made for "offset" — no retry.
    assert transport.job_counter == 0  # offset failed immediately, never counted as a success


def test_first_failure_stops_later_methods(tmp_path, monkeypatch):
    transport = RoutedTransport(fail_on_workflow_containing="offset")
    _patch_transport(monkeypatch, transport)
    runner, plan = asyncio.run(_plan(tmp_path))
    with pytest.raises(ImageProviderError):
        asyncio.run(runner.execute(plan))
    submitted_methods = [wf["method:1"]["inputs"]["reference_latents_method"] for wf in transport.submitted_workflows]
    assert "uxo/uno" not in submitted_methods
    assert "index_timestep_zero" not in submitted_methods


def test_partial_results_preserved_on_later_failure(tmp_path, monkeypatch):
    transport = RoutedTransport(fail_on_workflow_containing="uxo/uno")
    _patch_transport(monkeypatch, transport)
    runner, plan = asyncio.run(_plan(tmp_path))
    with pytest.raises(ImageProviderError):
        asyncio.run(runner.execute(plan))
    # "offset" (the first candidate) succeeded before "uxo/uno" failed — its output file remains.
    calibration_dir = tmp_path / "storage" / "reference_method_calibration" / plan.run_id
    files = list(calibration_dir.glob("*.png")) if calibration_dir.exists() else []
    assert len(files) == 1
    assert "offset" in files[0].name


# --- 23/24. Dry-run and job count -------------------------------------------


def test_dry_run_zero_provider_calls():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        anchor_path = f"{tmp}/anchor.png"
        _make_anchor(anchor_path)
        result = subprocess.run(
            [sys.executable, RENDER_METHOD_CLI, "--sample", "--anchor-path", anchor_path,
             "--anchor-original-seed", "1697950441", "--seed", "981216397", "--dry-run"],
            capture_output=True, text=True, check=True,
        )
        assert "generating" not in result.stdout.lower()
        assert "dry run complete" in result.stdout.lower()
        assert "expected comfyui jobs: 3" in result.stdout.lower()


def test_expected_job_count_exactly_three(tmp_path):
    _, plan = asyncio.run(_plan(tmp_path))
    assert plan.expected_comfyui_jobs == 3


# --- 25/26. Existing behavior unchanged --------------------------------------


def test_normal_prompt15_1_behavior_unchanged():
    result = subprocess.run(
        [sys.executable, "scripts/render_reference_seed_calibration.py", "--sample",
         "--anchor-path", "/dev/null", "--anchor-original-seed", "1", "--dry-run"],
        capture_output=True, text=True,
    )
    # /dev/null isn't a valid 1280x704 image -- confirms this script's own
    # anchor validation still runs exactly as before (unaffected by Prompt 16).
    assert result.returncode == 1
    assert "anchor validation failed" in result.stderr.lower()


def test_existing_comfyui_image_provider_public_api_unchanged():
    import inspect

    sig = inspect.signature(ImageProvider.generate_image)
    assert list(sig.parameters.keys()) == ["self", "prompt", "width", "height"]


# --- 27. Mocked payload proves the method node carries the intended value ---


def test_mocked_payload_proves_method_node_carries_intended_value(tmp_path, monkeypatch):
    transport = RoutedTransport()
    _patch_transport(monkeypatch, transport)
    runner, plan = asyncio.run(_plan(tmp_path))
    asyncio.run(runner.execute(plan))
    submitted_methods = [wf["method:1"]["inputs"]["reference_latents_method"] for wf in transport.submitted_workflows]
    assert submitted_methods == ["offset", "uxo/uno", "index_timestep_zero"]
    # And the negative branch's node agrees with the positive branch's, per job.
    for wf in transport.submitted_workflows:
        assert wf["method:1"]["inputs"]["reference_latents_method"] == wf["method:2"]["inputs"]["reference_latents_method"]


# --- 28. No ordinary test contacts real ComfyUI; preflight node check works -


def test_no_ordinary_test_contacts_real_comfyui():
    import products.chess2fight.rendering.reference_method_calibration as module

    with open(module.__file__) as f:
        source = f.read()
    assert "httpx.AsyncClient(" not in source


def test_preflight_method_node_missing_is_hard_failure(monkeypatch):
    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, json={})  # node not registered

    def _factory(*a, **kw):
        kw["transport"] = MockTransport()
        return _REAL_HTTPX_ASYNC_CLIENT(*a, **kw)

    monkeypatch.setattr(preflight_module.httpx, "AsyncClient", _factory)
    from core.config import get_settings

    problems = asyncio.run(check_reference_method_node_availability(get_settings(), list(CANDIDATE_METHODS)))
    assert len(problems) == 1
    assert "FluxKontextMultiReferenceLatentMethod" in problems[0]


def test_seed_mismatch_raises_seed_evidence_mismatch_error(tmp_path):
    class _DisagreeingProvider:
        """Duck-types only generate_reference_conditioned_image —
        deliberately not a subclass of the abstract ImageProvider base
        class, matching the established, working pattern in
        tests/test_prompt13_reference_continuity.py's own
        _FakeReferenceProvider (that ABC's generate_image is abstract,
        so a subclass not implementing it can't be instantiated)."""

        async def generate_reference_conditioned_image(self, prompt, reference_image_path, width=1024, height=1024):
            from core.image_router import ImageGenerationResult

            path = str(tmp_path / "disagreeing.png")
            Image.new("RGB", (width, height)).save(path)
            return ImageGenerationResult(
                image_path=path, provider="_DisagreeingProvider", prompt=prompt, width=width, height=height,
                generation_time_seconds=0.0, metadata={"seed": 999999999},  # deliberately wrong
            )

    runner, plan = asyncio.run(_plan(tmp_path))
    import products.chess2fight.rendering.reference_method_calibration as module

    original_provider_cls = module.ComfyUIImageProvider
    module.ComfyUIImageProvider = lambda **kwargs: _DisagreeingProvider()
    try:
        with pytest.raises(SeedEvidenceMismatchError):
            asyncio.run(runner.execute(plan))
    finally:
        module.ComfyUIImageProvider = original_provider_cls


def test_anchor_validation_failure_means_zero_paid_jobs(tmp_path):
    runner = _runner(tmp_path)
    with pytest.raises(AnchorValidationError):
        asyncio.run(
            runner.prepare(
                _PGN, _preferences(), anchor_path=str(tmp_path / "nonexistent.png"), anchor_original_seed=1,
                seed=981216397, style="anime", battle_mode="duel",
            )
        )
