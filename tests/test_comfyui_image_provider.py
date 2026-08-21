"""Tests for ComfyUIImageProvider.

Sprint 4 Prompt 6: rewritten to test against the actual, experimentally
validated FLUX.2 Klein workflow file
(products/chess2fight/rendering/workflows/flux2_klein_t2i_4b.json) —
not a synthetic title-based fixture. Every ComfyUI HTTP call is still
mocked; these tests never require a real ComfyUI server, GPU, or FLUX
checkpoint. For the gated live test, see
tests/test_comfyui_image_live_integration.py.
"""

import asyncio
import copy
import json

import httpx
from PIL import Image

from core.exceptions import ImageProviderError
from core.image_providers.comfyui import ComfyUIImageProvider, ComfyUIImageRequestError, _derive_seed
from core.image_router import ImageGenerationResult, ImageProvider, ImageProviderRegistry, ImageRouter, MockImageProvider, _default_registry

REAL_WORKFLOW_PATH = "products/chess2fight/rendering/workflows/flux2_klein_t2i_4b.json"


def _load_real_workflow() -> dict:
    with open(REAL_WORKFLOW_PATH) as f:
        return json.load(f)


class _MockTransport(httpx.AsyncBaseTransport):
    """A real httpx transport double — routes requests to a dict of
    {method+path: handler}, so tests exercise real httpx request/
    response serialization rather than mocking client methods."""

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


def _provider_with_transport(tmp_path, handlers, workflow_path=None):
    transport = _MockTransport(handlers)
    provider = ComfyUIImageProvider(
        base_url="http://fake-comfyui:8188",
        workflow_path=workflow_path or REAL_WORKFLOW_PATH,
        timeout_seconds=5.0,
        output_dir=str(tmp_path / "out"),
    )
    return provider, transport


_REAL_HTTPX_ASYNC_CLIENT = httpx.AsyncClient  # captured once, before any test can monkeypatch it


def _patch_httpx_client(monkeypatch, transport: "_MockTransport") -> None:
    """Always wraps the TRUE original class (captured at import time,
    above) — capturing it fresh inside this function would, on a
    second call within the same test, wrap the already-patched
    factory from the first call instead of the real class."""
    import core.image_providers.comfyui as module

    def _client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_HTTPX_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", _client_factory)


def _real_png_bytes(width=1280, height=704, color=(100, 50, 200)) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _success_handlers(image_bytes: bytes | None = None, prompt_id: str = "test-prompt-123"):
    image_bytes = image_bytes if image_bytes is not None else _real_png_bytes()

    def queue(request):
        body = json.loads(request.content)
        assert body["prompt"]
        return httpx.Response(200, json={"prompt_id": prompt_id, "number": 1, "node_errors": {}})

    def history(request):
        return httpx.Response(
            200,
            json={
                prompt_id: {
                    "status": {"status_str": "success"},
                    "outputs": {"78": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}},
                }
            },
        )

    def view(request):
        return httpx.Response(200, content=image_bytes)

    return {"POST /prompt": queue, f"GET /history/{prompt_id}": history, "GET /view": view}


# --- Registration / router resolution (unchanged from Prompt 5) -------------


def test_registered_in_the_default_registry():
    registry = _default_registry()
    assert registry.is_registered("comfyui")
    assert isinstance(registry.create("comfyui"), ComfyUIImageProvider)


def test_mock_provider_still_registered_alongside_comfyui():
    assert set(_default_registry().list_providers()) == {"mock", "comfyui"}


def test_router_resolves_comfyui_when_configured(tmp_path, monkeypatch):
    from core import config

    monkeypatch.setattr(config.get_settings(), "image_provider", "comfyui")
    registry = ImageProviderRegistry()
    registry.register("comfyui", lambda: ComfyUIImageProvider(workflow_path=str(tmp_path / "nonexistent.json")))
    router = ImageRouter(registry=registry)
    assert router.active_provider_name() == "comfyui"


def test_router_still_resolves_mock_by_default():
    assert ImageRouter().active_provider_name() == "mock"


# --- 1. API workflow loads ----------------------------------------------------


def test_real_workflow_file_loads_successfully():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    assert workflow["76"]["class_type"] == "PrimitiveStringMultiline"


def test_missing_workflow_raises_image_provider_error(tmp_path):
    provider = ComfyUIImageProvider(workflow_path=str(tmp_path / "does_not_exist.json"), output_dir=str(tmp_path))
    try:
        asyncio.run(provider.generate_image("a prompt"))
        raise AssertionError("should have raised")
    except ImageProviderError as e:
        assert "not found" in str(e).lower()


def test_malformed_workflow_json_raises_image_provider_error(tmp_path):
    bad_path = tmp_path / "malformed.json"
    bad_path.write_text("{not valid json")
    provider = ComfyUIImageProvider(workflow_path=str(bad_path), output_dir=str(tmp_path))
    try:
        asyncio.run(provider.generate_image("a prompt"))
        raise AssertionError("should have raised")
    except ImageProviderError:
        pass


# --- 2. String node IDs such as "77:84" work ---------------------------------


def test_colon_containing_node_ids_are_treated_as_string_keys():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    assert "77:84" in workflow
    assert "77:85" in workflow
    assert "77:86" in workflow
    prepared = provider._inject_parameters(workflow, "a prompt", 1280, 704)
    assert prepared["77:84"]["inputs"]["value"] == 1280


# --- 3. Prompt mutation targets only node 76 ---------------------------------


def test_prompt_injected_only_at_node_76():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    original = copy.deepcopy(workflow)
    prepared = provider._inject_parameters(workflow, "a unique battle prompt", 1280, 704)

    assert prepared["76"]["inputs"]["value"] == "a unique battle prompt"
    # No other node's "value"/"text" input changed.
    for node_id, node in prepared.items():
        if node_id == "76":
            continue
        for key in ("value", "text"):
            if key in node.get("inputs", {}) and key in original[node_id].get("inputs", {}):
                assert node["inputs"][key] == original[node_id]["inputs"][key], f"node {node_id} unexpectedly changed"


# --- 4/5. Width/height mutation targets only 77:84 / 77:85 -------------------


def test_width_injected_only_at_node_77_84():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(workflow, "a prompt", 1280, 704)
    assert prepared["77:84"]["inputs"]["value"] == 1280
    assert prepared["77:85"]["inputs"]["value"] == 704  # sanity: different node, different value


def test_height_injected_only_at_node_77_85():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(workflow, "a prompt", 960, 544)
    assert prepared["77:85"]["inputs"]["value"] == 544
    assert prepared["77:84"]["inputs"]["value"] == 960


def test_dimensions_normalized_to_nearest_multiple_of_16():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(workflow, "a prompt", 1000, 700)
    assert prepared["77:84"]["inputs"]["value"] % 16 == 0
    assert prepared["77:85"]["inputs"]["value"] % 16 == 0


# --- 6. Seed mutation targets only 77:86 -------------------------------------


def test_seed_injected_only_at_node_77_86():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(workflow, "a specific prompt", 1280, 704)
    assert prepared["77:86"]["inputs"]["noise_seed"] == _derive_seed("a specific prompt")


def test_same_prompt_always_derives_the_same_seed():
    assert _derive_seed("same prompt") == _derive_seed("same prompt")


def test_different_prompts_derive_different_seeds():
    assert _derive_seed("prompt A") != _derive_seed("prompt B")


# --- 7. Model names remain correct -------------------------------------------


def test_model_node_names_remain_unchanged_after_injection():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(workflow, "a prompt", 1280, 704)
    assert prepared["77:87"]["inputs"]["unet_name"] == "flux-2-klein-4b.safetensors"
    assert prepared["77:88"]["inputs"]["clip_name"] == "qwen_3_4b.safetensors"
    assert prepared["77:88"]["inputs"]["type"] == "flux2"
    assert prepared["77:89"]["inputs"]["vae_name"] == "flux2-vae.safetensors"


# --- 8/9/10. CFG=1, steps=4, sampler=euler remain unchanged ------------------


def test_cfg_remains_1():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(workflow, "a prompt", 1280, 704)
    assert prepared["77:90"]["inputs"]["cfg"] == 1


def test_steps_remains_4():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(workflow, "a prompt", 1280, 704)
    assert prepared["77:93"]["inputs"]["steps"] == 4


def test_sampler_remains_euler():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    prepared = provider._inject_parameters(workflow, "a prompt", 1280, 704)
    assert prepared["77:80"]["inputs"]["sampler_name"] == "euler"


# --- 11. SaveImage node 78 is correctly detected -----------------------------


def test_save_image_node_78_is_the_primary_output_target(tmp_path, monkeypatch):
    provider, transport = _provider_with_transport(tmp_path, _success_handlers())
    _patch_httpx_client(monkeypatch, transport)
    result = asyncio.run(provider.generate_image("a prompt"))
    assert result.image_path is not None


def test_extract_output_reference_checks_node_78_first():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    history_entry = {
        "outputs": {
            "78": {"images": [{"filename": "correct.png", "subfolder": "", "type": "output"}]},
            "99": {"images": [{"filename": "wrong.png", "subfolder": "", "type": "output"}]},
        }
    }
    filename, _, _ = provider._extract_output_reference(history_entry)
    assert filename == "correct.png"


# --- 12. Original workflow fixture is not mutated between requests ----------


def test_two_calls_do_not_mutate_or_leak_between_each_other():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    original_prompt_text = workflow["76"]["inputs"]["value"]

    prepared1 = provider._inject_parameters(workflow, "prompt one", 1280, 704)
    prepared2 = provider._inject_parameters(workflow, "prompt two", 960, 544)

    assert workflow["76"]["inputs"]["value"] == original_prompt_text  # loaded dict untouched
    assert prepared1["76"]["inputs"]["value"] == "prompt one"
    assert prepared2["76"]["inputs"]["value"] == "prompt two"
    assert prepared1["77:84"]["inputs"]["value"] == 1280
    assert prepared2["77:84"]["inputs"]["value"] == 960


# --- 13. Generated image output is downloaded correctly ---------------------


def test_full_successful_generation_flow(tmp_path, monkeypatch):
    provider, transport = _provider_with_transport(tmp_path, _success_handlers())
    _patch_httpx_client(monkeypatch, transport)

    result = asyncio.run(provider.generate_image("a fighter with a sword"))

    assert isinstance(result, ImageGenerationResult)
    assert result.provider == "ComfyUIImageProvider"

    import os

    assert os.path.exists(result.image_path)
    assert os.path.getsize(result.image_path) > 0


def test_downloaded_bytes_match_what_view_endpoint_returned(tmp_path, monkeypatch):
    specific_bytes = _real_png_bytes(width=800, height=600, color=(10, 20, 30))
    provider, transport = _provider_with_transport(tmp_path, _success_handlers(image_bytes=specific_bytes))
    _patch_httpx_client(monkeypatch, transport)

    result = asyncio.run(provider.generate_image("a prompt"))
    with open(result.image_path, "rb") as f:
        assert f.read() == specific_bytes


# --- 14. Useful errors for malformed/missing nodes ---------------------------


def test_missing_expected_node_raises_useful_error():
    provider = ComfyUIImageProvider(workflow_path=REAL_WORKFLOW_PATH)
    workflow = provider._load_workflow()
    del workflow["77:86"]  # simulate a hand-edited/corrupted workflow
    try:
        provider._inject_parameters(workflow, "a prompt", 1280, 704)
        raise AssertionError("should have raised")
    except ComfyUIImageRequestError as e:
        assert "77:86" in str(e)


def test_missing_node_error_surfaces_as_image_provider_error(tmp_path):
    workflow = _load_real_workflow()
    del workflow["76"]
    bad_path = tmp_path / "broken_workflow.json"
    bad_path.write_text(json.dumps(workflow))

    provider = ComfyUIImageProvider(workflow_path=str(bad_path), output_dir=str(tmp_path))
    try:
        asyncio.run(provider.generate_image("a prompt"))
        raise AssertionError("should have raised")
    except ImageProviderError as e:
        assert "76" in str(e)


# --- ComfyUI failure / timeout / invalid image (retained from Prompt 5) -----


def test_node_errors_in_queue_response_raises(tmp_path, monkeypatch):
    def queue(request):
        return httpx.Response(
            200, json={"prompt_id": None, "node_errors": {"77:87": {"errors": [{"message": "missing checkpoint"}]}}}
        )

    provider, transport = _provider_with_transport(tmp_path, {"POST /prompt": queue})
    _patch_httpx_client(monkeypatch, transport)
    try:
        asyncio.run(provider.generate_image("a prompt"))
        raise AssertionError("should have raised")
    except ImageProviderError as e:
        assert "rejected" in str(e).lower()
        assert "missing checkpoint" in str(e)


def test_error_status_in_history_raises(tmp_path, monkeypatch):
    def queue(request):
        return httpx.Response(200, json={"prompt_id": "test-123", "node_errors": {}})

    def history(request):
        return httpx.Response(200, json={"test-123": {"status": {"status_str": "error", "messages": ["OOM"]}}})

    provider, transport = _provider_with_transport(tmp_path, {"POST /prompt": queue, "GET /history/test-123": history})
    _patch_httpx_client(monkeypatch, transport)
    try:
        asyncio.run(provider.generate_image("a prompt"))
        raise AssertionError("should have raised")
    except ImageProviderError as e:
        assert "generation failed" in str(e).lower()


def test_unreachable_server_raises_image_provider_error(tmp_path):
    provider = ComfyUIImageProvider(
        base_url="http://this-host-does-not-exist.invalid:8188",
        workflow_path=REAL_WORKFLOW_PATH, timeout_seconds=3.0, output_dir=str(tmp_path),
    )
    try:
        asyncio.run(provider.generate_image("a prompt"))
        raise AssertionError("should have raised")
    except ImageProviderError:
        pass


def test_generation_that_never_completes_times_out(tmp_path, monkeypatch):
    def queue(request):
        return httpx.Response(200, json={"prompt_id": "never-completes", "node_errors": {}})

    def history(request):
        return httpx.Response(200, json={})

    provider = ComfyUIImageProvider(
        base_url="http://fake-comfyui:8188", workflow_path=REAL_WORKFLOW_PATH,
        timeout_seconds=0.5, output_dir=str(tmp_path / "out"),
    )
    transport = _MockTransport({"POST /prompt": queue, "GET /history/never-completes": history})
    _patch_httpx_client(monkeypatch, transport)
    try:
        asyncio.run(provider.generate_image("a prompt"))
        raise AssertionError("should have raised")
    except ImageProviderError as e:
        assert "timeout" in str(e).lower()


def test_missing_image_output_in_history_raises(tmp_path, monkeypatch):
    def queue(request):
        return httpx.Response(200, json={"prompt_id": "test-123", "node_errors": {}})

    def history(request):
        return httpx.Response(200, json={"test-123": {"status": {"status_str": "success"}, "outputs": {"78": {}}}})

    provider, transport = _provider_with_transport(tmp_path, {"POST /prompt": queue, "GET /history/test-123": history})
    _patch_httpx_client(monkeypatch, transport)
    try:
        asyncio.run(provider.generate_image("a prompt"))
        raise AssertionError("should have raised")
    except ImageProviderError as e:
        assert "no image output" in str(e).lower()


def test_invalid_downloaded_bytes_raises(tmp_path, monkeypatch):
    provider, transport = _provider_with_transport(tmp_path, _success_handlers(image_bytes=b"NOT_A_REAL_PNG"))
    _patch_httpx_client(monkeypatch, transport)
    try:
        asyncio.run(provider.generate_image("a prompt"))
        raise AssertionError("should have raised")
    except ImageProviderError as e:
        assert "invalid" in str(e).lower() or "not a valid image" in str(e).lower()


def test_empty_downloaded_bytes_raises(tmp_path, monkeypatch):
    provider, transport = _provider_with_transport(tmp_path, _success_handlers(image_bytes=b""))
    _patch_httpx_client(monkeypatch, transport)
    try:
        asyncio.run(provider.generate_image("a prompt"))
        raise AssertionError("should have raised")
    except ImageProviderError:
        pass


# --- Output path / metadata (retained, updated for the real model name) ----


def test_output_path_is_under_the_configured_output_dir(tmp_path, monkeypatch):
    provider, transport = _provider_with_transport(tmp_path, _success_handlers())
    _patch_httpx_client(monkeypatch, transport)
    result = asyncio.run(provider.generate_image("a prompt"))
    from pathlib import Path

    assert Path(result.image_path).parent == tmp_path / "out"
    assert result.image_path.endswith(".png")


def test_result_metadata_includes_prompt_id_seed_and_validated_model(tmp_path, monkeypatch):
    provider, transport = _provider_with_transport(tmp_path, _success_handlers())
    _patch_httpx_client(monkeypatch, transport)
    result = asyncio.run(provider.generate_image("a specific prompt"))
    assert result.metadata["prompt_id"] == "test-prompt-123"
    assert result.metadata["seed"] == _derive_seed("a specific prompt")
    assert "flux-2-klein-4b" in result.metadata["model"]


def test_result_width_height_reflect_actual_decoded_image(tmp_path, monkeypatch):
    provider, transport = _provider_with_transport(
        tmp_path, _success_handlers(image_bytes=_real_png_bytes(width=640, height=480))
    )
    _patch_httpx_client(monkeypatch, transport)
    result = asyncio.run(provider.generate_image("a prompt"))
    assert (result.width, result.height) == (640, 480)


def test_generation_time_is_a_real_non_negative_measurement(tmp_path, monkeypatch):
    provider, transport = _provider_with_transport(tmp_path, _success_handlers())
    _patch_httpx_client(monkeypatch, transport)
    result = asyncio.run(provider.generate_image("a prompt"))
    assert result.generation_time_seconds >= 0


# --- Isolation / interface conformance (unchanged from Prompt 5) ------------


def test_provider_never_imports_chess2fight_specific_modules():
    import ast
    import inspect

    import core.image_providers.comfyui as module

    tree = ast.parse(inspect.getsource(module))
    imported_modules = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}

    forbidden_prefixes = (
        "products.chess2fight.pgn_analyzer", "products.chess2fight.battle_director",
        "products.chess2fight.cinematic", "fastapi", "chess",
    )
    for imported in imported_modules:
        assert not imported.startswith(forbidden_prefixes), f"forbidden import: {imported}"


def test_provider_satisfies_the_image_provider_interface():
    assert issubclass(ComfyUIImageProvider, ImageProvider)


def test_provider_never_imports_router_or_registry_or_mock_by_name():
    import ast
    import inspect

    import core.image_providers.comfyui as module

    tree = ast.parse(inspect.getsource(module))
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store)
    }
    assert not (referenced_names & {"ImageRouter", "ImageProviderRegistry", "MockImageProvider"})


def test_comfyui_image_request_error_never_escapes_generate_image(tmp_path, monkeypatch):
    provider, _ = _provider_with_transport(tmp_path, {})

    async def _raising_queue(*args, **kwargs):
        raise ComfyUIImageRequestError("simulated deep failure")

    monkeypatch.setattr(provider, "_queue_prompt", _raising_queue)
    try:
        asyncio.run(provider.generate_image("a prompt"))
        raise AssertionError("should have raised ImageProviderError")
    except ImageProviderError as e:
        assert "simulated deep failure" in str(e)


def test_mock_image_provider_behavior_unchanged(tmp_path):
    provider = MockImageProvider(output_dir=str(tmp_path))
    result = asyncio.run(provider.generate_image("a test prompt"))
    assert result.provider == "MockImageProvider"
    import os

    assert os.path.exists(result.image_path)
