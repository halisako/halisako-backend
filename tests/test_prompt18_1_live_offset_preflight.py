"""Tests for Sprint 4 Prompt 18.1's fix to
`check_reference_method_node_availability()`'s combo-schema parsing.

Root cause fixed: the real, live ComfyUI `/object_info` response for
`FluxKontextMultiReferenceLatentMethod`'s `reference_latents_method`
input is `["COMBO", {..., "options": [...]}]` — a type-tag string
followed by a config dict — not the `[[...], {}]` shape this
codebase's own prior parsing exclusively assumed. On the real shape,
`method_input_spec[0]` evaluated to the string `"COMBO"`, the
`isinstance(..., list)` guard silently failed, and the entire
capability check was skipped with no problem reported — even when the
required candidate genuinely wasn't supported.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio

import httpx

from core.config import get_settings
from products.chess2fight.rendering.acceptance_preflight import (
    _extract_combo_options,
    check_reference_method_node_availability,
    preflight_check,
)
from tests.test_prompt13_1_workflow_hardening import _MockTransport, _model_visibility_handlers, _patch_client

# The exact real, live schema from this task's own audit report —
# independently observed on an RTX 4090 RunPod instance.
_REAL_LIVE_SCHEMA_ALL_METHODS = {
    "FluxKontextMultiReferenceLatentMethod": {
        "input": {
            "required": {
                "reference_latents_method": [
                    "COMBO",
                    {
                        "advanced": True,
                        "multiselect": False,
                        "options": ["offset", "index", "uxo/uno", "index_timestep_zero"],
                    },
                ]
            }
        }
    }
}


def _real_schema_with_options(options: list[str]) -> dict:
    return {
        "FluxKontextMultiReferenceLatentMethod": {
            "input": {
                "required": {
                    "reference_latents_method": [
                        "COMBO",
                        {"advanced": True, "multiselect": False, "options": options},
                    ]
                }
            }
        }
    }


class _SingleResponseTransport(httpx.AsyncBaseTransport):
    def __init__(self, status_code: int, json_body=None):
        self._status_code = status_code
        self._json_body = json_body

    async def handle_async_request(self, request):
        if self._json_body is None:
            return httpx.Response(self._status_code)
        return httpx.Response(self._status_code, json=self._json_body)


def _patch_single_response(monkeypatch, status_code: int, json_body=None):
    import products.chess2fight.rendering.acceptance_preflight as preflight_module

    real_client = httpx.AsyncClient
    transport = _SingleResponseTransport(status_code, json_body)
    monkeypatch.setattr(
        preflight_module.httpx, "AsyncClient", lambda *a, **kw: real_client(*a, **{**kw, "transport": transport})
    )


# --- _extract_combo_options: both schema shapes ------------------------------


def test_extract_combo_options_real_live_schema():
    spec = ["COMBO", {"advanced": True, "multiselect": False, "options": ["offset", "index"]}]
    assert _extract_combo_options(spec) == ["offset", "index"]


def test_extract_combo_options_legacy_schema():
    spec = [["offset", "index", "uxo/uno", "index_timestep_zero"], {}]
    assert _extract_combo_options(spec) == ["offset", "index", "uxo/uno", "index_timestep_zero"]


def test_extract_combo_options_unparseable_returns_none():
    assert _extract_combo_options(["COMBO", {"advanced": True}]) is None  # no "options" key
    assert _extract_combo_options("not a list at all") is None
    assert _extract_combo_options([]) is None
    assert _extract_combo_options(["SOMETHING_ELSE", {}]) is None


# --- 1. Real COMBO/options schema with offset -> PASS ------------------------


def test_real_schema_with_offset_passes(monkeypatch):
    _patch_single_response(monkeypatch, 200, _REAL_LIVE_SCHEMA_ALL_METHODS)
    problems = asyncio.run(check_reference_method_node_availability(get_settings(), ["offset"]))
    assert problems == []


# --- 2. Real COMBO/options schema, only ["index"], offset required -> HARD FAIL


def test_real_schema_only_index_offset_required_hard_fails(monkeypatch):
    """The exact regression this task fixes: before the fix, this
    scenario silently returned [] instead of a hard problem."""
    _patch_single_response(monkeypatch, 200, _real_schema_with_options(["index"]))
    problems = asyncio.run(check_reference_method_node_availability(get_settings(), ["offset"]))
    assert len(problems) == 1
    assert "offset" in problems[0]
    assert "not among" in problems[0]


# --- 3. Missing options / unparseable method input -> HARD FAIL -------------


def test_missing_options_key_hard_fails(monkeypatch):
    schema = {
        "FluxKontextMultiReferenceLatentMethod": {
            "input": {"required": {"reference_latents_method": ["COMBO", {"advanced": True}]}}  # no "options"
        }
    }
    _patch_single_response(monkeypatch, 200, schema)
    problems = asyncio.run(check_reference_method_node_availability(get_settings(), ["offset"]))
    assert len(problems) == 1
    assert "could not be reliably parsed" in problems[0]


def test_reference_latents_method_key_entirely_absent_hard_fails(monkeypatch):
    schema = {"FluxKontextMultiReferenceLatentMethod": {"input": {"required": {}}}}
    _patch_single_response(monkeypatch, 200, schema)
    problems = asyncio.run(check_reference_method_node_availability(get_settings(), ["offset"]))
    assert len(problems) == 1
    assert "does not expose" in problems[0]


# --- 4. Missing node -> HARD FAIL --------------------------------------------


def test_missing_node_hard_fails(monkeypatch):
    _patch_single_response(monkeypatch, 200, {})
    problems = asyncio.run(check_reference_method_node_availability(get_settings(), ["offset"]))
    assert len(problems) == 1
    assert "not found" in problems[0]


# --- 5. Unreachable /object_info -> HARD FAIL --------------------------------


def test_unreachable_object_info_hard_fails(monkeypatch):
    _patch_single_response(monkeypatch, 404)
    problems = asyncio.run(check_reference_method_node_availability(get_settings(), ["offset"]))
    assert len(problems) == 1
    assert "Could not confirm" in problems[0]


# --- 6. Legacy nested-list representation remains supported -----------------


def test_legacy_representation_still_supported(monkeypatch):
    schema = {
        "FluxKontextMultiReferenceLatentMethod": {
            "input": {"required": {"reference_latents_method": [
                ["offset", "index", "uxo/uno", "index_timestep_zero"], {}
            ]}}
        }
    }
    _patch_single_response(monkeypatch, 200, schema)
    problems = asyncio.run(check_reference_method_node_availability(get_settings(), ["offset"]))
    assert problems == []


# --- 7. Prompt 16 calibration still validates arbitrary selected methods ----


def test_calibration_arbitrary_method_selection_still_works(monkeypatch):
    """Confirms the fix is generic — not hardcoded to "offset" as the
    only parsable value; whatever candidate_methods the caller passes
    is what gets checked."""
    _patch_single_response(monkeypatch, 200, _real_schema_with_options(["offset", "index"]))
    problems = asyncio.run(
        check_reference_method_node_availability(get_settings(), ["uxo/uno", "index_timestep_zero"])
    )
    assert len(problems) == 2
    assert "uxo/uno" in problems[0] or "uxo/uno" in problems[1]
    assert "index_timestep_zero" in problems[0] or "index_timestep_zero" in problems[1]


def test_calibration_all_three_candidates_pass_when_all_supported(monkeypatch):
    _patch_single_response(monkeypatch, 200, _REAL_LIVE_SCHEMA_ALL_METHODS)
    problems = asyncio.run(
        check_reference_method_node_availability(get_settings(), ["offset", "uxo/uno", "index_timestep_zero"])
    )
    assert problems == []


# --- 8. Production preflight still requires only offset ---------------------


def test_production_preflight_requires_only_offset(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _model_visibility_handlers()
    handlers["GET /object_info/ReferenceLatent"] = lambda r: httpx.Response(200, json={"ReferenceLatent": {"input": {}}})
    handlers["GET /object_info/VAEEncode"] = lambda r: httpx.Response(200, json={"VAEEncode": {"input": {}}})
    # Real live schema, only offset+index supported (no uxo/uno, no
    # index_timestep_zero) — production preflight must still pass,
    # since it only ever requires "offset".
    handlers["GET /object_info/FluxKontextMultiReferenceLatentMethod"] = lambda r: httpx.Response(
        200, json=_real_schema_with_options(["offset", "index"])
    )
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(preflight_check(settings, check_reference_workflow=True))
    assert problems == []


def test_production_preflight_fails_when_offset_unsupported_on_real_schema(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "image_provider", "comfyui")
    monkeypatch.setattr(settings, "animation_provider", "mock")

    handlers = _model_visibility_handlers()
    handlers["GET /object_info/ReferenceLatent"] = lambda r: httpx.Response(200, json={"ReferenceLatent": {"input": {}}})
    handlers["GET /object_info/VAEEncode"] = lambda r: httpx.Response(200, json={"VAEEncode": {"input": {}}})
    handlers["GET /object_info/FluxKontextMultiReferenceLatentMethod"] = lambda r: httpx.Response(
        200, json=_real_schema_with_options(["index"])  # no offset — the exact live bug scenario
    )
    _patch_client(monkeypatch, _MockTransport(handlers))

    problems, warnings = asyncio.run(preflight_check(settings, check_reference_workflow=True))
    assert len(problems) == 1
    assert "offset" in problems[0]


# --- 9. No provider generation occurs after a failed preflight --------------


def test_no_generation_after_failed_preflight(monkeypatch):
    """A failed check_reference_method_node_availability() result
    means the CLI itself never proceeds to call execute() — verified
    at the unit level here: the function's own return value is a
    non-empty problem list, which every calling CLI (Sprint 4 Prompt
    16/18's own render_reference_method_calibration.py, and
    preflight_check's own production callers) treats as "stop before
    generation" — confirmed by inspecting both CLI scripts' own
    control flow, not re-executing them end-to-end here."""
    _patch_single_response(monkeypatch, 200, _real_schema_with_options(["index"]))
    problems = asyncio.run(check_reference_method_node_availability(get_settings(), ["offset"]))
    assert problems != []  # a non-empty result is exactly what callers gate generation on

    with open("scripts/render_reference_method_calibration.py") as f:
        cli_source = f.read()
    assert "if method_problems:" in cli_source
    assert "return 1" in cli_source


# --- 10. Ordinary tests never contact real ComfyUI ---------------------------


def test_preflight_module_has_no_hardcoded_network_target():
    with open("products/chess2fight/rendering/acceptance_preflight.py") as f:
        source = f.read()
    assert "requests.get(" not in source
    # Only settings-derived base_url construction, never a hardcoded literal host.
    assert 'f"{base_url}' in source
