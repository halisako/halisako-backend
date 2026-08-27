"""Tests for Sprint 4 Prompt 15.1's critical live-wiring fix and prompt
generalization.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
import importlib.util
import io
import json
import sys

import httpx
import pytest
from PIL import Image

from core.ai_router import TemplateProvider
from core.config import get_settings
import core.image_providers.comfyui as _comfyui_module
from products.chess2fight.cinematic.prompt_generator import _weapon_identity_clause, compose_reference_edit_prompt
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.reference_seed_calibration import (
    AmbiguousPlannedSeedError,
    CalibrationShotPlan,
    ReferenceSeedCalibrationRunner,
    UnplannedPromptError,
    build_plan_seed_override,
)
from products.chess2fight.rendering.visual_continuity import VisualSeedPolicy, build_seed_override
from tests.test_prompt12_visual_continuity import _prompted_timeline
from tests.test_prompt14_reference_seed_calibration import _make_anchor
from tests.test_single_shot_acceptance import _preferences, _sample_pgn

_PGN = _sample_pgn()

# Captured once, at module import time — before any test can patch
# core.image_providers.comfyui's own httpx.AsyncClient. Always wrapping
# THIS reference (never a fresh `httpx.AsyncClient` lookup inside the
# repeatedly-called helper below) is what makes repeated calls within
# this module correctly route to each call's own transport, instead of
# a second call's factory silently wrapping the first call's
# already-patched one. Same fix already established in
# tests/test_comfyui_image_provider.py's own _patch_httpx_client.
_REAL_HTTPX_ASYNC_CLIENT = httpx.AsyncClient


def _runner(tmp_path):
    return ReferenceSeedCalibrationRunner(TemplateProvider(), asset_manager=AssetManager(storage_root=str(tmp_path / "storage")))


async def _pinned_plan(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = await runner.prepare(
        _PGN, _preferences(), anchor_path=anchor_path, anchor_original_seed=1697950441,
        style="anime", battle_mode="duel", explicit_seeds=(2727023522, 981216397),
    )
    return plan


# --- 1/2. Explicit seed plan values -----------------------------------------


def test_explicit_seed_plan_shot1_is_2727023522(tmp_path):
    plan = asyncio.run(_pinned_plan(tmp_path))
    assert plan.shots[0].planned_flux_seed == 2727023522


def test_explicit_seed_plan_shot2_is_981216397(tmp_path):
    plan = asyncio.run(_pinned_plan(tmp_path))
    assert plan.shots[1].planned_flux_seed == 981216397


# --- 3/4/5. The REAL seed_override callable resolves the pinned values -----


def test_plan_seed_override_resolves_shot1_prompt_to_pinned_seed(tmp_path):
    plan = asyncio.run(_pinned_plan(tmp_path))
    seed_override = build_plan_seed_override(plan)
    assert seed_override(plan.shots[0].prompt) == 2727023522


def test_plan_seed_override_resolves_shot2_prompt_to_pinned_seed(tmp_path):
    plan = asyncio.run(_pinned_plan(tmp_path))
    seed_override = build_plan_seed_override(plan)
    assert seed_override(plan.shots[1].prompt) == 981216397


def test_plan_seed_override_does_not_resolve_to_new_derived_values(tmp_path):
    """The exact regression this task fixes: the buggy wiring would
    have resolved to the NEW-prompt derived values (1222993584 /
    3013132441), not the pinned ones."""
    plan = asyncio.run(_pinned_plan(tmp_path))
    seed_override = build_plan_seed_override(plan)
    assert seed_override(plan.shots[0].prompt) != 1222993584
    assert seed_override(plan.shots[1].prompt) != 3013132441


# --- 6/7. Unknown/ambiguous prompt handling ----------------------------------


def test_unknown_prompt_fails_loudly_before_any_network_request(tmp_path):
    plan = asyncio.run(_pinned_plan(tmp_path))
    seed_override = build_plan_seed_override(plan)
    with pytest.raises(UnplannedPromptError):
        seed_override("a prompt that was never part of this plan at all")


def test_ambiguous_pinned_seeds_fail_before_generation():
    ambiguous_shots = [
        CalibrationShotPlan(timeline_index=1, shot_id="s1", prompt="SAME PROMPT TEXT", planned_flux_seed=111),
        CalibrationShotPlan(timeline_index=2, shot_id="s2", prompt="SAME PROMPT TEXT", planned_flux_seed=222),
    ]
    from products.chess2fight.rendering.reference_seed_calibration import CalibrationAnchor, ReferenceSeedCalibrationPlan

    fake_plan = ReferenceSeedCalibrationPlan(
        run_id="test_run", fight_base_visual_seed=1,
        anchor=CalibrationAnchor(path="/fake/path.png", sha256="a" * 64, width=1280, height=704, original_seed=1),
        selected_shot_indices=[1, 2], shots=ambiguous_shots, expected_comfyui_jobs=2,
    )
    with pytest.raises(AmbiguousPlannedSeedError):
        build_plan_seed_override(fake_plan)


# --- 8. Without --explicit-seeds, Prompt 14 DERIVED behavior is unchanged --


def test_default_derived_policy_unchanged_without_explicit_seeds(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(
        runner.prepare(_PGN, _preferences(), anchor_path=anchor_path, anchor_original_seed=1697950441, style="anime", battle_mode="duel")
    )
    # This is exactly Prompt 14's own established resolution — untouched.
    expected_override = build_seed_override(VisualSeedPolicy.DERIVED, plan.fight_base_visual_seed)
    assert expected_override(plan.shots[0].prompt) == plan.shots[0].planned_flux_seed
    assert expected_override(plan.shots[1].prompt) == plan.shots[1].planned_flux_seed


# --- 9/10. Mocked non-dry-run CLI wiring + end-to-end metadata agreement ---


def _run_cli_with_mocked_comfyui(anchor_path: str, explicit_seeds: str | None, manifest_path: str):
    """Runs the actual CLI's _main() in-process, with httpx mocked at
    the transport level — exercising the REAL
    ComfyUIImageProvider/seed_override wiring the CLI itself
    constructs, not a hand-rolled substitute."""

    class RoutedTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.submitted_seeds = []
            self.job_counter = 0

        async def handle_async_request(self, request):
            await request.aread()
            if request.method == "POST" and request.url.path == "/upload/image":
                return httpx.Response(200, json={"name": "anchor.png", "subfolder": "", "type": "input"})
            if request.method == "POST" and request.url.path == "/prompt":
                body = json.loads(request.content)
                seed = body["prompt"]["77:86"]["inputs"]["noise_seed"]
                self.submitted_seeds.append(seed)
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

    transport = RoutedTransport()

    def _client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_HTTPX_ASYNC_CLIENT(*args, **kwargs)

    _comfyui_module.httpx.AsyncClient = _client_factory

    argv = [
        "render_reference_seed_calibration.py", "--sample", "--anchor-path", anchor_path,
        "--anchor-original-seed", "1697950441", "--skip-preflight", "--manifest-path", manifest_path,
    ]
    if explicit_seeds is not None:
        argv += ["--explicit-seeds", explicit_seeds]

    sys.argv = argv
    spec = importlib.util.spec_from_file_location("cli_module_test", "scripts/render_reference_seed_calibration.py")
    cli_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli_module)
    exit_code = asyncio.run(cli_module._main())
    return exit_code, transport.submitted_seeds


def test_mocked_cli_wiring_submits_pinned_seeds_not_derived_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "image_provider", "comfyui")
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    manifest_path = str(tmp_path / "manifest.json")

    exit_code, submitted_seeds = _run_cli_with_mocked_comfyui(anchor_path, "2727023522,981216397", manifest_path)

    assert exit_code == 0
    assert submitted_seeds == [2727023522, 981216397]
    assert 1222993584 not in submitted_seeds
    assert 3013132441 not in submitted_seeds


def test_mocked_cli_end_to_end_metadata_seed_agrees_with_planned(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "image_provider", "comfyui")
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    manifest_path = str(tmp_path / "manifest.json")

    exit_code, submitted_seeds = _run_cli_with_mocked_comfyui(anchor_path, "2727023522,981216397", manifest_path)
    assert exit_code == 0

    manifest = json.loads(open(manifest_path).read())
    for shot_entry in manifest["shots"]:
        assert shot_entry["planned_flux_seed"] == shot_entry["actual_flux_seed"]
    assert {s["planned_flux_seed"] for s in manifest["shots"]} == {2727023522, 981216397}


def test_mocked_cli_without_explicit_seeds_uses_derived_policy(tmp_path, monkeypatch):
    """Confirms the CLI's own branch: omitting --explicit-seeds still
    goes through build_seed_override(DERIVED, ...), exactly as
    Prompt 14 established — the fix is additive, not a rewrite of the
    default path."""
    monkeypatch.setattr(get_settings(), "image_provider", "comfyui")
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    manifest_path = str(tmp_path / "manifest.json")

    exit_code, submitted_seeds = _run_cli_with_mocked_comfyui(anchor_path, None, manifest_path)
    assert exit_code == 0
    # No pinning requested — the two submitted seeds are whatever DERIVED
    # policy produces for this run's own (new Prompt 15) prompt text.
    assert len(submitted_seeds) == 2
    assert submitted_seeds[0] != submitted_seeds[1]


# --- Weapon/equipment generalization -----------------------------------------


_WEAPON_EXAMPLES = [
    "dragon halberd", "twin daggers", "dual blades", "assault rifle", "sniper rifle",
    "ballistic shield", "breaching charge", "drone swarm", "energy shield",
    "deflector barrier", "reinforced gauntlets", "impact-resistant armor",
    "fortified cover position", "an unarmed stance",
]


@pytest.mark.parametrize("weapon", _WEAPON_EXAMPLES)
def test_weapon_clause_contains_original_description_verbatim(weapon):
    clause = _weapon_identity_clause(weapon)
    assert f'"{weapon}"' in clause


@pytest.mark.parametrize("weapon", _WEAPON_EXAMPLES)
def test_weapon_clause_has_count_multiplicity_language(weapon):
    assert "count/multiplicity" in _weapon_identity_clause(weapon)


@pytest.mark.parametrize("weapon", _WEAPON_EXAMPLES)
def test_weapon_clause_prohibits_add_remove_duplicate_merge_split_substitute_transform(weapon):
    clause = _weapon_identity_clause(weapon).lower()
    for verb in ["add", "remove", "duplicate", "merge", "split", "substitute", "redesign", "mutate", "transform"]:
        assert verb in clause


@pytest.mark.parametrize("weapon", _WEAPON_EXAMPLES)
def test_weapon_clause_output_never_leaks_a_different_weapon_name(weapon):
    """The function's OUTPUT for one weapon must never mention any
    other weapon from this list — confirms no hardcoding in the
    actual generated text (checking the function's docstring instead
    would false-positive: it legitimately names several of these as
    explanatory examples of why the function generalizes)."""
    clause = _weapon_identity_clause(weapon)
    for other_weapon in _WEAPON_EXAMPLES:
        if other_weapon != weapon:
            assert other_weapon not in clause


@pytest.mark.parametrize("weapon", _WEAPON_EXAMPLES)
def test_weapon_clause_has_no_morphology_specific_requirement(weapon):
    clause = _weapon_identity_clause(weapon).lower()
    for forbidden_term in ["shaft", "blade\"", "same head", "same spikes", "barrel"]:
        assert forbidden_term not in clause


def test_unarmed_stance_produces_grammatically_valid_sentence():
    clause = _weapon_identity_clause("an unarmed stance")
    assert "the an unarmed stance" not in clause.lower()
    assert '"an unarmed stance"' in clause


# --- Facial identity wording --------------------------------------------------


def test_preserve_block_does_not_require_identical_expression():
    prompted = _prompted_timeline(_PGN)
    edit_prompt = compose_reference_edit_prompt(prompted.shots[1])
    preserve_block = edit_prompt.split("CHANGE ONLY:", 1)[0]
    assert "same expression" not in preserve_block.lower()
    assert "facial identity" in preserve_block.lower()


def test_facial_features_text_still_appears_for_identity_traceability():
    prompted = _prompted_timeline(_PGN)
    edit_prompt = compose_reference_edit_prompt(prompted.shots[1])
    scene = prompted.shots[1].scene
    assert scene.white_fighter.facial_features in edit_prompt
    assert scene.black_fighter.facial_features in edit_prompt
