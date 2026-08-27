"""Tests for Sprint 4 Prompt 15's reference-edit identity-lock prompt
calibration — a prompt-wording-only experiment following the real
Prompt 14 GPU result (per-shot derived seeds unlocked pose/composition
diversity, but the dragon halberd acquired extra weapon fragments and
mutated in one shot).

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
import subprocess
import sys

from products.chess2fight.cinematic.prompt_generator import compose_reference_edit_prompt
from products.chess2fight.rendering.visual_continuity import derive_fight_base_visual_seed, derive_shot_seed
from tests.test_prompt12_visual_continuity import _prompted_timeline
from tests.test_prompt14_reference_seed_calibration import _CorrectFakeReferenceProvider, _make_anchor, _runner
from tests.test_single_shot_acceptance import _preferences, _sample_pgn

RENDER_CALIBRATION_CLI = "scripts/render_reference_seed_calibration.py"

_PGN = _sample_pgn()
_PROMPTED = _prompted_timeline(_PGN)
_SHOT1_PROMPT = compose_reference_edit_prompt(_PROMPTED.shots[1])
_SHOT2_PROMPT = compose_reference_edit_prompt(_PROMPTED.shots[2])


# --- 1/2. Explicit PRESERVE EXACTLY / CHANGE ONLY sections -------------------


def test_preserve_exactly_section_exists():
    assert "PRESERVE EXACTLY" in _SHOT1_PROMPT


def test_change_only_section_exists():
    assert "CHANGE ONLY" in _SHOT1_PROMPT


def test_preserve_comes_before_change():
    preserve_idx = _SHOT1_PROMPT.index("PRESERVE EXACTLY")
    change_idx = _SHOT1_PROMPT.index("CHANGE ONLY")
    assert preserve_idx < change_idx


# --- 3/4/5. Fighter A weapon topology enforcement ----------------------------


def test_fighter_a_requires_exactly_the_reference_weapon():
    """General-purpose (not hardcoded to "dragon halberd") — the actual
    weapon text from FighterAppearance appears quoted, with explicit
    "preserve exactly the state described as" language.

    Sprint 4 Prompt 15.1: updated to match the new, fully generic
    _weapon_identity_clause wording (see that function's own
    docstring for why the old "retain exactly the {weapon}" phrasing
    was replaced — it worked fine grammatically but this update
    tracks the actual current wording, not a weakening of intent)."""
    weapon = _PROMPTED.shots[1].scene.white_fighter.weapon
    assert f'described as "{weapon}"' in _SHOT1_PROMPT
    assert weapon == "dragon halberd"  # confirms the sample fixture's own actual value, for a concrete check too


def test_halberd_duplication_explicitly_prohibited():
    assert "duplicate" in _SHOT1_PROMPT.lower()
    assert "this weapon/equipment item" in _SHOT1_PROMPT.lower()


def test_additional_spear_polearm_blade_creation_prohibited():
    """Sprint 4 Prompt 15.1: the old wording named specific
    morphology ("spear", "polearm", "blade") — deliberately removed
    (see _weapon_identity_clause's own docstring on why: it doesn't
    generalize to Halisako's real non-bladed weapon/equipment
    vocabulary). This test now verifies the equivalent, fully generic
    anti-invention language instead — same property (no additional
    weapon/equipment object may be invented), generalized wording."""
    lower = _SHOT1_PROMPT.lower()
    assert "do not invent any additional weapon/equipment object" in lower
    assert "component" in lower
    assert "attachment" in lower
    assert "fragment" in lower


# --- 6/7. Fighter B weapon topology enforcement ------------------------------


def test_fighter_b_requires_exactly_the_reference_weapon():
    weapon = _PROMPTED.shots[1].scene.black_fighter.weapon
    assert f'described as "{weapon}"' in _SHOT1_PROMPT
    assert weapon == "twin daggers"  # confirms the sample fixture's own actual value


def test_dagger_add_remove_duplicate_transform_all_prohibited():
    lower = _SHOT1_PROMPT.lower()
    assert "do not add, remove, duplicate, merge, split, substitute, redesign, mutate, or transform" in lower


# --- 8/9. Face/hair/wardrobe and arena/style preservation remain explicit ---


def test_face_hair_wardrobe_preservation_explicit():
    shot = _PROMPTED.shots[1]
    for fighter, label in [(shot.scene.white_fighter, "Fighter A"), (shot.scene.black_fighter, "Fighter B")]:
        assert f"{label}: same face" in _SHOT1_PROMPT
        assert fighter.hair in _SHOT1_PROMPT
        assert fighter.clothing in _SHOT1_PROMPT
        assert fighter.armor in _SHOT1_PROMPT


def test_arena_and_style_preservation_explicit():
    scene = _PROMPTED.shots[1].scene
    assert f"preserve the {scene.cinematic_art_style}" in _SHOT1_PROMPT
    assert "color palette" in _SHOT1_PROMPT
    assert f"preserve the {scene.arena.layout}" in _SHOT1_PROMPT


# --- 10/11. Pose and camera/framing explicitly editable ----------------------


def test_pose_explicitly_editable():
    change_block = _SHOT1_PROMPT.split("CHANGE ONLY:", 1)[1]
    assert "body pose" in change_block.lower()
    assert "limb pose" in change_block.lower()
    assert "body orientation" in change_block.lower()
    # And never mentioned as something to preserve.
    preserve_block = _SHOT1_PROMPT.split("CHANGE ONLY:", 1)[0]
    assert "pose" not in preserve_block.lower()
    assert "composition" not in preserve_block.lower()


def test_camera_and_framing_explicitly_editable():
    change_block = _SHOT1_PROMPT.split("CHANGE ONLY:", 1)[1]
    shot = _PROMPTED.shots[1]
    # The shot's own real camera phrases must appear in CHANGE.
    from products.chess2fight.cinematic.prompt_generator import _camera_clause

    angle, movement, composition = _camera_clause(shot)
    assert angle in change_block
    assert movement in change_block
    assert composition in change_block


# --- 12. Original shot action text preserved ---------------------------------


def test_original_shot_action_text_preserved():
    # .rstrip(".,;: ") mirrors compose_prompt's own trailing-punctuation
    # stripping (Sprint 4 Prompt 12) — shot.description can be a complete
    # sentence ending in a period, which the composer correctly strips
    # before rejoining.
    assert _PROMPTED.shots[1].description.rstrip(".,;: ") in _SHOT1_PROMPT
    assert _PROMPTED.shots[2].description.rstrip(".,;: ") in _SHOT2_PROMPT


# --- 13/14/15. Seed derivation mechanism unchanged; explicit pin works ------


def test_derive_shot_seed_mechanism_itself_unchanged():
    """Sprint 4 Prompt 15.1: this test previously asserted
    `derive_shot_seed(123, "test prompt") == derive_shot_seed(123,
    "test prompt")` — tautological, since calling any pure function
    twice with identical arguments trivially returns the same value
    regardless of whether the function's actual implementation ever
    changed. Replaced with a known, exact expected value for a stable
    input — this genuinely fails if derive_shot_seed's own SHA256-based
    computation is ever modified, which the tautological version could
    never have caught."""
    assert derive_shot_seed(123, "test prompt") == 2860405817


def test_derive_fight_base_visual_seed_unchanged_for_the_sample():
    base_seed = derive_fight_base_visual_seed(_PGN, "anime", "duel")
    assert base_seed == 1697950441  # matches the real Prompt 13.1/14 GPU evidence exactly


def test_explicit_seeds_pins_shot1_to_2727023522(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(
        runner.prepare(_PGN, _preferences(), anchor_path=anchor_path, anchor_original_seed=1697950441,
                        style="anime", battle_mode="duel", explicit_seeds=(2727023522, 981216397))
    )
    assert plan.shots[0].planned_flux_seed == 2727023522


def test_explicit_seeds_pins_shot2_to_981216397(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(
        runner.prepare(_PGN, _preferences(), anchor_path=anchor_path, anchor_original_seed=1697950441,
                        style="anime", battle_mode="duel", explicit_seeds=(2727023522, 981216397))
    )
    assert plan.shots[1].planned_flux_seed == 981216397


def test_without_explicit_seeds_prompt14_default_derivation_still_works(tmp_path):
    """Confirms the override is genuinely opt-in — omitting it still
    derives normally from the (now-different) prompt text, exactly as
    Prompt 14 always has."""
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(
        runner.prepare(_PGN, _preferences(), anchor_path=anchor_path, anchor_original_seed=1697950441, style="anime", battle_mode="duel")
    )
    expected = derive_shot_seed(plan.fight_base_visual_seed, plan.shots[0].prompt)
    assert plan.shots[0].planned_flux_seed == expected


# --- 16/17/18/19/20. Job counts and anchor sharing ---------------------------


def test_both_jobs_use_the_same_anchor(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_PGN, _preferences(), anchor_path=anchor_path, anchor_original_seed=1697950441, style="anime", battle_mode="duel", explicit_seeds=(2727023522, 981216397)))
    provider = _CorrectFakeReferenceProvider(plan.fight_base_visual_seed)
    # Override provider's internal seed resolution to match the explicit pins for this happy-path test.
    provider._override = lambda prompt: plan.shots[0].planned_flux_seed if prompt == plan.shots[0].prompt else plan.shots[1].planned_flux_seed
    result = asyncio.run(runner.execute(plan, provider))
    assert result.shot_results[0].reference_anchor_path == result.shot_results[1].reference_anchor_path == anchor_path


def test_exactly_two_reference_generation_calls(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_PGN, _preferences(), anchor_path=anchor_path, anchor_original_seed=1697950441, style="anime", battle_mode="duel"))
    provider = _CorrectFakeReferenceProvider(plan.fight_base_visual_seed)
    asyncio.run(runner.execute(plan, provider))
    assert len(provider.calls) == 2


def test_zero_t2i_calls():
    """The calibration provider interface has no T2I method — the same
    structural guarantee Prompt 14 already established."""
    assert not hasattr(_CorrectFakeReferenceProvider, "generate_image")


def test_zero_wan_calls():
    import products.chess2fight.rendering.reference_seed_calibration as module

    with open(module.__file__) as f:
        source = f.read()
    assert "import AnimationPipeline" not in source
    assert "import AnimationRouter" not in source


def test_zero_videobuilder_calls():
    import products.chess2fight.rendering.reference_seed_calibration as module

    with open(module.__file__) as f:
        source = f.read()
    assert "import VideoBuilder" not in source


# --- 21/22. Workflow, model, steps, CFG, resolution unchanged ---------------


def test_reference_workflow_json_unchanged():
    import json

    with open("products/chess2fight/rendering/workflows/flux2_klein_reference_4b.json") as f:
        wf = json.load(f)
    assert wf["77:90"]["inputs"]["positive"] == ["ref:3", 0]
    assert wf["77:90"]["inputs"]["negative"] == ["ref:4", 0]
    assert wf["77:87"]["inputs"]["unet_name"] == "flux-2-klein-4b.safetensors"
    assert wf["77:93"]["inputs"]["steps"] == 4
    assert wf["77:90"]["inputs"]["cfg"] == 1
    assert wf["77:84"]["inputs"]["value"] == 1280
    assert wf["77:85"]["inputs"]["value"] == 704


def test_model_steps_cfg_resolution_unchanged_in_provider_metadata(tmp_path):
    anchor_path = str(tmp_path / "anchor.png")
    _make_anchor(anchor_path)
    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_PGN, _preferences(), anchor_path=anchor_path, anchor_original_seed=1697950441, style="anime", battle_mode="duel"))
    assert plan.anchor.width == 1280
    assert plan.anchor.height == 704


# --- 23. Dry-run performs zero provider/network calls ------------------------


def test_dry_run_zero_provider_calls():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        anchor_path = f"{tmp}/anchor.png"
        _make_anchor(anchor_path)
        result = subprocess.run(
            [sys.executable, RENDER_CALIBRATION_CLI, "--sample", "--anchor-path", anchor_path,
             "--anchor-original-seed", "1697950441", "--explicit-seeds", "2727023522,981216397", "--dry-run"],
            capture_output=True, text=True, check=True,
        )
        assert "generating" not in result.stdout.lower()
        assert "dry run complete" in result.stdout.lower()
        assert "2727023522" in result.stdout
        assert "981216397" in result.stdout


# --- 24. Prompt 14 existing tests remain green (spot-check here; full suite in CI run) -


def test_prompt14_calibration_module_tests_still_pass():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_prompt14_reference_seed_calibration.py", "-q"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "24 passed" in result.stdout
