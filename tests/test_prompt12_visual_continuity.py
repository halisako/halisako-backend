"""Tests for Sprint 4 Prompt 12's visual continuity foundation:
prompt composition hygiene, the stable-continuity/shot-specific
composition contract, and the FLUX seed policy.

TEST CATEGORY: MOCKED UNIT / LOCAL INTEGRATION TESTS — no real ComfyUI
server or GPU is ever contacted.
"""

import asyncio
import subprocess
import sys

from core.ai_router import TemplateProvider
from core.image_providers.comfyui import _derive_seed as _derive_flux_seed
from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.battle_mode_engine import generate_battle_mode_intelligence
from products.chess2fight.cinematic.prompt_composer import compose_prompt, compose_prompt_from_blocks
from products.chess2fight.cinematic.prompt_generator import generate_prompts
from products.chess2fight.cinematic.scene_composer import compose_scene
from products.chess2fight.cinematic.timeline_engine import generate_shot_timeline
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.narrative_generator import NarrativeGenerator
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.rendering.visual_continuity import (
    VisualSeedPolicy,
    build_seed_override,
    derive_fight_base_visual_seed,
    derive_shot_seed,
)
from products.chess2fight.schemas import BattleMode
from products.chess2fight.style_engine import generate_style_profile
from tests.test_single_shot_acceptance import _sample_pgn

RENDER_MULTI_SHOT = "scripts/render_multi_shot_acceptance.py"

SCHOLARS_MATE_PGN = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)


def _prompted_timeline(pgn: str, style: str = "anime", battle_mode: BattleMode = BattleMode.DUEL):
    """Runs the real, full six-stage pipeline (analysis through Prompt
    Generator) — not a hand-built fixture — so these tests exercise
    the actual production wiring, same as the real GPU evidence did."""
    analysis = analyze_game(pgn)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    profile = generate_style_profile(battle, combat, style)
    mode_intel = generate_battle_mode_intelligence(battle_mode, combat, battle)
    story = asyncio.run(NarrativeGenerator(TemplateProvider()).generate(analysis, combat, battle, profile, mode_intel))
    timeline = generate_shot_timeline(battle, story)
    composed = compose_scene(timeline, battle, profile, mode_intel)
    return generate_prompts(composed)


# --- 1/2/3. Fighter visual/wardrobe/weapon identity is stable across shots --


def test_fighter_hair_stable_across_every_shot():
    prompted = _prompted_timeline(_sample_pgn())
    hairs = {shot.scene.white_fighter.hair for shot in prompted.shots}
    assert len(hairs) == 1
    hairs_black = {shot.scene.black_fighter.hair for shot in prompted.shots}
    assert len(hairs_black) == 1


def test_fighter_wardrobe_stable_across_every_shot():
    prompted = _prompted_timeline(_sample_pgn())
    clothing = {(shot.scene.white_fighter.clothing, shot.scene.white_fighter.armor) for shot in prompted.shots}
    assert len(clothing) == 1


def test_fighter_weapon_stable_across_every_shot():
    prompted = _prompted_timeline(_sample_pgn())
    weapons = {shot.scene.white_fighter.weapon for shot in prompted.shots}
    assert len(weapons) == 1
    weapons_black = {shot.scene.black_fighter.weapon for shot in prompted.shots}
    assert len(weapons_black) == 1


def test_fighter_identity_text_byte_identical_in_generated_prompts():
    """Not just the underlying FighterAppearance object (which is
    trivially the same object by construction) — the actual composed
    prompt text itself, exactly as sent to FLUX, must be identical."""
    prompted = _prompted_timeline(_sample_pgn())
    fragments = []
    for shot in prompted.shots:
        white = shot.scene.white_fighter
        expected = f"{white.hair}, {white.facial_features}, wearing {white.clothing} and {white.armor}, wielding a {white.weapon}"
        assert expected in shot.image_prompt
        fragments.append(expected)
    assert len(set(fragments)) == 1


# --- 4. Arena identity is stable across shots -------------------------------


def test_arena_identity_stable_across_every_shot():
    prompted = _prompted_timeline(_sample_pgn())
    arenas = {(shot.scene.arena.layout, shot.scene.arena.weather, shot.scene.arena.time_of_day) for shot in prompted.shots}
    assert len(arenas) == 1


# --- 5. Changing fighter focus does not mutate identity descriptors --------


def test_changing_focus_does_not_alter_underlying_identity_only_framing():
    prompted = _prompted_timeline(_sample_pgn())
    from products.chess2fight.cinematic.schemas import ShotFocus

    white_focused = [s for s in prompted.shots if s.focus == ShotFocus.WHITE]
    black_focused = [s for s in prompted.shots if s.focus == ShotFocus.BLACK]
    assert white_focused, "fixture must contain at least one WHITE-focused shot for this test to be meaningful"
    assert black_focused, "fixture must contain at least one BLACK-focused shot for this test to be meaningful"

    white_id = f"{prompted.shots[0].scene.white_fighter.hair}, {prompted.shots[0].scene.white_fighter.weapon}"
    for shot in white_focused + black_focused:
        assert white_id.split(", ")[0] in shot.image_prompt  # hair
        assert white_id.split(", ")[1] in shot.image_prompt  # weapon
        # Both fighters remain mentioned regardless of who's prominent.
        assert shot.scene.black_fighter.hair in shot.image_prompt


def test_prominence_framing_phrase_changes_but_identity_words_do_not():
    prompted = _prompted_timeline(_sample_pgn())
    from products.chess2fight.cinematic.schemas import ShotFocus

    for shot in prompted.shots:
        if shot.focus == ShotFocus.WHITE:
            assert f"a fighter with {shot.scene.white_fighter.hair}" in shot.image_prompt
            assert f"another fighter in the background with {shot.scene.black_fighter.hair}" in shot.image_prompt
        elif shot.focus == ShotFocus.BLACK:
            assert f"a fighter with {shot.scene.black_fighter.hair}" in shot.image_prompt
            assert f"another fighter in the background with {shot.scene.white_fighter.hair}" in shot.image_prompt


# --- 6/7. Shot action and camera remain shot-specific -----------------------


def test_shot_action_is_shot_specific_not_uniform():
    prompted = _prompted_timeline(_sample_pgn())
    descriptions = [shot.description for shot in prompted.shots]
    assert len(set(descriptions)) > 1  # genuinely varies, not a copy-pasted constant


def test_camera_language_is_shot_specific_not_uniform():
    prompted = _prompted_timeline(_sample_pgn())
    camera_pairs = [(shot.camera_angle, shot.camera_motion) for shot in prompted.shots]
    assert len(set(camera_pairs)) > 1


# --- 8/9/10. Prompt composition hygiene --------------------------------------


def test_compose_prompt_correct_spacing():
    result = compose_prompt(["fragment one", "fragment two", "fragment three"])
    assert result == "fragment one, fragment two, fragment three"
    assert "  " not in result  # no double spaces


def test_compose_prompt_no_duplicate_punctuation_from_sentence_fragment():
    """The exact defect class reproduced from the real GPU evidence:
    a fragment that's a complete sentence (ending in '.') followed by
    another fragment must never produce 'word.,'."""
    result = compose_prompt(["a complete sentence.", "the next fragment"])
    assert result == "a complete sentence, the next fragment"
    assert ".," not in result


def test_compose_prompt_no_duplicate_punctuation_various_trailing_marks():
    for trailing in [".", ",", ";", ":", ".,", " . "]:
        result = compose_prompt([f"fragment{trailing}", "next"])
        assert ".," not in result
        assert ",," not in result
        assert ";," not in result


def test_compose_prompt_drops_empty_fragments_without_stray_commas():
    result = compose_prompt(["first", "", "   ", "second"])
    assert result == "first, second"
    assert ", ," not in result
    assert ",," not in result


def test_known_malformed_patterns_eliminated_in_real_pipeline_output():
    """The specific patterns the task named, plus the one genuinely
    confirmed present in the real evidence ('arena.,')."""
    prompted = _prompted_timeline(_sample_pgn())
    for shot in prompted.shots:
        prompt = shot.image_prompt
        assert "arena.," not in prompt
        assert ".," not in prompt
        assert ",," not in prompt
        assert ", ," not in prompt
        # No run of two lowercase words glued together without a space
        # where a fragment boundary should have inserted one (the
        # illustrative "lightis"/"dynamicspeed"/"thescene" class of
        # defect the task named as an example, even though those exact
        # strings were not found in the real evidence supplied).
        for glued in ("lightis", "dynamicspeed", "thescene"):
            assert glued not in prompt.lower()


def test_reproduces_and_fixes_the_exact_real_evidence_defect():
    """Directly reproduces the real three-shot GPU evidence's own
    establishing-shot prompt construction and confirms the specific
    'arena.,' artifact it contained is gone."""
    prompted = _prompted_timeline(_sample_pgn())
    establishing = next(s for s in prompted.shots if s.shot_type.value == "establishing")
    assert "arena.," not in establishing.image_prompt
    assert "temple ruins" in establishing.image_prompt  # the real content is still present, just not malformed


# --- 11. Prompt output is deterministic --------------------------------------


def test_prompt_generation_is_deterministic():
    first = _prompted_timeline(_sample_pgn())
    second = _prompted_timeline(_sample_pgn())
    assert [s.image_prompt for s in first.shots] == [s.image_prompt for s in second.shots]


def test_compose_prompt_from_blocks_is_deterministic_and_order_preserving():
    result = compose_prompt_from_blocks(["a", "b"], ["c"], ["d", "e"])
    assert result == "a, b, c, d, e"
    assert compose_prompt_from_blocks(["a", "b"], ["c"], ["d", "e"]) == result


# --- 12/13. Base visual seed determinism -------------------------------------


def test_same_input_produces_same_base_fight_visual_seed():
    seed1 = derive_fight_base_visual_seed(SCHOLARS_MATE_PGN, "anime", "duel")
    seed2 = derive_fight_base_visual_seed(SCHOLARS_MATE_PGN, "anime", "duel")
    assert seed1 == seed2


def test_different_fights_can_produce_different_base_seeds():
    seed_a = derive_fight_base_visual_seed(SCHOLARS_MATE_PGN, "anime", "duel")
    seed_b = derive_fight_base_visual_seed(_sample_pgn(), "anime", "duel")
    assert seed_a != seed_b


def test_different_style_same_pgn_produces_different_base_seed():
    seed_anime = derive_fight_base_visual_seed(SCHOLARS_MATE_PGN, "anime", "duel")
    seed_fantasy = derive_fight_base_visual_seed(SCHOLARS_MATE_PGN, "fantasy", "duel")
    assert seed_anime != seed_fantasy


def test_base_seed_never_forces_unrelated_fights_together():
    """No hardcoded/global seed constant, no random module usage —
    purely a function of inputs. Checks actual code patterns, not
    prose (the function's own docstring legitimately discusses "no
    random process state" in English, which isn't itself a usage)."""
    import inspect

    source = inspect.getsource(derive_fight_base_visual_seed)
    body = source.split('"""', 2)[-1]  # drop the docstring, check only the code body
    assert "import random" not in body
    assert "random." not in body
    assert "global " not in body


# --- 14/15. Shared and derived seed policies ---------------------------------


def test_shared_policy_gives_every_shot_the_identical_flux_seed():
    base = derive_fight_base_visual_seed(SCHOLARS_MATE_PGN, "anime", "duel")
    override = build_seed_override(VisualSeedPolicy.SHARED, base)
    seeds = [override(f"prompt {i}") for i in range(3)]
    assert len(set(seeds)) == 1
    assert seeds[0] == base


def test_derived_policy_gives_deterministic_but_varying_per_shot_seeds():
    base = derive_fight_base_visual_seed(SCHOLARS_MATE_PGN, "anime", "duel")
    override = build_seed_override(VisualSeedPolicy.DERIVED, base)
    prompts = ["shot 0 prompt", "shot 1 prompt", "shot 2 prompt"]
    seeds_first_run = [override(p) for p in prompts]
    seeds_second_run = [override(p) for p in prompts]
    assert seeds_first_run == seeds_second_run  # deterministic
    assert len(set(seeds_first_run)) == 3  # genuinely varies per shot


def test_derive_shot_seed_directly():
    base = 12345
    seed1 = derive_shot_seed(base, "prompt a")
    seed2 = derive_shot_seed(base, "prompt a")
    seed3 = derive_shot_seed(base, "prompt b")
    assert seed1 == seed2
    assert seed1 != seed3


def test_default_policy_override_is_none():
    assert build_seed_override(VisualSeedPolicy.DEFAULT, 999) is None


# --- 16. Explicit seed override still works ----------------------------------


def test_wan_explicit_seed_override_unchanged(tmp_path):
    """The pre-existing, unchanged AnimationInstruction.seed field."""
    from core.animation_router import AnimationInstruction

    instruction = AnimationInstruction(
        shot_id="s1", source_image_path=str(tmp_path / "x.png"), prompt="test",
        duration_seconds=2.0, camera_motion="static", subject_motion="test", seed=42,
    )
    assert instruction.seed == 42


def test_comfyui_image_provider_seed_override_produces_fixed_seed():
    from core.image_providers.comfyui import ComfyUIImageProvider

    provider = ComfyUIImageProvider(seed_override=lambda prompt: 777)
    assert provider._resolve_seed("any prompt at all") == 777
    assert provider._resolve_seed("a different prompt") == 777


def test_comfyui_image_provider_without_override_uses_derive_seed():
    from core.image_providers.comfyui import ComfyUIImageProvider

    provider = ComfyUIImageProvider()
    assert provider._resolve_seed("some prompt") == _derive_flux_seed("some prompt")


# --- 17/18. Dry-run: zero provider calls, shows seed policy -----------------


def test_dry_run_makes_zero_provider_calls_with_seed_policy_set():
    result = subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--dry-run", "--visual-seed-policy", "shared"],
        capture_output=True, text=True, check=True,
    )
    assert "rendering via image_provider" not in result.stdout.lower()  # the CLI's own generation-start message
    assert "dry run complete" in result.stdout.lower()


def test_dry_run_shows_resolved_seed_policy_and_base_seed():
    result = subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--dry-run", "--visual-seed-policy", "shared"],
        capture_output=True, text=True, check=True,
    )
    assert "visual seed policy:  shared" in result.stdout
    assert "fight base visual seed:" in result.stdout
    assert "flux seed:" in result.stdout
    assert "wan seed:" in result.stdout


def test_dry_run_default_policy_shows_none_base_seed():
    result = subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    assert "visual seed policy:  default" in result.stdout
    assert "fight base visual seed: None" in result.stdout


def test_shared_policy_dry_run_shows_identical_flux_seeds_across_shots():
    result = subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--dry-run", "--visual-seed-policy", "shared"],
        capture_output=True, text=True, check=True,
    )
    import re

    flux_seeds = re.findall(r"flux seed: (\d+)", result.stdout)
    assert len(flux_seeds) == 3
    assert len(set(flux_seeds)) == 1


# --- 19. Manifest records continuity/seed evidence --------------------------


def test_manifest_records_canonical_visuals_and_seeds():
    """Sprint 4 Prompt 12.1 note: this test previously used
    --visual-seed-policy shared against the default mock provider —
    which doesn't actually respect seed_override at all, so the old
    assertion that all manifest seeds matched was passing only because
    the manifest at the time recorded the *planned* value directly
    (plan.resolved_flux_seeds), never actually checking what the
    provider used — precisely the Issue 4 bug this same task fixed.
    Switched to the default policy here, which the mock-backed
    subprocess CLI genuinely supports correctly; the shared/derived
    policies' own behavior, and the planned-vs-actual evidence check
    itself, are covered properly by tests/test_prompt12_1_seed_evidence.py
    using a provider that actually respects seed_override."""
    import json
    import os
    import shutil

    manifest_path = "test_manifest_prompt12.json"
    subprocess.run(
        [sys.executable, RENDER_MULTI_SHOT, "--sample", "--manifest-path", manifest_path],
        capture_output=True, text=True, check=True,
    )
    manifest = json.loads(open(manifest_path).read())

    assert "canonical_visuals" in manifest
    assert "white_fighter" in manifest["canonical_visuals"]
    assert "black_fighter" in manifest["canonical_visuals"]
    assert "arena" in manifest["canonical_visuals"]
    assert manifest["visual_seed_policy"] == "default"

    for shot_entry in manifest["shots"]:
        assert "planned_flux_seed" in shot_entry
        assert "actual_flux_seed" in shot_entry
        assert shot_entry["planned_flux_seed"] == shot_entry["actual_flux_seed"]  # must agree under default too
        assert "wan_seed" in shot_entry
        assert "image_prompt" in shot_entry

    os.remove(manifest_path)
    shutil.rmtree("generated_animations", ignore_errors=True)
    shutil.rmtree("generated_images", ignore_errors=True)
    shutil.rmtree("storage", ignore_errors=True)


# --- 20. Existing 3-shot generation-count tests still pass ------------------


def test_six_job_expectation_unaffected_by_prompt12_changes(tmp_path):
    from tests.test_multi_shot_acceptance import _runner
    from tests.test_single_shot_acceptance import _preferences

    runner = _runner(tmp_path)
    plan = asyncio.run(runner.prepare(_sample_pgn(), _preferences(), max_animation_seconds=2.0))
    assert plan.expected_comfyui_job_count == 6
    assert len(plan.resolved_flux_seeds) == 3
    assert len(plan.resolved_wan_seeds) == 3


# --- 21/22. Generic defaults remain unchanged --------------------------------


def test_generic_image_provider_generate_image_signature_unchanged():
    import inspect

    from core.image_router import ImageProvider

    sig = inspect.signature(ImageProvider.generate_image)
    assert list(sig.parameters.keys()) == ["self", "prompt", "width", "height"]


def test_generic_animation_instruction_defaults_unchanged():
    from core.animation_router import AnimationInstruction, AnimationType

    instruction = AnimationInstruction(
        shot_id="s1", source_image_path="/tmp/x.png", prompt="test",
        duration_seconds=2.0, camera_motion="static", subject_motion="test",
    )
    assert instruction.animation_type == AnimationType.IMAGE_TO_VIDEO
    assert (instruction.width, instruction.height) == (1024, 1024)
    assert instruction.seed is None


# --- 23/24. No ordinary test contacts ComfyUI; live tests remain gated -----


def test_no_visual_continuity_code_makes_network_calls_by_default():
    """visual_continuity.py's own functions are pure/local — confirmed
    by inspection: no httpx, no network imports at all."""
    import products.chess2fight.rendering.visual_continuity as module

    source_path = module.__file__
    with open(source_path) as f:
        source = f.read()
    assert "httpx" not in source
    assert "requests" not in source
