"""Unit tests for the Prompt Generator.

Uses the real pipeline through the Scene Composer wherever possible —
the Prompt Generator's whole job is assembling text from data every
earlier stage already computed, so testing against real
ComposedTimeline output is what actually exercises it.
"""

import asyncio

from core.ai_router import TemplateProvider
from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.battle_mode_engine import generate_battle_mode_intelligence
from products.chess2fight.cinematic.prompt_generator import _ANGLE_PHRASE, _MOVEMENT_PHRASE, generate_prompts
from products.chess2fight.cinematic.scene_composer import compose_scene
from products.chess2fight.cinematic.timeline_engine import generate_shot_timeline
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.narrative_generator import NarrativeGenerator
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.schemas import BattleMode
from products.chess2fight.style_engine import generate_style_profile

SCHOLARS_MATE = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)
LEGALL_TRAP = (
    '[Result "1-0"]\n\n'
    "1. e4 e5 2. Nf3 d6 3. Bc4 Bg4 4. Nc3 g6 5. Nxe5 Bxd1 6. Bxf7+ Ke7 7. Nd5# 1-0"
)
DRAW_GAME = '[Result "1/2-1/2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1/2-1/2'

ALL_STYLES = ("anime", "fantasy", "modern_warfare", "superhero", "scifi")


def _build(pgn: str, style: str, mode: BattleMode = BattleMode.DUEL):
    """Runs the real pipeline through the Scene Composer."""
    analysis = analyze_game(pgn)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    profile = generate_style_profile(battle, combat, style)
    battle_mode = generate_battle_mode_intelligence(mode, combat, battle)
    story = asyncio.run(
        NarrativeGenerator(TemplateProvider()).generate(analysis, combat, battle, profile, battle_mode)
    )
    timeline = generate_shot_timeline(battle, story)
    return compose_scene(timeline, battle, profile, battle_mode)


def _all_required_elements_present(shot) -> list[str]:
    """Returns a list of any required element missing from `shot`'s
    image_prompt — empty list means everything is present."""
    p = shot.image_prompt.lower()
    checks = {
        "white appearance/clothing/weapon": (
            shot.scene.white_fighter.hair.lower() in p
            and shot.scene.white_fighter.clothing.lower() in p
            and shot.scene.white_fighter.weapon.lower() in p
        ),
        "black appearance/clothing/weapon": (
            shot.scene.black_fighter.hair.lower() in p
            and shot.scene.black_fighter.clothing.lower() in p
            and shot.scene.black_fighter.weapon.lower() in p
        ),
        "action": shot.description.lower()[:20] in p,
        "camera angle": _ANGLE_PHRASE[shot.camera_angle].lower() in p,
        "camera movement": _MOVEMENT_PHRASE[shot.camera_motion].lower() in p,
        "mood": shot.mood.lower() in p,
        "art style": shot.scene.cinematic_art_style.lower() in p,
        "quality modifiers": "8k" in p and "masterpiece" in p,
    }
    return [name for name, present in checks.items() if not present]


# --- Every required element present, every shot -----------------------------


def test_every_required_element_present_in_every_shot():
    composed = _build(LEGALL_TRAP, "anime")
    prompted = generate_prompts(composed)
    for shot in prompted.shots:
        missing = _all_required_elements_present(shot)
        assert not missing, f"Shot {shot.sequence_order} missing: {missing}"


def test_every_required_element_present_across_all_five_styles():
    for style in ALL_STYLES:
        composed = _build(SCHOLARS_MATE, style)
        prompted = generate_prompts(composed)
        for shot in prompted.shots:
            missing = _all_required_elements_present(shot)
            assert not missing, f"{style} shot {shot.sequence_order} missing: {missing}"


def test_both_fighters_clothing_and_weapon_present_regardless_of_focus():
    """Regression test: clothing/weapon must be present for BOTH
    fighters in every shot, not just whichever fighter the shot
    happens to focus on — an earlier draft only included full detail
    for the "prominent" fighter and dropped clothing/weapon for the
    other one entirely."""
    composed = _build(LEGALL_TRAP, "anime")
    prompted = generate_prompts(composed)
    for shot in prompted.shots:
        p = shot.image_prompt.lower()
        assert shot.scene.white_fighter.clothing.lower() in p
        assert shot.scene.black_fighter.clothing.lower() in p
        assert shot.scene.white_fighter.weapon.lower() in p
        assert shot.scene.black_fighter.weapon.lower() in p


# --- The "Anime cinematic style" requirement, generalized to genre ----------


def test_non_anime_styles_never_contain_the_word_anime():
    for style in ("fantasy", "modern_warfare", "superhero", "scifi"):
        composed = _build(SCHOLARS_MATE, style)
        prompted = generate_prompts(composed)
        for shot in prompted.shots:
            assert "anime" not in shot.image_prompt.lower()


def test_anime_style_prompts_do_contain_anime_style_language():
    composed = _build(SCHOLARS_MATE, "anime")
    prompted = generate_prompts(composed)
    assert all("anime" in shot.image_prompt.lower() for shot in prompted.shots)


def test_cinematic_art_style_from_scene_appears_verbatim_in_every_prompt():
    composed = _build(SCHOLARS_MATE, "scifi")
    prompted = generate_prompts(composed)
    for shot in prompted.shots:
        assert shot.scene.cinematic_art_style in shot.image_prompt


# --- No duplication (regression test for a real bug) -------------------------


def test_environment_name_never_duplicated_within_a_single_prompt():
    """Regression test: the establishing shot's own description
    ("Setting the scene: ..., in {environment}.") already restates the
    environment verbatim, and an earlier draft's environment clause
    added it a second time, producing "a fateful final arena (a
    fateful final arena)"."""
    composed = _build(LEGALL_TRAP, "anime")
    prompted = generate_prompts(composed)
    for shot in prompted.shots:
        assert shot.image_prompt.count(shot.environment) == 1, (
            f"Shot {shot.sequence_order} duplicates its environment text"
        )


# --- Determinism ---------------------------------------------------------------


def test_deterministic_same_input_same_output():
    composed = _build(SCHOLARS_MATE, "anime")
    results = [generate_prompts(composed) for _ in range(10)]
    assert all(r == results[0] for r in results)


def test_different_styles_produce_different_prompts_for_the_same_facts():
    prompts_by_style = {}
    for style in ALL_STYLES:
        composed = _build(SCHOLARS_MATE, style)
        prompted = generate_prompts(composed)
        prompts_by_style[style] = prompted.shots[0].image_prompt
    assert len(set(prompts_by_style.values())) == 5


# --- Structural correctness: nothing upstream is altered --------------------


def test_shot_count_and_total_duration_match_the_composed_timeline():
    composed = _build(LEGALL_TRAP, "fantasy")
    prompted = generate_prompts(composed)
    assert prompted.shot_count == composed.shot_count
    assert prompted.total_duration_seconds == composed.total_duration_seconds


def test_scene_continuity_carried_through_unchanged():
    composed = _build(SCHOLARS_MATE, "anime")
    prompted = generate_prompts(composed)
    assert prompted.scene_continuity == composed.scene_continuity


def test_every_enriched_shot_field_preserved_exactly():
    composed = _build(SCHOLARS_MATE, "anime")
    prompted = generate_prompts(composed)
    for original, prompted_shot in zip(composed.shots, prompted.shots):
        assert original.model_dump() == prompted_shot.model_dump(exclude={"image_prompt"})


# --- No image/API calls (structural guarantee, not just an assertion) -------


def test_output_is_plain_strings_only_no_binary_or_url_content():
    composed = _build(SCHOLARS_MATE, "anime")
    prompted = generate_prompts(composed)
    for shot in prompted.shots:
        assert isinstance(shot.image_prompt, str)
        assert "http://" not in shot.image_prompt
        assert "https://" not in shot.image_prompt


# --- Edge cases ------------------------------------------------------------------


def test_draw_game_with_minimal_tactics_still_generates_valid_prompts():
    composed = _build(DRAW_GAME, "fantasy")
    prompted = generate_prompts(composed)
    assert prompted.shot_count == composed.shot_count
    for shot in prompted.shots:
        assert shot.image_prompt
        assert not _all_required_elements_present(shot)


def test_army_mode_prompts_reference_army_environment():
    composed = _build(SCHOLARS_MATE, "anime", BattleMode.ARMY)
    prompted = generate_prompts(composed)
    for shot in prompted.shots:
        assert shot.scene.arena.layout in shot.image_prompt
