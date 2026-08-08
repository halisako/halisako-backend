"""Unit tests for the Scene Composer.

Uses the real pipeline (through the Timeline Engine) wherever possible,
since the Scene Composer's job is specifically about staying
consistent with what the rest of the pipeline already established
(e.g. BattleModeIntelligence's environment) — a hand-built fixture
would not exercise that consistency at all.
"""

import asyncio

from core.ai_router import TemplateProvider
from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.battle_mode_engine import generate_battle_mode_intelligence
from products.chess2fight.cinematic.scene_composer import compose_scene
from products.chess2fight.cinematic.timeline_engine import generate_shot_timeline
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.narrative_generator import NarrativeGenerator
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.schemas import BattleMode, StyleId, StyleProfile
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
    """Runs the real pipeline through the Timeline Engine, returning
    (battle, style_profile, battle_mode_intelligence, timeline)."""
    analysis = analyze_game(pgn)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    profile = generate_style_profile(battle, combat, style)
    battle_mode = generate_battle_mode_intelligence(mode, combat, battle)
    story = asyncio.run(
        NarrativeGenerator(TemplateProvider()).generate(analysis, combat, battle, profile, battle_mode)
    )
    timeline = generate_shot_timeline(battle, story)
    return battle, profile, battle_mode, timeline


# --- Continuity guarantee: the core purpose of this module ------------------


def test_every_shot_shares_the_identical_scene_continuity_object():
    battle, profile, mode, timeline = _build(LEGALL_TRAP, "anime")
    composed = compose_scene(timeline, battle, profile, mode)
    assert all(shot.scene == composed.scene_continuity for shot in composed.shots)


def test_shot_count_and_total_duration_match_the_original_timeline_exactly():
    battle, profile, mode, timeline = _build(LEGALL_TRAP, "fantasy")
    composed = compose_scene(timeline, battle, profile, mode)
    assert composed.shot_count == timeline.shot_count
    assert composed.total_duration_seconds == timeline.total_duration_seconds


def test_original_shot_fields_are_preserved_exactly():
    """Enriching must not alter anything the Timeline Engine already
    decided — only add to it."""
    battle, profile, mode, timeline = _build(SCHOLARS_MATE, "anime")
    composed = compose_scene(timeline, battle, profile, mode)
    for original, enriched in zip(timeline.shots, composed.shots):
        assert original.model_dump() == enriched.model_dump(exclude={"scene"})


# --- Determinism ---------------------------------------------------------------


def test_deterministic_same_inputs_same_output():
    battle, profile, mode, timeline = _build(SCHOLARS_MATE, "anime")
    results = [compose_scene(timeline, battle, profile, mode) for _ in range(10)]
    assert all(r == results[0] for r in results)


def test_different_styles_produce_different_appearance_and_palette():
    compositions = {}
    for style in ALL_STYLES:
        battle, profile, mode, timeline = _build(SCHOLARS_MATE, style)
        compositions[style] = compose_scene(timeline, battle, profile, mode)

    art_styles = {c.scene_continuity.cinematic_art_style for c in compositions.values()}
    palettes = {tuple(c.scene_continuity.color_palette) for c in compositions.values()}
    assert len(art_styles) == 5
    assert len(palettes) == 5


# --- Arena consistency with the rest of the response (regression test) ------


def test_arena_layout_matches_battle_mode_environment_exactly():
    """Regression test for a real consistency gap: arena.layout must
    match BattleModeIntelligence.environment exactly, not a separately
    derived description that could describe a dueling ground while the
    screenplay elsewhere says FORCES and a battlefield."""
    battle, profile, mode, timeline = _build(SCHOLARS_MATE, "anime", BattleMode.DUEL)
    composed = compose_scene(timeline, battle, profile, mode)
    assert composed.scene_continuity.arena.layout == mode.environment


def test_duel_and_army_mode_produce_different_arena_layouts_for_the_same_battle():
    battle_d, profile_d, mode_d, timeline_d = _build(SCHOLARS_MATE, "anime", BattleMode.DUEL)
    battle_a, profile_a, mode_a, timeline_a = _build(SCHOLARS_MATE, "anime", BattleMode.ARMY)
    composed_d = compose_scene(timeline_d, battle_d, profile_d, mode_d)
    composed_a = compose_scene(timeline_a, battle_a, profile_a, mode_a)
    assert composed_d.scene_continuity.arena.layout != composed_a.scene_continuity.arena.layout


# --- Fighter appearance -----------------------------------------------------------


def test_white_and_black_fighters_get_different_appearances():
    battle, profile, mode, timeline = _build(LEGALL_TRAP, "anime")
    composed = compose_scene(timeline, battle, profile, mode)
    white, black = composed.scene_continuity.white_fighter, composed.scene_continuity.black_fighter
    assert white != black


def test_fighter_weapon_comes_from_style_profile_weapons():
    battle, profile, mode, timeline = _build(LEGALL_TRAP, "anime")
    composed = compose_scene(timeline, battle, profile, mode)
    assert composed.scene_continuity.white_fighter.weapon in profile.weapons
    assert composed.scene_continuity.black_fighter.weapon in profile.weapons


def test_single_weapon_style_profile_does_not_crash():
    battle, profile, mode, timeline = _build(SCHOLARS_MATE, "anime")

    single_weapon_profile = StyleProfile(
        style=StyleId.ANIME,
        weapons=["katana"],
        environment="dojo",
        finisher="Dragon Slash",
    )

    composed = compose_scene(timeline, battle, single_weapon_profile, mode)
    assert composed.scene_continuity.white_fighter.weapon == "katana"
    assert composed.scene_continuity.black_fighter.weapon == "katana"


def test_no_weapons_at_all_falls_back_gracefully():
    battle, profile, mode, timeline = _build(SCHOLARS_MATE, "anime")

    no_weapon_profile = StyleProfile(
        style=StyleId.ANIME,
        weapons=[],
        environment="dojo",
        finisher="None",
    )

    composed = compose_scene(timeline, battle, no_weapon_profile, mode)
    assert composed.scene_continuity.white_fighter.weapon
    assert composed.scene_continuity.black_fighter.weapon

# --- Edge cases ------------------------------------------------------------------


def test_draw_game_with_minimal_tactics_still_composes_a_valid_scene():
    battle, profile, mode, timeline = _build(DRAW_GAME, "fantasy")
    composed = compose_scene(timeline, battle, profile, mode)
    assert composed.shot_count == timeline.shot_count
    assert all(shot.scene == composed.scene_continuity for shot in composed.shots)


def test_every_continuity_field_is_non_empty():
    battle, profile, mode, timeline = _build(SCHOLARS_MATE, "superhero")
    composed = compose_scene(timeline, battle, profile, mode)
    continuity = composed.scene_continuity
    for fighter in (continuity.white_fighter, continuity.black_fighter):
        assert fighter.hair and fighter.facial_features and fighter.clothing and fighter.armor and fighter.weapon
    assert continuity.arena.layout and continuity.arena.weather and continuity.arena.time_of_day
    assert continuity.lighting_continuity
    assert continuity.cinematic_art_style
    assert continuity.color_palette

