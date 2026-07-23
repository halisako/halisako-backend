"""Unit tests for narrative_generator (the Battle Screenplay Generator).

Covers the brief's required determinism tests at the function level
(faster and more precise than the API-level equivalents in
test_api_regression.py), plus per-field correctness checks.

v1.5: _story()'s default battle_mode is DUEL, matching every existing
assertion below (which were all written against duel-mode screenplay
section labels). Army-mode-specific behavior is covered separately in
test_battle_mode_engine.py and the dedicated army-mode tests at the
end of this file."""

import asyncio

from core.ai_router import TemplateProvider
from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.battle_mode_engine import generate_battle_mode_intelligence
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.narrative_generator import NarrativeGenerator
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.schemas import BattleMode
from products.chess2fight.style_engine import generate_style_profile

SCHOLARS_MATE = """[Event "Example"]
[White "Halisako"]
[Black "Guest"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"""

LEGALL_TRAP = """[Event "Example"]
[White "Halisako"]
[Black "Guest"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. Bc4 Bg4 4. Nc3 g6 5. Nxe5 Bxd1 6. Bxf7+ Ke7 7. Nd5# 1-0"""

DRAW_GAME = '[Result "1/2-1/2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1/2-1/2'

ALL_STYLES = ["anime", "fantasy", "modern_warfare", "superhero", "scifi"]


def _full_pipeline(pgn: str, style: str, mode: BattleMode = BattleMode.DUEL):
    analysis = analyze_game(pgn)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    profile = generate_style_profile(battle, combat, style)
    battle_mode = generate_battle_mode_intelligence(mode, combat, battle)
    return analysis, combat, battle, profile, battle_mode


def _story(pgn: str, style: str, mode: BattleMode = BattleMode.DUEL):
    analysis, combat, battle, profile, battle_mode = _full_pipeline(pgn, style, mode)
    generator = NarrativeGenerator(TemplateProvider())
    return asyncio.run(generator.generate(analysis, combat, battle, profile, battle_mode))


# --- Required: determinism -----------------------------------------------


def test_same_pgn_same_style_identical_narrative():
    stories = [_story(SCHOLARS_MATE, "anime") for _ in range(10)]
    assert all(s == stories[0] for s in stories)


def test_same_pgn_same_preferences_identical_narrative_including_mode():
    """v1.5 extension of the determinism requirement: PGN + preferences
    (style AND battle_mode) must always produce the same output."""
    stories = [_story(SCHOLARS_MATE, "scifi", BattleMode.ARMY) for _ in range(10)]
    assert all(s == stories[0] for s in stories)


def test_same_pgn_different_styles_significantly_different():
    stories = {style: _story(SCHOLARS_MATE, style) for style in ALL_STYLES}
    assert len({s.fight_style for s in stories.values()}) == 5
    assert len({s.battle_summary for s in stories.values()}) == 5
    assert len({s.prompt for s in stories.values()}) == 5
    # Facts stay facts regardless of style.
    assert len({s.winner for s in stories.values()}) == 1
    assert len({s.opening for s in stories.values()}) == 1


# --- winner / opening: untouched facts ------------------------------------


def test_winner_and_opening_unchanged_from_prior_revision():
    story = _story(SCHOLARS_MATE, "anime")
    assert story.winner == "White wins by checkmate"
    assert story.opening == "Italian Game (early queen sortie)"


def test_draw_game_does_not_crash_and_has_sensible_fields():
    story = _story(DRAW_GAME, "fantasy")
    assert story.winner == "Draw"
    assert "-" in story.estimated_length


# --- fight_style: personality/arc/style driven -----------------------------


def test_fight_style_anime_uses_winner_personality():
    story = _story(SCHOLARS_MATE, "anime")
    analysis = analyze_game(SCHOLARS_MATE)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    assert battle.fighter_personality.white.label in story.fight_style


def test_fight_style_modern_warfare_reads_as_a_doctrine_name_not_a_sentence():
    story = _story(SCHOLARS_MATE, "modern_warfare")
    assert len(story.fight_style.split()) <= 4  # short doctrine-style name, not a sentence


# --- best_move / turning_point: cite the real move, explain why ----------


def test_best_move_cites_the_actual_move_and_is_not_just_notation():
    story = _story(SCHOLARS_MATE, "anime")
    assert "4. Qxf7#" in story.best_move
    assert len(story.best_move) > len("4. Qxf7# captures the pawn")  # more than bare notation


def test_best_move_never_asserts_unsupported_intentional_sacrifice():
    """Required caution: a sacrifice-classified move should read as
    'appears to' / hedged, never as a bare factual claim of intent."""
    story = _story(LEGALL_TRAP, "fantasy")
    if "sacrifice" in story.best_move.lower() or "offer" in story.best_move.lower():
        assert "appear" in story.best_move.lower() or "forced" in story.best_move.lower()


def test_turning_point_cites_the_real_move_and_explains_why():
    story = _story(SCHOLARS_MATE, "scifi")
    assert "4. Qxf7#" in story.turning_point
    assert "shattered" in story.turning_point.lower() or "left" in story.turning_point.lower() \
        or "exposed" in story.turning_point.lower() or "escape" in story.turning_point.lower() \
        or "breached" in story.turning_point.lower() or "route" in story.turning_point.lower()


# --- battle_summary: incorporates all six required elements ---------------


def test_battle_summary_incorporates_required_elements():
    analysis, combat, battle, profile, battle_mode = _full_pipeline(SCHOLARS_MATE, "anime")
    generator = NarrativeGenerator(TemplateProvider())
    story = asyncio.run(generator.generate(analysis, combat, battle, profile, battle_mode))

    assert analysis.opening in story.battle_summary
    assert battle.battle_arc.value.replace("_", " ") in story.battle_summary
    assert battle.fighter_personality.white.label in story.battle_summary
    assert battle.fighter_personality.black.label in story.battle_summary
    assert "White" in story.battle_summary


# --- prompt: full screenplay structure (duel mode) ------------------------


DUEL_REQUIRED_SECTIONS = [
    "STYLE", "SETTING", "ENVIRONMENT", "LIGHTING", "CAMERA", "FIGHTERS",
    "White Fighter", "Black Fighter", "PERSONALITIES", "BATTLE SCALE", "VISUAL STYLE",
    "WEAPONS", "POWERS", "VISUAL EFFECTS", "COMBAT CHOREOGRAPHY", "ENDING", "FINAL SHOT",
]

ARMY_REQUIRED_SECTIONS = [
    "STYLE", "SETTING", "ENVIRONMENT", "LIGHTING", "CAMERA", "FORCES",
    "White Force", "Black Force", "PERSONALITIES", "BATTLE SCALE", "VISUAL STYLE",
    "WEAPONS", "POWERS", "VISUAL EFFECTS", "COMBAT CHOREOGRAPHY", "ENDING", "FINAL SHOT",
]


def test_prompt_contains_every_required_screenplay_section_duel_mode():
    story = _story(SCHOLARS_MATE, "superhero", BattleMode.DUEL)
    for section in DUEL_REQUIRED_SECTIONS:
        assert section in story.prompt, f"missing: {section}"


def test_prompt_contains_every_required_screenplay_section_army_mode():
    story = _story(SCHOLARS_MATE, "superhero", BattleMode.ARMY)
    for section in ARMY_REQUIRED_SECTIONS:
        assert section in story.prompt, f"missing: {section}"


def test_prompt_choreography_has_one_line_per_combat_event():
    analysis, combat, battle, profile, battle_mode = _full_pipeline(SCHOLARS_MATE, "anime")
    story = asyncio.run(NarrativeGenerator(TemplateProvider()).generate(analysis, combat, battle, profile, battle_mode))
    choreography_lines = [
        line for line in story.prompt.split("COMBAT CHOREOGRAPHY\n")[1].split("\n\n")[0].split("\n") if line
    ]
    assert len(choreography_lines) == len(combat.events)


def test_prompt_weapons_and_powers_match_style_profile():
    analysis, combat, battle, profile, battle_mode = _full_pipeline(SCHOLARS_MATE, "scifi")
    story = asyncio.run(NarrativeGenerator(TemplateProvider()).generate(analysis, combat, battle, profile, battle_mode))
    for weapon in profile.weapons:
        assert weapon in story.prompt
    for power in profile.powers:
        assert power in story.prompt


# --- v1.5: battle mode changes presentation, not chess facts --------------


def test_duel_vs_army_same_pgn_different_screenplay_same_facts():
    """Required Test 2 (from the battle-mode brief): same PGN, duel vs
    army preferences -> different screenplay/environment/combat
    interpretation, but identical winner/moves/chess analysis."""
    analysis_d, combat_d, battle_d, profile_d, mode_d = _full_pipeline(SCHOLARS_MATE, "anime", BattleMode.DUEL)
    analysis_a, combat_a, battle_a, profile_a, mode_a = _full_pipeline(SCHOLARS_MATE, "anime", BattleMode.ARMY)

    story_d = asyncio.run(NarrativeGenerator(TemplateProvider()).generate(analysis_d, combat_d, battle_d, profile_d, mode_d))
    story_a = asyncio.run(NarrativeGenerator(TemplateProvider()).generate(analysis_a, combat_a, battle_a, profile_a, mode_a))

    # Different presentation.
    assert story_d.prompt != story_a.prompt
    assert "FIGHTERS" in story_d.prompt and "FORCES" not in story_d.prompt
    assert "FORCES" in story_a.prompt and "FIGHTERS" not in story_a.prompt
    assert mode_d.environment != mode_a.environment

    # Identical chess facts.
    assert analysis_d.winner == analysis_a.winner
    assert analysis_d.num_moves == analysis_a.num_moves
    assert analysis_d.moves == analysis_a.moves
    assert story_d.winner == story_a.winner


def test_army_mode_choreography_uses_unit_vocabulary():
    story = _story(SCHOLARS_MATE, "anime", BattleMode.ARMY)
    # 2...Nc6 is a knight move -> should read as "cavalry" in army mode.
    assert "cavalry" in story.prompt.lower()


def test_duel_mode_choreography_uses_style_vocabulary_not_units():
    story = _story(SCHOLARS_MATE, "anime", BattleMode.DUEL)
    assert "cavalry" not in story.prompt.lower()
    assert "infantry" not in story.prompt.lower()


# --- estimated_length: arc-driven, not move-count-driven ------------------


def test_estimated_length_is_a_range_string():
    story = _story(SCHOLARS_MATE, "anime")
    assert "-" in story.estimated_length and "sec" in story.estimated_length


def test_blitz_execution_is_short_matching_the_briefs_example():
    story = _story(SCHOLARS_MATE, "anime")
    low = int(story.estimated_length.split("-")[0])
    high = int(story.estimated_length.split("-")[1].split()[0])
    assert low <= 15  # brief's own example: "10-12 seconds"
    assert high <= 20


def test_longer_more_tactical_game_estimates_longer_than_blitz_mate():
    scholars = _story(SCHOLARS_MATE, "anime")
    legall = _story(LEGALL_TRAP, "anime")
    scholars_high = int(scholars.estimated_length.split("-")[1].split()[0])
    legall_high = int(legall.estimated_length.split("-")[1].split()[0])
    assert legall_high >= scholars_high
