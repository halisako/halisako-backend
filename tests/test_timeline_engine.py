"""Unit tests for the Cinematic Timeline Engine.

Uses the real pipeline (pgn_analyzer -> combat_mapper -> battle_director
-> style_engine -> battle_mode_engine -> narrative_generator) to build
genuine BattleIntelligence/FightStory inputs wherever possible, since
the Timeline Engine's parsing is specifically coupled to the exact
screenplay format narrative_generator.py produces — a hand-written
fake FightStory.prompt would only prove the parser handles a shape it
was never actually asked to handle. A few tests below build FightStory
by hand anyway, specifically to exercise parser edge cases (missing
sections, no real move to extract) in isolation.
"""

import asyncio

from core.ai_router import TemplateProvider
from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.battle_mode_engine import generate_battle_mode_intelligence
from products.chess2fight.cinematic.schemas import ShotFocus, ShotType
from products.chess2fight.cinematic.timeline_engine import generate_shot_timeline
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.narrative_generator import NarrativeGenerator
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.schemas import BattleArc, BattleIntelligence, BattleMode, FighterPersonality, FightStory, PersonalityProfile
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
    """Runs the real pipeline end to end, returning (battle, fight_story)."""
    analysis = analyze_game(pgn)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    profile = generate_style_profile(battle, combat, style)
    battle_mode = generate_battle_mode_intelligence(mode, combat, battle)
    story = asyncio.run(
        NarrativeGenerator(TemplateProvider()).generate(analysis, combat, battle, profile, battle_mode)
    )
    return battle, story


def _dummy_battle(arc=BattleArc.FINAL_DUEL):
    from products.chess2fight.schemas import CombatStyle

    profile = PersonalityProfile(label="The Contender", rationale="test")
    return BattleIntelligence(battle_arc=arc, combat_style=CombatStyle.BALANCED, fighter_personality=FighterPersonality(white=profile, black=profile))


# --- Basic structure --------------------------------------------------------


def test_timeline_starts_with_establishing_and_ends_with_aftermath():
    battle, story = _build(SCHOLARS_MATE, "anime")
    timeline = generate_shot_timeline(battle, story)
    assert timeline.shots[0].shot_type == ShotType.ESTABLISHING
    assert timeline.shots[-1].shot_type == ShotType.AFTERMATH


def test_sequence_order_is_contiguous_from_one():
    battle, story = _build(SCHOLARS_MATE, "anime")
    timeline = generate_shot_timeline(battle, story)
    assert [s.sequence_order for s in timeline.shots] == list(range(1, timeline.shot_count + 1))


def test_shot_count_matches_shots_length():
    battle, story = _build(LEGALL_TRAP, "fantasy")
    timeline = generate_shot_timeline(battle, story)
    assert timeline.shot_count == len(timeline.shots)


def test_total_duration_equals_sum_of_shot_durations():
    battle, story = _build(LEGALL_TRAP, "scifi")
    timeline = generate_shot_timeline(battle, story)
    assert timeline.total_duration_seconds == round(sum(s.duration_seconds for s in timeline.shots), 2)


def test_every_shot_has_positive_duration():
    battle, story = _build(SCHOLARS_MATE, "anime")
    timeline = generate_shot_timeline(battle, story)
    assert all(s.duration_seconds > 0 for s in timeline.shots)


def test_every_shot_has_non_empty_environment_lighting_mood_description():
    battle, story = _build(SCHOLARS_MATE, "anime")
    timeline = generate_shot_timeline(battle, story)
    for shot in timeline.shots:
        assert shot.environment
        assert shot.lighting
        assert shot.mood
        assert shot.description


# --- Determinism -------------------------------------------------------------


def test_deterministic_same_inputs_same_output():
    battle, story = _build(LEGALL_TRAP, "anime")
    timelines = [generate_shot_timeline(battle, story) for _ in range(10)]
    assert all(t == timelines[0] for t in timelines)


def test_different_styles_can_produce_different_shot_content():
    """Style affects mood/environment (via the parsed screenplay), even
    though the underlying chess facts and shot structure don't."""
    timelines = {style: generate_shot_timeline(*_build(SCHOLARS_MATE, style)) for style in ALL_STYLES}
    environments = {t.shots[0].environment for t in timelines.values()}
    assert len(environments) == 5
    # Facts stay facts: same number of shots dramatizing the same moves.
    move_sets = {tuple(s.source_moves[0] for s in t.shots if s.source_moves) for t in timelines.values()}
    assert len(move_sets) == 1


# --- Chronological ordering (regression test for a real bug) ----------------


def test_middle_shots_are_chronologically_ordered_by_move_number():
    """Regression test: best_move and turning_point are not assumed to
    occur in a fixed order relative to each other — whichever happened
    earlier in the game must appear earlier in the timeline."""
    import re

    battle, story = _build(LEGALL_TRAP, "anime")
    timeline = generate_shot_timeline(battle, story)
    middle = [s for s in timeline.shots if s.shot_type not in (ShotType.ESTABLISHING, ShotType.AFTERMATH)]
    move_numbers = [int(re.match(r"(\d+)", s.source_moves[0]).group(1)) for s in middle if s.source_moves]
    assert move_numbers == sorted(move_numbers)


def test_climax_move_matches_best_move_and_turning_point_matches_turning_point():
    battle, story = _build(LEGALL_TRAP, "anime")
    timeline = generate_shot_timeline(battle, story)
    climax_shots = [s for s in timeline.shots if s.shot_type == ShotType.CLIMAX]
    turning_point_shots = [s for s in timeline.shots if s.shot_type == ShotType.TURNING_POINT]
    assert len(climax_shots) == 1
    assert len(turning_point_shots) == 1
    assert climax_shots[0].source_moves[0] in story.best_move
    assert turning_point_shots[0].source_moves[0] in story.turning_point


# --- Draw / no-standout-tactic handling (regression test for a real bug) ----


def test_draw_with_no_standout_tactic_does_not_fabricate_move_references():
    """Regression test: narrative_generator's fallback sentences ("No
    clear turning point emerges — ...", "No standout tactic emerges —
    ...") also contain " — " as ordinary prose punctuation and must
    never be misread as a real move label."""
    battle, story = _build(DRAW_GAME, "fantasy")
    assert "No clear turning point" in story.turning_point
    assert "No standout tactic" in story.best_move

    timeline = generate_shot_timeline(battle, story)
    for shot in timeline.shots:
        for move in shot.source_moves:
            assert "No clear" not in move
            assert "No standout" not in move
    # No real turning point/climax to dramatize -> no such shots at all.
    assert not any(s.shot_type in (ShotType.TURNING_POINT, ShotType.CLIMAX) for s in timeline.shots)


def test_draw_game_still_produces_a_valid_timeline():
    battle, story = _build(DRAW_GAME, "fantasy")
    timeline = generate_shot_timeline(battle, story)
    assert timeline.shot_count >= 2  # at minimum, establishing + aftermath
    assert timeline.shots[0].shot_type == ShotType.ESTABLISHING
    assert timeline.shots[-1].shot_type == ShotType.AFTERMATH


# --- Focus derivation ---------------------------------------------------------


def test_bookend_shots_have_environment_and_both_focus():
    battle, story = _build(SCHOLARS_MATE, "anime")
    timeline = generate_shot_timeline(battle, story)
    assert timeline.shots[0].focus == ShotFocus.ENVIRONMENT
    assert timeline.shots[-1].focus == ShotFocus.BOTH


def test_move_derived_shots_have_white_or_black_focus_matching_move_label():
    battle, story = _build(LEGALL_TRAP, "anime")
    timeline = generate_shot_timeline(battle, story)
    for shot in timeline.shots:
        if shot.shot_type in (ShotType.ESTABLISHING, ShotType.AFTERMATH) or not shot.source_moves:
            continue
        label = shot.source_moves[0]
        if "..." in label:
            assert shot.focus == ShotFocus.BLACK
        else:
            assert shot.focus == ShotFocus.WHITE


# --- Camera assignment ---------------------------------------------------------


def test_establishing_shot_is_always_wide():
    battle, story = _build(SCHOLARS_MATE, "anime")
    timeline = generate_shot_timeline(battle, story)
    assert timeline.shots[0].camera_angle.value == "wide"


def test_climax_shot_is_always_extreme_close_up():
    battle, story = _build(LEGALL_TRAP, "anime")
    timeline = generate_shot_timeline(battle, story)
    climax = next(s for s in timeline.shots if s.shot_type == ShotType.CLIMAX)
    assert climax.camera_angle.value == "extreme_close_up"


# --- Battle mode integration ---------------------------------------------------


def test_army_mode_environment_differs_from_duel_mode():
    battle_d, story_d = _build(SCHOLARS_MATE, "anime", BattleMode.DUEL)
    battle_a, story_a = _build(SCHOLARS_MATE, "anime", BattleMode.ARMY)
    timeline_d = generate_shot_timeline(battle_d, story_d)
    timeline_a = generate_shot_timeline(battle_a, story_a)
    assert timeline_d.shots[0].environment != timeline_a.shots[0].environment


# --- Section-extraction fallbacks (hand-built FightStory) ---------------------


def test_missing_screenplay_sections_fall_back_gracefully():
    """If FightStory.prompt doesn't contain the expected section
    headers at all, the engine must not crash — it should fall back to
    generic defaults."""
    battle = _dummy_battle()
    story = FightStory(
        winner="White wins by checkmate", opening="Test Opening",
        fight_style="Test Style", best_move="No standout tactic emerges — quiet game.",
        turning_point="No clear turning point emerges — balanced throughout.",
        battle_summary="A quiet game.", prompt="STYLE\nanime\n\nSETTING\nTest Opening",
        estimated_length="20-30 sec",
    )
    timeline = generate_shot_timeline(battle, story)
    assert timeline.shot_count >= 2
    assert timeline.shots[0].environment == "an unspecified arena"
    assert timeline.shots[0].lighting == "even, neutral lighting"


def test_unparseable_estimated_length_falls_back_to_default_duration():
    battle = _dummy_battle()
    story = FightStory(
        winner="Draw", opening="Test Opening", fight_style="Test Style",
        best_move="No standout tactic emerges — quiet.", turning_point="No clear turning point emerges — quiet.",
        battle_summary="A quiet game.", prompt="STYLE\nanime", estimated_length="unknown",
    )
    timeline = generate_shot_timeline(battle, story)
    assert timeline.total_duration_seconds > 0
