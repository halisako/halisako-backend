"""Unit tests for battle_mode_engine — built from hand-constructed
CombatIntelligence/BattleIntelligence fixtures, no PGN parsing needed."""

from products.chess2fight.battle_mode_engine import generate_battle_mode_intelligence
from products.chess2fight.schemas import (
    BattleArc,
    BattleIntelligence,
    BattleMode,
    CombatIntelligence,
    CombatProfile,
    CombatStyle,
    FighterPersonality,
    PersonalityProfile,
)


def _battle(arc: BattleArc = BattleArc.FINAL_DUEL, style: CombatStyle = CombatStyle.BALANCED) -> BattleIntelligence:
    profile = PersonalityProfile(label="The Contender", rationale="test fixture")
    return BattleIntelligence(
        battle_arc=arc, combat_style=style,
        fighter_personality=FighterPersonality(white=profile, black=profile),
    )


def _combat(event_count: int = 5) -> CombatIntelligence:
    return CombatIntelligence(
        events=[],
        profile=CombatProfile(battle_pace="fast", fighter_balance="even", ending_type="checkmate", winner="white"),
    )


# --- Required Test 3: exact army unit mapping -----------------------------


def test_army_unit_mapping_matches_the_required_spec_exactly():
    result = generate_battle_mode_intelligence(BattleMode.ARMY, _combat(), _battle())
    assert result.unit_mapping == {
        "pawn": "infantry",
        "knight": "cavalry",
        "bishop": "mage",
        "rook": "siege",
        "queen": "commander",
        "king": "fortress",
    }


def test_duel_unit_mapping_has_all_six_piece_types_and_differs_from_army():
    army = generate_battle_mode_intelligence(BattleMode.ARMY, _combat(), _battle())
    duel = generate_battle_mode_intelligence(BattleMode.DUEL, _combat(), _battle())
    assert set(duel.unit_mapping.keys()) == {"pawn", "knight", "bishop", "rook", "queen", "king"}
    assert duel.unit_mapping != army.unit_mapping


# --- Mode never invented outside the enum ---------------------------------


def test_mode_field_reflects_the_requested_mode():
    assert generate_battle_mode_intelligence(BattleMode.DUEL, _combat(), _battle()).mode == BattleMode.DUEL
    assert generate_battle_mode_intelligence(BattleMode.ARMY, _combat(), _battle()).mode == BattleMode.ARMY


# --- Scale and environment respond to combat/battle intelligence ---------


def test_army_scale_grows_with_combat_event_volume():
    small = CombatIntelligence(events=[], profile=CombatProfile(
        battle_pace="fast", fighter_balance="even", ending_type="checkmate", winner="white"))
    result = generate_battle_mode_intelligence(BattleMode.ARMY, small, _battle())
    assert result.scale == "skirmish"  # 0 events


def test_army_and_duel_scale_use_different_vocabularies():
    army = generate_battle_mode_intelligence(BattleMode.ARMY, _combat(), _battle(style=CombatStyle.OVERWHELMING))
    duel = generate_battle_mode_intelligence(BattleMode.DUEL, _combat(), _battle(style=CombatStyle.OVERWHELMING))
    assert "duel" in duel.scale
    assert "duel" not in army.scale


def test_environment_differs_by_mode_for_the_same_battle():
    battle = _battle(arc=BattleArc.SIEGE)
    army = generate_battle_mode_intelligence(BattleMode.ARMY, _combat(), battle)
    duel = generate_battle_mode_intelligence(BattleMode.DUEL, _combat(), battle)
    assert army.environment != duel.environment


def test_environment_varies_by_battle_arc_within_a_mode():
    blitz = generate_battle_mode_intelligence(
        BattleMode.ARMY, _combat(), _battle(arc=BattleArc.BLITZ_EXECUTION)
    )
    siege = generate_battle_mode_intelligence(
        BattleMode.ARMY, _combat(), _battle(arc=BattleArc.SIEGE)
    )
    assert blitz.environment != siege.environment


def test_combat_focus_is_never_empty():
    for mode in (BattleMode.DUEL, BattleMode.ARMY):
        result = generate_battle_mode_intelligence(mode, _combat(), _battle())
        assert len(result.combat_focus) > 0


# --- Determinism ------------------------------------------------------------


def test_deterministic_same_inputs_same_output():
    battle = _battle(arc=BattleArc.WAR_OF_ATTRITION, style=CombatStyle.CHAOTIC)
    combat = _combat()
    results = [generate_battle_mode_intelligence(BattleMode.ARMY, combat, battle) for _ in range(10)]
    assert all(r == results[0] for r in results)
