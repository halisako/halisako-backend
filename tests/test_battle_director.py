"""Unit tests for battle_director.

Includes the two PGN-driven test cases from the brief (Scholar's Mate
-> blitz_execution, a long real tactical game -> war_of_attrition) run
through the real analyze_game + combat_mapper + battle_director chain,
plus hand-built-fixture tests for the individual decision branches
(faster, and pin down *why* a classification happens, not just that it
does)."""

from products.chess2fight.battle_director import generate_battle_intelligence
from products.chess2fight.combat_mapper import generate_combat_intelligence
from products.chess2fight.pgn_analyzer import analyze_game
from products.chess2fight.schemas import (
    BattleArc,
    CombatEvent,
    CombatEventType,
    CombatIntelligence,
    CombatProfile,
    CombatStyle,
    GameAnalysis,
    GameMetadata,
    MoveRecord,
)

SCHOLARS_MATE = """[Event "Example"]
[White "Halisako"]
[Black "Guest"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"""

# The real 20-move game from earlier in this project (winded_wayz vs
# SARDA_SUII) — 12 captures across 20 moves, no early decisive drama by
# the eventual winner. Reused deliberately: it's a real, already-
# verified-legal game, not a synthetic one built to game the test.
LONG_TACTICAL_GAME = """[Event "winded_wayz vs. SARDA_SUII"]
[Site "Chess.com"]
[White "winded_wayz"]
[Black "SARDA_SUII"]
[Result "1-0"]
[WhiteElo "557"]
[BlackElo "557"]
[TimeControl "300"]
[Termination "winded_wayz won by checkmate"]

1. e4 e5 2. Nf3 Nf6 3. Nxe5 Nxe4 4. Qe2 f5 5. g4 Bd6 6. gxf5 Bxe5 7. Qxe4 d6
8. d4 O-O 9. dxe5 Bxf5 10. Qd5+ Kh8 11. exd6 cxd6 12. Qxb7 Qe8+ 13. Be2 Be4
14. Rg1 Bxb7 15. Nc3 Nc6 16. Be3 g6 17. O-O-O Nb4 18. Bc4 Rc8 19. Bd4+ Rf6
20. Bxf6# 1-0"""

MISSING_METADATA_PGN = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"


def _run(pgn: str):
    analysis = analyze_game(pgn)
    combat = generate_combat_intelligence(analysis)
    battle = generate_battle_intelligence(analysis, combat)
    return analysis, combat, battle


# --- Required test 1: Scholar's Mate -----------------------------------


def test_scholars_mate_is_blitz_execution():
    _, _, battle = _run(SCHOLARS_MATE)
    assert battle.battle_arc == BattleArc.BLITZ_EXECUTION


# --- Required test 2: long tactical game --------------------------------


def test_long_tactical_game_is_war_of_attrition():
    _, _, battle = _run(LONG_TACTICAL_GAME)
    assert battle.battle_arc == BattleArc.WAR_OF_ATTRITION


# --- Required test 3: missing metadata must not crash -------------------


def test_missing_metadata_does_not_crash():
    """No headers at all beyond a bare Result — every field falls back
    to a default; battle_director must still produce a complete,
    valid BattleIntelligence."""
    analysis, combat, battle = _run(MISSING_METADATA_PGN)
    assert analysis.metadata.white_player == "Unknown"
    assert battle.battle_arc is not None
    assert battle.combat_style is not None
    assert battle.fighter_personality.white.label
    assert battle.fighter_personality.black.label


# --- Fixture-driven tests for individual decision branches --------------


def _move(**kwargs) -> MoveRecord:
    defaults = dict(
        ply=1, move_number=1, move_label="1. e4", san="e4", color="white",
        piece_moved="pawn", from_square="e2", to_square="e4",
        is_capture=False, captured_piece=None, is_check=False,
        is_checkmate=False, is_castle=False,
    )
    defaults.update(kwargs)
    return MoveRecord(**defaults)


def _analysis(moves, winner="white", metadata=None) -> GameAnalysis:
    return GameAnalysis(
        white_player="White", black_player="Black", opening="Test Opening",
        num_moves=max((m.move_number for m in moves), default=0),
        winner=winner,
        is_checkmate=any(m.is_checkmate for m in moves),
        checkmate_move_number=next((m.move_number for m in moves if m.is_checkmate), None),
        moves=moves, metadata=metadata or GameMetadata(winner=winner),
    )


def _combat(events) -> CombatIntelligence:
    return CombatIntelligence(
        events=events,
        profile=CombatProfile(battle_pace="fast", fighter_balance="even",
                               ending_type="checkmate", winner="white"),
    )


def test_gambit_assault_when_winner_has_early_drama():
    moves = [_move(move_number=i) for i in range(1, 5)]
    events = [
        CombatEvent(event_type=CombatEventType.BREAKTHROUGH_ATTACK, intensity=7,
                    attacker="white", description="d", move_number=3, move_label="3. x"),
        CombatEvent(event_type=CombatEventType.FINISHING_STRIKE, intensity=10,
                    attacker="white", description="d", move_number=4, move_label="4. x"),
    ]
    battle = generate_battle_intelligence(_analysis(moves), _combat(events))
    assert battle.battle_arc == BattleArc.GAMBIT_ASSAULT


def test_tactical_ambush_when_drama_is_not_the_winners_and_game_is_medium_length():
    moves = [_move(move_number=i) for i in range(1, 13)]
    events = [
        CombatEvent(event_type=CombatEventType.BREAKTHROUGH_ATTACK, intensity=7,
                    attacker="black", description="d", move_number=9, move_label="9. x"),
    ]
    battle = generate_battle_intelligence(_analysis(moves), _combat(events))
    assert battle.battle_arc == BattleArc.TACTICAL_AMBUSH


def test_desperate_combat_style_when_winner_was_ever_behind():
    moves = [
        _move(move_number=1, is_capture=True, captured_piece="queen", color="black"),
    ]
    battle = generate_battle_intelligence(_analysis(moves, winner="white"), _combat([]))
    assert battle.combat_style == CombatStyle.DESPERATE


def test_fortress_personality_on_defensive_draw():
    moves = [_move(move_number=i) for i in range(1, 10)]
    events: list[CombatEvent] = []  # no attacking events attributed to either side
    battle = generate_battle_intelligence(_analysis(moves, winner="draw"), _combat(events))
    assert battle.fighter_personality.white.label == "The Fortress"
    assert battle.fighter_personality.black.label == "The Fortress"


def test_never_crashes_on_empty_moves():
    """Defensive edge case: a GameAnalysis with zero moves (shouldn't
    happen via the real pipeline — analyze_game rejects 0-move PGNs —
    but battle_director shouldn't assume it can't receive one)."""
    battle = generate_battle_intelligence(_analysis([]), _combat([]))
    assert battle.battle_arc is not None
