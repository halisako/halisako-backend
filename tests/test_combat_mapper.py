"""Unit tests for combat_mapper — built entirely from hand-constructed
GameAnalysis fixtures, no PGN parsing needed, matching the "combat
mapper must be unit-testable" requirement directly."""

from products.chess2fight.combat_mapper import (
    build_combat_profile,
    generate_combat_intelligence,
    map_combat_events,
)
from products.chess2fight.schemas import CombatEventType, GameAnalysis, GameMetadata, MoveRecord


def _move(**kwargs) -> MoveRecord:
    defaults = dict(
        ply=1,
        move_number=1,
        move_label="1. e4",
        san="e4",
        color="white",
        piece_moved="pawn",
        from_square="e2",
        to_square="e4",
        is_capture=False,
        captured_piece=None,
        is_check=False,
        is_checkmate=False,
        is_castle=False,
    )
    defaults.update(kwargs)
    return MoveRecord(**defaults)


def _analysis(moves: list[MoveRecord], metadata: GameMetadata | None = None) -> GameAnalysis:
    return GameAnalysis(
        white_player="White",
        black_player="Black",
        opening="Test Opening",
        num_moves=max((m.move_number for m in moves), default=0),
        winner="white",
        is_checkmate=any(m.is_checkmate for m in moves),
        checkmate_move_number=next((m.move_number for m in moves if m.is_checkmate), None),
        moves=moves,
        metadata=metadata or GameMetadata(),
    )


def test_pawn_push_is_territorial_advance():
    events = map_combat_events(_analysis([_move()]))
    assert events[0].event_type == CombatEventType.TERRITORIAL_ADVANCE


def test_checkmate_move_is_finishing_strike_with_max_intensity():
    moves = [_move(is_checkmate=True, san="Qxf7#", move_label="4. Qxf7#", is_capture=True,
                    captured_piece="pawn", piece_moved="queen")]
    events = map_combat_events(_analysis(moves))
    assert events[0].event_type == CombatEventType.FINISHING_STRIKE
    assert events[0].intensity == 10


def test_capture_with_check_is_breakthrough_attack():
    """The brief's own example: Bxf7+ -> breakthrough_attack."""
    moves = [
        _move(piece_moved="bishop", is_capture=True, captured_piece="pawn", is_check=True,
              san="Bxf7+", move_label="6. Bxf7+")
    ]
    events = map_combat_events(_analysis(moves))
    assert events[0].event_type == CombatEventType.BREAKTHROUGH_ATTACK


def test_castling_is_defensive_repositioning():
    moves = [_move(piece_moved="king", is_castle=True, san="O-O", move_label="5. O-O")]
    events = map_combat_events(_analysis(moves))
    assert events[0].event_type == CombatEventType.DEFENSIVE_REPOSITIONING


def test_queen_move_is_power_deployment():
    moves = [_move(piece_moved="queen", san="Qh5", move_label="2. Qh5")]
    events = map_combat_events(_analysis(moves))
    assert events[0].event_type == CombatEventType.POWER_DEPLOYMENT


def test_knight_development_is_tactical_setup():
    moves = [_move(piece_moved="knight", from_square="b1", to_square="c3",
                    san="Nc3", move_label="2. Nc3")]
    events = map_combat_events(_analysis(moves))
    assert events[0].event_type == CombatEventType.TACTICAL_SETUP


def test_knight_move_after_move_ten_is_not_development():
    """Development only counts early — the same move type later in the
    game is just positioning."""
    moves = [_move(piece_moved="knight", from_square="b1", to_square="c3",
                    san="Nc3", move_label="15. Nc3", move_number=15)]
    events = map_combat_events(_analysis(moves))
    assert events[0].event_type == CombatEventType.STRATEGIC_POSITIONING


def test_sacrifice_detected_when_recaptured_for_less():
    """Queen moves to a square, gets captured by a pawn next ply ->
    net material loss -> calculated_sacrifice."""
    moves = [
        _move(piece_moved="queen", to_square="h7", san="Qxh7", move_label="10. Qxh7",
              is_capture=True, captured_piece="pawn"),
        _move(piece_moved="pawn", to_square="h7", color="black", is_capture=True,
              captured_piece="queen", san="Kxh7", move_label="10...Kxh7", move_number=10),
    ]
    events = map_combat_events(_analysis(moves))
    assert events[0].event_type == CombatEventType.CALCULATED_SACRIFICE


def test_even_trade_is_not_flagged_as_sacrifice():
    """Knight takes knight, recaptured by knight — a fair trade, not a
    sacrifice (net loss is 0, below the threshold)."""
    moves = [
        _move(piece_moved="knight", to_square="d5", san="Nxd5", move_label="8. Nxd5",
              is_capture=True, captured_piece="knight"),
        _move(piece_moved="knight", to_square="d5", color="black", is_capture=True,
              captured_piece="knight", san="Nxd5", move_label="8...Nxd5", move_number=8),
    ]
    events = map_combat_events(_analysis(moves))
    assert events[0].event_type != CombatEventType.CALCULATED_SACRIFICE


def test_profile_infers_fast_pace_from_blitz_time_control():
    metadata = GameMetadata(time_control="180+0")
    analysis = _analysis([_move()], metadata=metadata)
    profile = build_combat_profile(analysis, map_combat_events(analysis))
    assert profile.battle_pace == "fast"


def test_profile_infers_strategic_pace_from_classical_time_control():
    metadata = GameMetadata(time_control="1800+30")
    analysis = _analysis([_move()], metadata=metadata)
    profile = build_combat_profile(analysis, map_combat_events(analysis))
    assert profile.battle_pace == "strategic"


def test_profile_falls_back_to_tactical_density_when_time_control_unknown():
    metadata = GameMetadata(time_control="Unknown")
    analysis = _analysis([_move()], metadata=metadata)
    profile = build_combat_profile(analysis, map_combat_events(analysis))
    assert profile.battle_pace in ("fast", "strategic")  # never raises, always resolves


def test_profile_infers_veteran_vs_challenger_from_large_rating_gap():
    metadata = GameMetadata(white_rating=1400, black_rating=700)
    analysis = _analysis([_move()], metadata=metadata)
    profile = build_combat_profile(analysis, map_combat_events(analysis))
    assert profile.fighter_balance == "veteran vs challenger"


def test_profile_balance_is_unknown_without_ratings():
    metadata = GameMetadata(white_rating=None, black_rating=None)
    analysis = _analysis([_move()], metadata=metadata)
    profile = build_combat_profile(analysis, map_combat_events(analysis))
    assert profile.fighter_balance == "unknown"


def test_generate_combat_intelligence_never_raises_on_empty_metadata():
    """Full pipeline entry point, worst-case input (defaults everywhere)."""
    analysis = _analysis([_move()], metadata=GameMetadata())
    intelligence = generate_combat_intelligence(analysis)
    assert len(intelligence.events) == 1
    assert intelligence.profile is not None
