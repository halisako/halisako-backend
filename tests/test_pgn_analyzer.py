"""Unit tests for analyze_game — including an explicit regression check
that every field present before this revision is still present and
correctly computed."""

import pytest

from core.exceptions import InvalidPGNError
from products.chess2fight.pgn_analyzer import analyze_game

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


def test_existing_fields_unchanged_scholars_mate():
    """Every field that existed before v1.1 must still compute the same
    values — this is the regression check for 'do not break the
    existing contract.'"""
    a = analyze_game(SCHOLARS_MATE)
    assert a.white_player == "Halisako"
    assert a.black_player == "Guest"
    assert a.opening == "Italian Game (early queen sortie)"
    assert a.num_moves == 4
    assert a.winner == "white"
    assert a.is_checkmate is True
    assert a.checkmate_move_number == 4
    assert len(a.captures) == 1
    assert a.captures[0].move_label == "4. Qxf7#"
    assert len(a.tactical_moments) == 1
    assert len(a.turning_points) == 1


def test_new_fields_present_and_populated():
    """New in v1.1: moves + metadata, purely additive."""
    a = analyze_game(SCHOLARS_MATE)
    assert len(a.moves) == 7  # 7 plies in this game
    assert a.moves[-1].is_checkmate is True
    assert a.metadata.winner == "white"
    assert a.metadata.opening == a.opening


def test_legall_trap_move_numbering_and_metadata():
    a = analyze_game(LEGALL_TRAP)
    assert a.num_moves == 7
    assert len(a.moves) == 13  # 13 plies
    assert a.moves[9].san == "Bxd1"  # ply 10 (index 9): 5...Bxd1
    assert a.moves[9].color == "black"
    assert a.moves[9].captured_piece == "queen"


@pytest.mark.parametrize("bad_input", ["", "   ", "this is not chess", "[Result \"*\"]\n\n*"])
def test_invalid_pgn_raises(bad_input):
    with pytest.raises(InvalidPGNError):
        analyze_game(bad_input)


def test_draw_game_does_not_crash_and_has_sensible_defaults():
    draw_pgn = '[Result "1/2-1/2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1/2-1/2'
    a = analyze_game(draw_pgn)
    assert a.winner == "draw"
    assert a.is_checkmate is False
    assert a.metadata.winner == "draw"
