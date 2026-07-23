"""Unit tests for metadata_normalizer — independently testable with
hand-built headers, no PGN parsing or move-walking required."""

import chess.pgn

from products.chess2fight.metadata_normalizer import (
    guess_opening,
    normalize_game_metadata,
)


def _headers(**tags: str) -> chess.pgn.Headers:
    h = chess.pgn.Headers()
    h.update(tags)
    return h


def test_normalizes_chesscom_style_headers():
    """Chess.com's example header set has no ECO/Opening tag."""
    headers = _headers(
        White="maia5",
        Black="n1000",
        Result="1-0",
        WhiteElo="1400",
        BlackElo="700",
        TimeControl="300+3",
        Termination="Normal",
    )
    metadata = normalize_game_metadata(headers, san_moves=["e4", "e5", "Qh5"])

    assert metadata.white_player == "maia5"
    assert metadata.black_player == "n1000"
    assert metadata.white_rating == 1400
    assert metadata.black_rating == 700
    assert metadata.time_control == "300+3"
    assert metadata.termination == "Normal"
    assert metadata.winner == "white"
    assert metadata.opening != ""  # falls back to the heuristic book


def test_normalizes_lichess_style_headers():
    """Lichess's example header set adds ECO/Opening/RatingDiff — the
    normalizer should use the explicit Opening tag rather than guessing."""
    headers = _headers(
        White="maia5",
        Black="n1000",
        Result="0-1",
        WhiteElo="1400",
        BlackElo="1600",
        ECO="A00",
        Opening="Van't Kruijs Opening",
        TimeControl="300+3",
        Termination="Normal",
        WhiteRatingDiff="-8",
        BlackRatingDiff="+8",
    )
    metadata = normalize_game_metadata(headers, san_moves=["e3"])

    assert metadata.opening == "Van't Kruijs Opening"
    assert metadata.winner == "black"


def test_missing_fields_use_safe_defaults():
    """A PGN with no optional tags at all must never raise, and must
    fall back to safe defaults — not python-chess's own '?' placeholder."""
    headers = _headers(Result="1-0")
    metadata = normalize_game_metadata(headers, san_moves=["e4"])

    assert metadata.white_player == "Unknown"
    assert metadata.black_player == "Unknown"
    assert metadata.white_rating is None
    assert metadata.black_rating is None
    assert metadata.time_control == "Unknown"
    assert metadata.termination == "Unknown"


def test_unparseable_rating_becomes_none_not_a_crash():
    headers = _headers(Result="1-0", WhiteElo="not-a-number")
    metadata = normalize_game_metadata(headers, san_moves=["e4"])
    assert metadata.white_rating is None


def test_draw_and_unknown_results():
    assert normalize_game_metadata(_headers(Result="1/2-1/2"), []).winner == "draw"
    assert normalize_game_metadata(_headers(Result="*"), []).winner == "unknown"


def test_guess_opening_prefers_explicit_header_over_heuristic():
    headers = _headers(Opening="Some Rare Line")
    assert guess_opening(headers, ["e4", "e5"]) == "Some Rare Line"


def test_guess_opening_falls_back_to_heuristic_book():
    headers = _headers()
    assert guess_opening(headers, ["e4", "e5", "Nf3", "Nc6", "Bb5"]) == "Ruy Lopez"


def test_guess_opening_falls_back_to_generic_label():
    headers = _headers()
    assert guess_opening(headers, ["a4", "a5"]) == "Unclassified Opening"
