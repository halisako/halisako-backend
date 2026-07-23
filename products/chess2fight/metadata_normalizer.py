"""
Normalizes PGN header metadata into a single, source-agnostic shape —
this is the "Metadata Normalizer" stage in the pipeline:

    PGN -> Metadata Normalizer -> Analysis -> Combat Mapper -> Narrative

Chess.com, Lichess, and other platforms all emit different header sets
(Lichess adds ECO/Opening/WhiteRatingDiff/BlackRatingDiff; Chess.com
doesn't, in the examples this was built against). Nothing in this
module branches on *which* platform a PGN came from — it just reads
whichever of a fixed set of well-known header names happen to be
present and falls back to a safe default for anything that's missing.
That's what makes it source-agnostic: adding support for a PGN from
some other platform tomorrow requires no code change here, as long as
that platform uses the same standard PGN tag names (which nearly all
do — these are the standard "Seven Tag Roster" extensions, not
anything platform-specific).

This module is intentionally self-contained: given just `headers` and
the game's SAN move list, it produces a complete GameMetadata with zero
dependency on pgn_analyzer's move-walking state, which is what makes it
independently unit-testable (see tests/test_metadata_normalizer.py).
"""

from __future__ import annotations

import chess.pgn

from products.chess2fight.schemas import GameMetadata

# A small, hand-picked set of common openings matched against the first
# few moves in SAN. Not an ECO database — just enough to name the
# openings a casual game is likely to actually be. Only used when the
# PGN itself doesn't already carry an Opening/ECO header (Lichess
# usually does; Chess.com, in practice, often doesn't).
_OPENING_BOOK: list[tuple[list[str], str]] = [
    (["e4", "e5", "Nf3", "Nc6", "Bb5"], "Ruy Lopez"),
    (["e4", "e5", "Nf3", "Nc6", "Bc4"], "Italian Game"),
    (["e4", "e5", "Qh5"], "Italian Game (early queen sortie)"),
    (["e4", "e5", "Nf3", "d6"], "Philidor Defense"),
    (["e4", "c5"], "Sicilian Defense"),
    (["e4", "e6"], "French Defense"),
    (["e4", "c6"], "Caro-Kann Defense"),
    (["e4", "d5"], "Scandinavian Defense"),
    (["e4", "Nf6"], "Alekhine's Defense"),
    (["d4", "d5", "c4"], "Queen's Gambit"),
    (["d4", "Nf6", "c4", "g6"], "King's Indian Defense"),
    (["d4", "Nf6", "c4", "e6"], "Nimzo/Queen's Indian setup"),
    (["d4", "f5"], "Dutch Defense"),
    (["c4"], "English Opening"),
    (["Nf3"], "Reti Opening"),
    (["e4"], "Open Game (1.e4)"),
    (["d4"], "Closed Game (1.d4)"),
]


def guess_opening(headers: chess.pgn.Headers, san_moves: list[str]) -> str:
    """Prefer whatever the source platform already tells us (Lichess
    PGNs usually carry Opening/ECO); fall back to the small heuristic
    book; fall back to a generic label if nothing matches. Never raises."""
    for tag in ("Opening", "ECO"):
        value = headers.get(tag)
        if value:
            return value

    for pattern, name in _OPENING_BOOK:
        if san_moves[: len(pattern)] == pattern:
            return name

    return "Unclassified Opening"


def _parse_rating(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except (ValueError, AttributeError):
        return None


def _clean_name(raw: str | None) -> str | None:
    """python-chess auto-fills the standard Seven Tag Roster (including
    White/Black) with the PGN spec's own placeholder value "?" when a
    tag is absent from the source text — unlike every other tag, which
    it simply omits. A plain `or "Unknown"` fallback only catches
    missing/empty values, not this placeholder, so it's normalized
    explicitly here."""
    if raw is None or raw.strip() in ("", "?"):
        return None
    return raw


def _extract_winner(result: str) -> str:
    if result == "1-0":
        return "white"
    if result == "0-1":
        return "black"
    if result == "1/2-1/2":
        return "draw"
    return "unknown"


def normalize_game_metadata(
    headers: chess.pgn.Headers,
    san_moves: list[str],
) -> GameMetadata:
    """Builds a GameMetadata from PGN headers, regardless of which
    platform produced them. Every field falls back to a safe default —
    this function never raises, even given a headers object with
    almost nothing in it (e.g. a hand-typed PGN with no tags at all)."""
    return GameMetadata(
        white_player=_clean_name(headers.get("White")) or "Unknown",
        black_player=_clean_name(headers.get("Black")) or "Unknown",
        white_rating=_parse_rating(headers.get("WhiteElo")),
        black_rating=_parse_rating(headers.get("BlackElo")),
        opening=guess_opening(headers, san_moves),
        time_control=headers.get("TimeControl") or "Unknown",
        termination=headers.get("Termination") or "Unknown",
        winner=_extract_winner(headers.get("Result") or "*"),
    )
