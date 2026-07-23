"""
Chess Combat Intelligence layer.

Converts a GameAnalysis (chess facts) into CombatIntelligence: a list of
universal combat events plus a battle profile. This module produces
structured intelligence only — it does NOT generate prose or a final
story. That stays narrative_generator's job. Nothing here decides how
an event *reads* in anime vs. fantasy vs. sci-fi framing either; each
CombatEvent carries a style-neutral event_type and description
("A fighter breaks through enemy defenses..."), and a future style
engine transforms that into "Energy blade pierces the barrier" (anime)
or "Knight breaches the fortress wall" (fantasy) — see module examples
in the project brief. No visual style is hardcoded anywhere below.

Pipeline position:

    PGN -> Metadata Normalizer -> Analysis -> Combat Mapper -> Narrative

`generate_combat_intelligence()` is the single public entry point,
taking the GameAnalysis produced by pgn_analyzer.analyze_game() (which
already carries the per-ply `moves` list and normalized `metadata`).
"""

from __future__ import annotations

from products.chess2fight.schemas import (
    CombatEvent,
    CombatEventType,
    CombatIntelligence,
    CombatProfile,
    GameAnalysis,
    GameMetadata,
    MoveRecord,
)

# Standard starting squares for knights/bishops, used to detect
# "piece development" (a non-capture move off the back rank early in
# the game) without hardcoding anything style-specific.
_DEVELOPMENT_SQUARES = {
    "white": {"b1", "g1", "c1", "f1"},
    "black": {"b8", "g8", "c8", "f8"},
}

_DEVELOPMENT_MOVE_LIMIT = 10  # only the first N moves count as "development"

# Net material points given up (via a 1-ply lookahead recapture) needed
# to call a move a "sacrifice" rather than just an ordinary trade.
_SACRIFICE_THRESHOLD = 2

_PIECE_VALUES: dict[str, int] = {
    "pawn": 1,
    "knight": 3,
    "bishop": 3,
    "rook": 5,
    "queen": 9,
    "king": 0,
}


def _is_sacrifice(moves: list[MoveRecord], i: int) -> tuple[bool, int]:
    """One-ply lookahead: if the piece that just moved to `to_square`
    is captured on the very next ply on that same square, and what this
    move itself gained (if it was a capture) is worth less than what's
    being given up, it's a sacrifice. Simple and explainable, not a
    full static-exchange evaluation."""
    this_move = moves[i]
    if i + 1 >= len(moves):
        return False, 0
    next_move = moves[i + 1]
    if not next_move.is_capture or next_move.to_square != this_move.to_square:
        return False, 0

    value_given_up = _PIECE_VALUES.get(this_move.piece_moved, 0)
    value_gained = _PIECE_VALUES.get(this_move.captured_piece or "", 0) if this_move.is_capture else 0
    net_loss = value_given_up - value_gained
    return net_loss >= _SACRIFICE_THRESHOLD, net_loss


def _is_coordinated_assault(moves: list[MoveRecord], i: int) -> bool:
    """A light heuristic for "tactical combination": this move and the
    very next move are both captures on the same exchange (an
    immediate trade sequence), or this move follows one."""
    this_move = moves[i]
    if not this_move.is_capture:
        return False
    prev_is_exchange = i > 0 and moves[i - 1].is_capture and moves[i - 1].to_square == this_move.to_square
    next_is_exchange = (
        i + 1 < len(moves) and moves[i + 1].is_capture and moves[i + 1].to_square == this_move.to_square
    )
    return prev_is_exchange or next_is_exchange


def _is_development(move: MoveRecord) -> bool:
    return (
        move.piece_moved in ("knight", "bishop")
        and not move.is_capture
        and move.move_number <= _DEVELOPMENT_MOVE_LIMIT
        and move.from_square in _DEVELOPMENT_SQUARES.get(move.color, set())
    )


def _classify_move(moves: list[MoveRecord], i: int) -> tuple[CombatEventType, int, str]:
    """Returns (event_type, intensity 1-10, description) for one move.
    Priority order (highest drama first): checkmate > sacrifice >
    breakthrough (capture+check, or a high-value capture) > check >
    coordinated exchange > plain capture > castle > queen move >
    pawn push > development > everything else."""
    move = moves[i]
    is_sacrifice, net_loss = _is_sacrifice(moves, i)

    if move.is_checkmate:
        return (
            CombatEventType.FINISHING_STRIKE,
            10,
            f"{move.move_label} delivers the finishing blow — the fight is over.",
        )

    if is_sacrifice:
        return (
            CombatEventType.CALCULATED_SACRIFICE,
            min(10, 6 + net_loss),
            f"{move.move_label} gives up material on purpose, trading position for a future strike.",
        )

    if move.is_capture and move.is_check:
        captured = move.captured_piece or "piece"
        value = _PIECE_VALUES.get(captured, 1)
        return (
            CombatEventType.BREAKTHROUGH_ATTACK,
            min(10, 7 + (1 if value >= 5 else 0)),
            f"{move.move_label} breaks through enemy defenses with a surprise strike.",
        )

    if move.is_capture:
        captured = move.captured_piece or "piece"
        value = _PIECE_VALUES.get(captured, 1)
        if value >= 5:
            return (
                CombatEventType.BREAKTHROUGH_ATTACK,
                7,
                f"{move.move_label} lands a heavy blow, taking the {captured}.",
            )
        if _is_coordinated_assault(moves, i):
            return (
                CombatEventType.COORDINATED_ASSAULT,
                6,
                f"{move.move_label} joins a rapid exchange of blows.",
            )
        return (
            CombatEventType.ATTACK_LANDED,
            3 + min(2, value),
            f"{move.move_label} lands an attack, taking the {captured}.",
        )

    if move.is_check:
        return (
            CombatEventType.CRITICAL_THREAT,
            6,
            f"{move.move_label} presses forward with a direct threat.",
        )

    if move.is_castle:
        return (
            CombatEventType.DEFENSIVE_REPOSITIONING,
            3,
            f"{move.move_label} pulls back into a fortified stance.",
        )

    if move.piece_moved == "queen":
        return (
            CombatEventType.POWER_DEPLOYMENT,
            3,
            f"{move.move_label} commits their strongest fighter to the field.",
        )

    if move.piece_moved == "pawn":
        return (
            CombatEventType.TERRITORIAL_ADVANCE,
            2,
            f"{move.move_label} pushes forward, claiming ground.",
        )

    if _is_development(move):
        return (
            CombatEventType.TACTICAL_SETUP,
            2,
            f"{move.move_label} brings a fighter into position.",
        )

    return (
        CombatEventType.STRATEGIC_POSITIONING,
        2,
        f"{move.move_label} maneuvers for advantage.",
    )


def map_combat_events(analysis: GameAnalysis) -> list[CombatEvent]:
    """Classifies every move in `analysis.moves` into a universal
    combat event. Returns one event per ply, in move order."""
    events: list[CombatEvent] = []
    for i, move in enumerate(analysis.moves):
        event_type, intensity, description = _classify_move(analysis.moves, i)
        events.append(
            CombatEvent(
                event_type=event_type,
                intensity=intensity,
                attacker=move.color,
                description=description,
                move_number=move.move_number,
                move_label=move.move_label,
            )
        )
    return events


def _infer_battle_pace(metadata: GameMetadata, analysis: GameAnalysis) -> str:
    """Prefers the actual time control when present (blitz/bullet reads
    as fast-paced, classical as strategic); falls back to tactical
    density (captures+checks per move) when time control is unknown."""
    base_seconds = None
    if metadata.time_control and "+" in metadata.time_control:
        base_part = metadata.time_control.split("+")[0]
        if base_part.isdigit():
            base_seconds = int(base_part)
    elif metadata.time_control and metadata.time_control.isdigit():
        base_seconds = int(metadata.time_control)

    if base_seconds is not None:
        if base_seconds < 600:
            return "fast"
        if base_seconds < 1500:
            return "moderate"
        return "strategic"

    tactical_density = len(analysis.tactical_moments) / max(analysis.num_moves, 1)
    return "fast" if tactical_density > 0.3 else "strategic"


def _infer_fighter_balance(metadata: GameMetadata) -> str:
    if metadata.white_rating is None or metadata.black_rating is None:
        return "unknown"
    diff = abs(metadata.white_rating - metadata.black_rating)
    if diff < 100:
        return "even"
    if diff < 300:
        return "uneven"
    return "veteran vs challenger"


def _infer_ending_type(analysis: GameAnalysis, metadata: GameMetadata) -> str:
    if metadata.winner == "draw":
        return "draw"
    if analysis.is_checkmate:
        return "checkmate"
    termination = (metadata.termination or "").lower()
    if "time" in termination:
        return "time_forfeit"
    if metadata.winner in ("white", "black"):
        return "resignation"
    return "unknown"


def build_combat_profile(analysis: GameAnalysis, events: list[CombatEvent]) -> CombatProfile:
    metadata = analysis.metadata
    return CombatProfile(
        battle_pace=_infer_battle_pace(metadata, analysis),
        fighter_balance=_infer_fighter_balance(metadata),
        ending_type=_infer_ending_type(analysis, metadata),
        winner=metadata.winner,
    )


def generate_combat_intelligence(analysis: GameAnalysis) -> CombatIntelligence:
    """Single public entry point: GameAnalysis in, CombatIntelligence
    out. Pure function — no I/O, no AI provider, fully unit-testable
    with a hand-built GameAnalysis fixture."""
    events = map_combat_events(analysis)
    profile = build_combat_profile(analysis, events)
    return CombatIntelligence(events=events, profile=profile)
