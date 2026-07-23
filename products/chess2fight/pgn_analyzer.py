"""
Deterministic chess-game analysis using python-chess. Nothing in this
module calls an AI provider — every field here is derived directly from
the game notation, so it's exact and reproducible, not an LLM's
interpretation of the game.

Note on scope: opening detection and the tactical/turning-point
detection below are lightweight heuristics, not a real chess engine
evaluation. python-chess gives board representation and rules, not
positional evaluation — a proper "best move" or accurate turning-point
detector would run the game through Stockfish (python-chess's
`chess.engine` module supports this) and compare centipawn scores move
to move. That's a reasonable next step; this module intentionally
stays engine-free to keep the deployment simple (no engine binary to
install/manage) for an MVP that only needs *plausible, explainable*
signals, not a certified one.

`move_number` throughout this module is the standard chess move number
(as printed in a PGN, e.g. the "7" in "7. Nd5#") — not a half-move/ply
count. Since White's and Black's halves of the same move share that
number, `move_label` disambiguates ("7. Nd5#" for White, "7...Bxd1" for
Black), matching how any chess-literate reader expects moves to be cited.

v1.1: this module now also builds a full per-ply `moves` list (every
move, not just captures/checks) and calls out to
metadata_normalizer.normalize_game_metadata() for source-agnostic
header extraction — both purely additive to GameAnalysis. Opening
detection moved to metadata_normalizer.py (it's genuinely a metadata
concern, and that module needed it anyway); everything else here is
unchanged from the previous revision.
"""

from __future__ import annotations

import io

import chess
import chess.pgn

from core.exceptions import InvalidPGNError
from products.chess2fight.metadata_normalizer import normalize_game_metadata
from products.chess2fight.schemas import (
    Capture,
    GameAnalysis,
    MoveRecord,
    TacticalMoment,
    TurningPoint,
)

PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

PIECE_NAMES: dict[int, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def _material_balance(board: chess.Board) -> int:
    balance = 0
    for piece_type, value in PIECE_VALUES.items():
        balance += value * len(board.pieces(piece_type, chess.WHITE))
        balance -= value * len(board.pieces(piece_type, chess.BLACK))
    return balance


def _move_label(full_move_number: int, is_white: bool, san: str) -> str:
    return f"{full_move_number}. {san}" if is_white else f"{full_move_number}...{san}"


def analyze_game(pgn: str) -> GameAnalysis:
    """Parse a PGN string and return a structured GameAnalysis.

    Raises InvalidPGNError if the text doesn't contain a parseable game.
    Signature is unchanged from the previous revision — callers
    (FightOrchestrator) don't need to change.
    """
    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
    except Exception as exc:  # python-chess is lenient; malformed input rarely raises
        raise InvalidPGNError(f"Could not parse PGN: {exc}") from exc

    if game is None:
        raise InvalidPGNError("No valid game found in the supplied PGN.")

    headers = game.headers
    white_player = headers.get("White", "White")
    black_player = headers.get("Black", "Black")

    if not game.variations:
        # python-chess parses unrecognizable text leniently into an
        # empty Game rather than raising or returning None — catch that
        # case explicitly, since a fight scene needs at least one move.
        raise InvalidPGNError("The supplied text contains no recognizable chess moves.")

    board = game.board()
    san_moves: list[str] = []
    move_labels: list[str] = []
    full_move_numbers: list[int] = []
    captures: list[Capture] = []
    tactical_moments: list[TacticalMoment] = []
    moves: list[MoveRecord] = []
    balances: list[int] = [_material_balance(board)]  # balance BEFORE each ply, index-aligned

    ply = 0
    is_checkmate = False
    checkmate_move_number: int | None = None

    node = game
    while node.variations:
        next_node = node.variations[0]
        move = next_node.move
        ply += 1
        is_white = ply % 2 == 1
        full_move_number = (ply + 1) // 2

        san = board.san(move)
        label = _move_label(full_move_number, is_white, san)
        san_moves.append(san)
        move_labels.append(label)
        full_move_numbers.append(full_move_number)

        is_capture = board.is_capture(move)
        gives_check = board.gives_check(move)
        is_castle = board.is_castling(move)

        captured_piece_type: int | None = None
        if is_capture:
            if board.is_en_passant(move):
                captured_piece_type = chess.PAWN
            else:
                captured_piece_type = board.piece_type_at(move.to_square)

        capturing_piece_type = board.piece_type_at(move.from_square)
        from_square_name = chess.square_name(move.from_square)
        to_square_name = chess.square_name(move.to_square)

        board.push(move)
        balances.append(_material_balance(board))

        is_mate_move = board.is_checkmate()
        if is_mate_move:
            is_checkmate = True
            checkmate_move_number = full_move_number

        moves.append(
            MoveRecord(
                ply=ply,
                move_number=full_move_number,
                move_label=label,
                san=san,
                color="white" if is_white else "black",
                piece_moved=PIECE_NAMES.get(capturing_piece_type, "piece"),
                from_square=from_square_name,
                to_square=to_square_name,
                is_capture=is_capture,
                captured_piece=(
                    PIECE_NAMES.get(captured_piece_type, "piece") if is_capture else None
                ),
                is_check=gives_check,
                is_checkmate=is_mate_move,
                is_castle=is_castle,
            )
        )

        if is_capture and captured_piece_type is not None:
            captures.append(
                Capture(
                    move_number=full_move_number,
                    move_label=label,
                    san=san,
                    capturing_piece=PIECE_NAMES.get(capturing_piece_type, "piece"),
                    captured_piece=PIECE_NAMES.get(captured_piece_type, "piece"),
                )
            )

        is_significant_capture = is_capture and captured_piece_type is not None and (
            PIECE_VALUES.get(captured_piece_type, 0) >= 3
        )
        if is_significant_capture or gives_check or is_mate_move:
            descriptor = "Delivers checkmate" if is_mate_move else (
                f"Captures the {PIECE_NAMES.get(captured_piece_type, 'piece')}"
                if is_significant_capture
                else "Delivers check"
            )
            tactical_moments.append(
                TacticalMoment(
                    move_number=full_move_number, move_label=label, san=san, description=descriptor
                )
            )

        node = next_node

    total_full_moves = full_move_numbers[-1] if full_move_numbers else 0

    metadata = normalize_game_metadata(headers, san_moves)

    # Turning point: the single ply with the biggest material swing in
    # the eventual winner's favor, plus the mating move itself if there
    # is one. This is a material-counting heuristic, so it can miss
    # genuine sacrifices (a queen sac that leads to mate reads as a
    # "loss" of material) — see module docstring.
    turning_points: list[TurningPoint] = []
    if len(balances) > 1:
        deltas = [balances[i] - balances[i - 1] for i in range(1, len(balances))]
        sign = 1 if metadata.winner == "white" else -1 if metadata.winner == "black" else 0
        if sign != 0:
            best_ply_idx = max(range(len(deltas)), key=lambda i: sign * deltas[i])
            if sign * deltas[best_ply_idx] > 0:
                turning_points.append(
                    TurningPoint(
                        move_number=full_move_numbers[best_ply_idx],
                        move_label=move_labels[best_ply_idx],
                        san=san_moves[best_ply_idx],
                        description=(
                            f"{move_labels[best_ply_idx]} swings the material balance decisively."
                        ),
                    )
                )

    if checkmate_move_number is not None and not any(
        tp.move_label == move_labels[-1] for tp in turning_points
    ):
        turning_points.append(
            TurningPoint(
                move_number=checkmate_move_number,
                move_label=move_labels[-1],
                san=san_moves[-1],
                description=f"{move_labels[-1]} delivers checkmate.",
            )
        )

    return GameAnalysis(
        white_player=white_player,
        black_player=black_player,
        opening=metadata.opening,
        num_moves=total_full_moves,
        winner=metadata.winner,
        is_checkmate=is_checkmate,
        checkmate_move_number=checkmate_move_number,
        captures=captures,
        tactical_moments=tactical_moments,
        turning_points=turning_points,
        moves=moves,
        metadata=metadata,
    )
