"""
Deterministic chess-game analysis using python-chess. Nothing in this
module calls an AI provider — every field here is derived directly from
the game notation, so it's exact and reproducible, not an LLM's
interpretation of the game.

Note on scope: `_guess_opening` and the tactical/turning-point detection
below are lightweight heuristics, not a real chess engine evaluation.
python-chess gives board representation and rules, not positional
evaluation — a proper "best move" or accurate turning-point detector
would run the game through Stockfish (python-chess's `chess.engine`
module supports this) and compare centipawn scores move to move. That's
a reasonable next step; this module intentionally stays engine-free to
keep the deployment simple (no engine binary to install/manage) for an
MVP that only needs *plausible, explainable* signals, not a certified
one.

`move_number` throughout this module is the standard chess move number
(as printed in a PGN, e.g. the "7" in "7. Nd5#") — not a half-move/ply
count. Since White's and Black's halves of the same move share that
number, `move_label` disambiguates ("7. Nd5#" for White, "7...Bxd1" for
Black), matching how any chess-literate reader expects moves to be cited.
"""

from __future__ import annotations

import io

import chess
import chess.pgn

from core.exceptions import InvalidPGNError
from products.chess2fight.schemas import Capture, GameAnalysis, TacticalMoment, TurningPoint

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

# A small, hand-picked set of common openings matched against the first
# few moves in SAN. Not an ECO database — just enough to name the
# openings a casual game is likely to actually be. Falls back to a
# generic structural description when nothing matches.
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


def _guess_opening(headers: chess.pgn.Headers, san_moves: list[str]) -> str:
    for tag in ("Opening", "ECO"):
        value = headers.get(tag)
        if value:
            return value

    for pattern, name in _OPENING_BOOK:
        if san_moves[: len(pattern)] == pattern:
            return name

    return "Unclassified Opening"


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
    result = headers.get("Result", "*")

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

        captured_piece_type: int | None = None
        if is_capture:
            if board.is_en_passant(move):
                captured_piece_type = chess.PAWN
            else:
                captured_piece_type = board.piece_type_at(move.to_square)

        capturing_piece_type = board.piece_type_at(move.from_square)

        board.push(move)
        balances.append(_material_balance(board))

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

        is_mate_move = board.is_checkmate()
        if is_mate_move:
            is_checkmate = True
            checkmate_move_number = full_move_number

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

    winner_color: str | None = None
    if result == "1-0":
        winner_color = "white"
    elif result == "0-1":
        winner_color = "black"

    winner_field = (
        "draw" if result == "1/2-1/2" else (winner_color if winner_color else "unknown")
    )

    # Turning point: the single ply with the biggest material swing in
    # the eventual winner's favor, plus the mating move itself if there
    # is one. This is a material-counting heuristic, so it can miss
    # genuine sacrifices (a queen sac that leads to mate reads as a
    # "loss" of material) — see module docstring.
    turning_points: list[TurningPoint] = []
    if len(balances) > 1:
        deltas = [balances[i] - balances[i - 1] for i in range(1, len(balances))]
        sign = 1 if winner_color == "white" else -1 if winner_color == "black" else 0
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
        opening=_guess_opening(headers, san_moves),
        num_moves=total_full_moves,
        winner=winner_field,
        is_checkmate=is_checkmate,
        checkmate_move_number=checkmate_move_number,
        captures=captures,
        tactical_moments=tactical_moments,
        turning_points=turning_points,
    )
