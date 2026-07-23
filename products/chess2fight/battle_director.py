"""
Battle Director — converts CombatIntelligence + GameAnalysis into a
higher-level cinematic interpretation: an overall battle_arc, a
combat_style, and a per-fighter personality for white and black.

Pipeline position:

    PGN -> Metadata Normalizer -> Analysis -> Combat Mapper ->
    Battle Director (here) -> Narrative Generator

Like combat_mapper, this is a pure, deterministic function — no AI
provider call, so nothing here can hallucinate a fact about the game.
Every classification is grounded in a countable signal: move count,
capture/tactical density, material swings, or which side an event is
attributed to.

On intent: neither python-chess nor a rule-based heuristic can know
whether a player *meant* to sacrifice a piece or was simply forced into
it — that's a claim about a mind, not a position. So this module never
asserts intent. Where a label like "gambit_assault" or a personality
like "The Silent Assassin" implies a sacrifice, the rationale text uses
hedged phrasing ("appeared to give up material") rather than asserting
it was deliberate. This mirrors combat_mapper's own documented
limitation — a "sacrifice" here means "lost material that wasn't
immediately regained," not "confirmed intentional sacrifice."

Material balance is reconstructed from GameAnalysis.moves' existing
capture data (color + captured_piece), not a new field — see
`_material_balance_history`. This keeps the integration to a single
new file plus one new response field; pgn_analyzer.py and
combat_mapper.py are untouched by this revision.

Known gap: this reconstruction doesn't account for pawn promotions
(a non-capturing promotion changes material but isn't a "capture").
Rare in this product's typical short games; flagged rather than fixed
here to avoid touching pgn_analyzer.py for this revision.
"""

from __future__ import annotations

from products.chess2fight.schemas import (
    BattleArc,
    BattleIntelligence,
    CombatEventType,
    CombatIntelligence,
    CombatStyle,
    FighterPersonality,
    GameAnalysis,
    PersonalityProfile,
)

PIECE_VALUES: dict[str, int] = {
    "pawn": 1,
    "knight": 3,
    "bishop": 3,
    "rook": 5,
    "queen": 9,
    "king": 0,
}

# Thresholds — documented here rather than buried in conditionals, so
# they're easy to tune without hunting through the logic below.
SHORT_GAME_MOVES = 8       # blitz_execution / overwhelming ceiling
EARLY_GAME_MOVES = 10      # "early" enough to read as a gambit
MEDIUM_GAME_MOVES = 16     # tactical_ambush ceiling
LONG_GAME_MOVES = 16       # war_of_attrition / siege / patient floor
ATTRITION_DENSITY = 0.3    # captures-per-move, long game -> attrition vs siege
HIGH_TACTICAL_DENSITY = 0.3
LOW_DENSITY = 0.15
HIGH_VOLATILITY = 6        # material-balance range, in points
BEHIND_THRESHOLD = 2       # points down before counting as "ever behind"

_DRAMA_EVENTS = {CombatEventType.CALCULATED_SACRIFICE, CombatEventType.BREAKTHROUGH_ATTACK}
_ATTACK_EVENTS = {
    CombatEventType.BREAKTHROUGH_ATTACK,
    CombatEventType.CALCULATED_SACRIFICE,
    CombatEventType.ATTACK_LANDED,
    CombatEventType.CRITICAL_THREAT,
    CombatEventType.FINISHING_STRIKE,
    CombatEventType.COORDINATED_ASSAULT,
    CombatEventType.POWER_DEPLOYMENT,
}


def _material_balance_history(analysis: GameAnalysis) -> list[int]:
    """Running material balance (white minus black) after each ply,
    reconstructed from `analysis.moves` alone. See module docstring for
    the one known gap (promotions)."""
    balance = 0
    history = [0]
    for move in analysis.moves:
        if move.is_capture and move.captured_piece:
            value = PIECE_VALUES.get(move.captured_piece, 0)
            balance += value if move.color == "white" else -value
        history.append(balance)
    return history


def _was_side_ever_behind(color: str, history: list[int]) -> bool:
    if color == "white":
        return any(b < -BEHIND_THRESHOLD for b in history)
    if color == "black":
        return any(b > BEHIND_THRESHOLD for b in history)
    return False


def _material_volatility(history: list[int]) -> int:
    return (max(history) - min(history)) if history else 0


def _determine_battle_arc(analysis: GameAnalysis, combat: CombatIntelligence) -> BattleArc:
    num_moves = analysis.num_moves
    winner = analysis.winner
    capture_density = len(analysis.captures) / max(num_moves, 1)
    history = _material_balance_history(analysis)

    drama_events = [e for e in combat.events if e.event_type in _DRAMA_EVENTS]
    winner_drama = [e for e in drama_events if e.attacker == winner]

    if analysis.is_checkmate and num_moves <= SHORT_GAME_MOVES and not drama_events:
        return BattleArc.BLITZ_EXECUTION

    if winner_drama and min(e.move_number for e in winner_drama) <= EARLY_GAME_MOVES:
        return BattleArc.GAMBIT_ASSAULT

    if drama_events and num_moves <= MEDIUM_GAME_MOVES:
        return BattleArc.TACTICAL_AMBUSH

    if num_moves >= LONG_GAME_MOVES:
        return BattleArc.WAR_OF_ATTRITION if capture_density >= ATTRITION_DENSITY else BattleArc.SIEGE

    if _was_side_ever_behind(winner, history):
        return BattleArc.COMEBACK

    return BattleArc.FINAL_DUEL


def _determine_combat_style(analysis: GameAnalysis) -> CombatStyle:
    num_moves = analysis.num_moves
    tactical_density = len(analysis.tactical_moments) / max(num_moves, 1)
    capture_density = len(analysis.captures) / max(num_moves, 1)
    history = _material_balance_history(analysis)
    volatility = _material_volatility(history)

    if _was_side_ever_behind(analysis.winner, history):
        return CombatStyle.DESPERATE
    if analysis.is_checkmate and num_moves <= SHORT_GAME_MOVES:
        return CombatStyle.OVERWHELMING
    if volatility >= HIGH_VOLATILITY:
        return CombatStyle.CHAOTIC
    if tactical_density >= HIGH_TACTICAL_DENSITY:
        return CombatStyle.AGGRESSIVE
    if num_moves >= LONG_GAME_MOVES:
        return CombatStyle.PATIENT if capture_density < LOW_DENSITY else CombatStyle.CALCULATED
    if analysis.winner == "draw":
        return CombatStyle.BALANCED
    if capture_density < LOW_DENSITY:
        return CombatStyle.DEFENSIVE
    return CombatStyle.CALCULATED


def _personality_for(color: str, analysis: GameAnalysis, combat: CombatIntelligence) -> PersonalityProfile:
    """Evidence-bound personality for one side. Every branch cites the
    specific signal behind it in `rationale`; nothing here claims
    intent the data can't support."""
    own_events = [e for e in combat.events if e.attacker == color]
    attack_count = sum(1 for e in own_events if e.event_type in _ATTACK_EVENTS)
    attack_ratio = attack_count / max(len(own_events), 1)
    has_sacrifice = any(e.event_type == CombatEventType.CALCULATED_SACRIFICE for e in own_events)
    delivered_finish = any(e.event_type == CombatEventType.FINISHING_STRIKE for e in own_events)
    history = _material_balance_history(analysis)
    ever_behind = _was_side_ever_behind(color, history)
    won = analysis.winner == color
    drew = analysis.winner == "draw"

    if won:
        if delivered_finish and has_sacrifice:
            return PersonalityProfile(
                label="The Silent Assassin",
                rationale=(
                    "Appeared to give up material during a forced tactical sequence, "
                    "then delivered the finishing blow."
                ),
            )
        if delivered_finish and attack_ratio >= 0.4:
            return PersonalityProfile(
                label="The Relentless Predator",
                rationale="Generated a high share of attacking moments and closed the game out directly.",
            )
        if ever_behind:
            return PersonalityProfile(
                label="The Resilient",
                rationale="Was materially behind at points in the game before recovering to win.",
            )
        if attack_ratio >= 0.3:
            return PersonalityProfile(
                label="The Tactical Commander",
                rationale="Maintained an actively aggressive event profile on the way to victory.",
            )
        return PersonalityProfile(
            label="The Patient Strategist",
            rationale="Won without relying on high-risk tactics or major material swings.",
        )

    if drew:
        if attack_ratio < LOW_DENSITY:
            return PersonalityProfile(
                label="The Fortress",
                rationale="Held a mostly defensive posture through to a drawn result.",
            )
        return PersonalityProfile(
            label="The Balanced Combatant",
            rationale="Traded chances evenly without either side breaking through.",
        )

    # lost
    if analysis.num_moves >= LONG_GAME_MOVES:
        return PersonalityProfile(
            label="The Survivor",
            rationale="Kept the fight going through an extended sequence despite the eventual loss.",
        )
    if attack_ratio < LOW_DENSITY or ever_behind:
        return PersonalityProfile(
            label="The Cornered Defender",
            rationale="Spent much of the game under pressure in a desperate defense rather than attacking.",
        )
    return PersonalityProfile(
        label="The Contender",
        rationale="Contested the game without a standout defensive or offensive signature.",
    )


def generate_battle_intelligence(
    analysis: GameAnalysis, combat: CombatIntelligence
) -> BattleIntelligence:
    """Single public entry point: GameAnalysis + CombatIntelligence in,
    BattleIntelligence out. Pure function — no AI provider, no I/O,
    fully unit-testable with hand-built fixtures (see
    tests/test_battle_director.py)."""
    return BattleIntelligence(
        battle_arc=_determine_battle_arc(analysis, combat),
        combat_style=_determine_combat_style(analysis),
        fighter_personality=FighterPersonality(
            white=_personality_for("white", analysis, combat),
            black=_personality_for("black", analysis, combat),
        ),
    )
