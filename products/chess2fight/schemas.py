"""Pydantic models for the Chess2Fight product: request/response shapes
and the internal structures passed between pgn_analyzer, orchestrator,
and narrative_generator.

v1.1 note: this file only gained new models and new *optional* fields
on GameAnalysis in this revision (moves, metadata) — every field that
existed before is untouched, so anything already parsing a
GenerateResponse keeps working without changes."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field
from products.cinema.schemas import CinematicSequence


# --- Request -----------------------------------------------------------


class BattleMode(str, Enum):
    """Which visualization frame the battle is interpreted through —
    orthogonal to StyleId (the visual genre). See battle_mode_engine.py
    and the "Rule 2" note in this task's engineering review for why
    these two dimensions are deliberately kept separate."""

    DUEL = "duel"
    ARMY = "army"


class BattlePreferences(BaseModel):
    """User-controllable generation preferences. Deliberately minimal
    for now — only battle_mode, style, and combat_intensity are wired
    to anything; camera_style, violence_level, age rating, realism
    level, character design, and duration are documented future
    additions, not implemented until something downstream actually
    consumes them.

    `style` here is intentionally a plain string, matching
    GenerateRequest.style's existing type exactly — NOT the StyleId
    enum the original brief for this field suggested. Making it a
    strict enum would mean an unrecognized style name gets rejected
    with a 422 at the request boundary, instead of gracefully falling
    back to "anime" the way it already does today (see
    style_engine.py's `resolve_style_id`). Keeping it a string
    preserves that existing lenient behavior exactly."""

    battle_mode: BattleMode = BattleMode.DUEL
    style: str = Field(default="anime", description="Same free-form style id as the top-level `style` field.")
    combat_intensity: str = "cinematic"


class GenerateRequest(BaseModel):
    pgn: str = Field(..., min_length=1, description="PGN text of the game to analyze.")
    style: str = Field(default="anime", description="Visual/narrative style for the fight scene.")
    preferences: BattlePreferences | None = Field(
        default=None,
        description=(
            "Optional structured preferences (battle_mode, style, combat_intensity). "
            "If omitted, the server builds one automatically using battle_mode=duel "
            "and the `style` field above — existing requests with only `pgn` (and "
            "optionally `style`) continue to work unchanged."
        ),
    )


# --- Game analysis (deterministic, derived from the PGN itself) -------


class Capture(BaseModel):
    move_number: int
    move_label: str
    san: str
    capturing_piece: str
    captured_piece: str


class TacticalMoment(BaseModel):
    move_number: int
    move_label: str
    san: str
    description: str


class TurningPoint(BaseModel):
    move_number: int
    move_label: str
    san: str
    description: str


class GameMetadata(BaseModel):
    """Normalized, source-agnostic PGN metadata — the same shape
    whether the PGN came from Chess.com, Lichess, or anywhere else.
    Every field has a safe default so a platform that omits a header
    (e.g. no WhiteElo) never breaks processing; see
    products/chess2fight/metadata_normalizer.py."""

    white_player: str = "Unknown"
    black_player: str = "Unknown"
    white_rating: int | None = None
    black_rating: int | None = None
    opening: str = "Unclassified Opening"
    time_control: str = "Unknown"
    termination: str = "Unknown"
    winner: str = "unknown"  # "white" | "black" | "draw" | "unknown"


class MoveRecord(BaseModel):
    """One ply of the game, with enough detail for combat_mapper to
    classify it without re-parsing the PGN. Internal/analysis detail —
    new in this revision, additive."""

    ply: int
    move_number: int
    move_label: str
    san: str
    color: str  # "white" | "black"
    piece_moved: str
    from_square: str
    to_square: str
    is_capture: bool
    captured_piece: str | None = None
    is_check: bool
    is_checkmate: bool
    is_castle: bool


class GameAnalysis(BaseModel):
    white_player: str
    black_player: str
    opening: str
    num_moves: int
    winner: str  # "white" | "black" | "draw" | "unknown"
    is_checkmate: bool
    checkmate_move_number: int | None = None
    captures: list[Capture] = Field(default_factory=list)
    tactical_moments: list[TacticalMoment] = Field(default_factory=list)
    turning_points: list[TurningPoint] = Field(default_factory=list)
    # --- new in this revision, purely additive ---
    moves: list[MoveRecord] = Field(default_factory=list)
    metadata: GameMetadata = Field(default_factory=GameMetadata)


# --- Combat intelligence (new: universal, style-agnostic combat events) -


class CombatEventType(str, Enum):
    TERRITORIAL_ADVANCE = "territorial_advance"  # pawn advancement
    TACTICAL_SETUP = "tactical_setup"  # piece development
    ATTACK_LANDED = "attack_landed"  # routine capture
    BREAKTHROUGH_ATTACK = "breakthrough_attack"  # high-significance capture
    CRITICAL_THREAT = "critical_threat"  # check
    FINISHING_STRIKE = "finishing_strike"  # checkmate
    CALCULATED_SACRIFICE = "calculated_sacrifice"  # sacrifice
    DEFENSIVE_REPOSITIONING = "defensive_repositioning"  # castling
    POWER_DEPLOYMENT = "power_deployment"  # queen movement
    COORDINATED_ASSAULT = "coordinated_assault"  # tactical combination
    STRATEGIC_POSITIONING = "strategic_positioning"  # everything else


class CombatEvent(BaseModel):
    event_type: CombatEventType
    intensity: int = Field(ge=1, le=10)
    attacker: str  # "white" | "black"
    description: str
    move_number: int
    move_label: str


class CombatProfile(BaseModel):
    battle_pace: str  # "fast" | "moderate" | "strategic"
    fighter_balance: str  # "even" | "uneven" | "veteran vs challenger" | "unknown"
    ending_type: str  # "checkmate" | "resignation" | "draw" | "time_forfeit" | "unknown"
    winner: str


class CombatIntelligence(BaseModel):
    events: list[CombatEvent] = Field(default_factory=list)
    profile: CombatProfile


# --- Battle intelligence (new: cinematic-but-evidence-based interpretation) -


class BattleArc(str, Enum):
    BLITZ_EXECUTION = "blitz_execution"
    TACTICAL_AMBUSH = "tactical_ambush"
    WAR_OF_ATTRITION = "war_of_attrition"
    COMEBACK = "comeback"
    SIEGE = "siege"
    GAMBIT_ASSAULT = "gambit_assault"
    FINAL_DUEL = "final_duel"


class CombatStyle(str, Enum):
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    BALANCED = "balanced"
    CALCULATED = "calculated"
    CHAOTIC = "chaotic"
    PATIENT = "patient"
    DESPERATE = "desperate"
    OVERWHELMING = "overwhelming"


class PersonalityProfile(BaseModel):
    label: str
    # Deliberately hedged, evidence-grounded phrasing (e.g. "appeared
    # to", "forced tactical sequence") rather than asserting intent —
    # see battle_director.py's module docstring.
    rationale: str


class FighterPersonality(BaseModel):
    white: PersonalityProfile
    black: PersonalityProfile


class BattleIntelligence(BaseModel):
    battle_arc: BattleArc
    combat_style: CombatStyle
    fighter_personality: FighterPersonality


# --- Battle mode intelligence (new: duel vs. army presentation frame) -


class BattleModeIntelligence(BaseModel):
    """Output of battle_mode_engine.py. Presentation-only, like
    StyleProfile — describes how the SAME combat/battle intelligence
    reads as either a 1v1 duel or an army-scale war. Never touches
    chess facts; never generates narrative prose (that's
    narrative_generator's job)."""

    mode: BattleMode
    scale: str
    unit_mapping: dict[str, str] = Field(default_factory=dict)
    combat_focus: list[str] = Field(default_factory=list)
    environment: str


# --- Style profile (new: presentation-only, doesn't touch chess analysis) -


class StyleId(str, Enum):
    ANIME = "anime"
    FANTASY = "fantasy"
    MODERN_WARFARE = "modern_warfare"
    SUPERHERO = "superhero"
    SCIFI = "scifi"


class StyleProfile(BaseModel):
    style: StyleId
    weapons: list[str] = Field(default_factory=list)
    powers: list[str] = Field(default_factory=list)
    environment: str
    visual_effects: list[str] = Field(default_factory=list)
    finisher: str


# --- Fight story (partly deterministic, partly AI/template-generated) -


class FightStory(BaseModel):
    winner: str
    opening: str
    fight_style: str
    best_move: str
    turning_point: str
    battle_summary: str
    prompt: str
    estimated_length: str


class VideoPlaceholder(BaseModel):
    status: str = "not_generated"
    message: str = "Video rendering is not implemented yet."
    estimated_duration_seconds: int


class GenerateResponse(BaseModel):
    status: str = "completed"
    game_analysis: GameAnalysis
    fight_story: FightStory
    video_placeholder: VideoPlaceholder

    # Intelligence layers
    game_metadata: GameMetadata
    combat_intelligence: CombatIntelligence
    battle_intelligence: BattleIntelligence
    style_profile: StyleProfile
    battle_mode_intelligence: BattleModeIntelligence

    # Halisako Cinematic Engine output
    cinematic_sequence: CinematicSequence
