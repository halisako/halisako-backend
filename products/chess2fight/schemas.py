"""Pydantic models for the Chess2Fight product: request/response shapes
and the internal structures passed between pgn_analyzer, orchestrator,
and narrative_generator."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- Request -----------------------------------------------------------


class GenerateRequest(BaseModel):
    pgn: str = Field(..., min_length=1, description="PGN text of the game to analyze.")
    style: str = Field(default="anime", description="Visual/narrative style for the fight scene.")


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
