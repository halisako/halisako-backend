"""HTTP layer for Chess2Fight. Thin on purpose: parse the request,
delegate to FightOrchestrator, map domain exceptions to HTTP status
codes. No chess logic or AI logic lives here."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from core.ai_router import AIProvider, get_ai_provider
from core.exceptions import Chess2FightError, InvalidPGNError
from products.chess2fight.orchestrator import FightOrchestrator
from products.chess2fight.rendering.pipeline import (
    FightVideoPipeline,
    FightVideoResponse,
    RenderVideoRequest,
)
from products.chess2fight.schemas import BattleMode, BattlePreferences, GenerateRequest, GenerateResponse
from products.chess2fight.style_engine import resolve_style_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chess2fight", tags=["chess2fight"])


def get_orchestrator(provider: AIProvider = Depends(get_ai_provider)) -> FightOrchestrator:
    # Constructing FightOrchestrator is cheap (no I/O) — the expensive
    # part, the AIProvider itself, is already a cached singleton.
    return FightOrchestrator(provider)

def get_video_pipeline(
    provider: AIProvider = Depends(get_ai_provider),
) -> FightVideoPipeline:
    return FightVideoPipeline(provider)

def _resolve_preferences(
    pgn_style: str,
    preferences: BattlePreferences | None,
) -> BattlePreferences:
    if preferences is not None:
        return preferences

    return BattlePreferences(
        battle_mode=BattleMode.DUEL,
        style=resolve_style_id(pgn_style).value,
        combat_intensity="cinematic",
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate_fight(
    request: GenerateRequest,
    orchestrator: FightOrchestrator = Depends(get_orchestrator),
) -> GenerateResponse:
    preferences = _resolve_preferences(request.style, request.preferences)
    try:
        return await orchestrator.generate_fight(request.pgn, preferences)
    except InvalidPGNError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error generating fight scene.")
        raise HTTPException(
            status_code=500, detail="Something went wrong generating the fight scene."
        ) from None

@router.post("/render", response_model=FightVideoResponse)
async def render_fight_video(
    request: RenderVideoRequest,
    pipeline: FightVideoPipeline = Depends(get_video_pipeline),
) -> FightVideoResponse:
    """Runs the complete Sprint 3 pipeline from PGN to fight.mp4."""
    preferences = _resolve_preferences(request.style, request.preferences)

    try:
        return await pipeline.run(
            request.pgn,
            preferences,
            fps=request.fps,
            width=request.width,
            height=request.height,
            frame_duration_seconds=request.frame_duration_seconds,
        )

    except InvalidPGNError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Chess2FightError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    except Exception:
        logger.exception("Unexpected error rendering fight video.")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong rendering the fight video.",
        ) from None