"""HTTP layer for Chess2Fight. Thin on purpose: parse the request,
delegate to FightOrchestrator, map domain exceptions to HTTP status
codes. No chess logic or AI logic lives here."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from core.ai_router import AIProvider, get_ai_provider
from core.exceptions import InvalidPGNError
from products.chess2fight.orchestrator import FightOrchestrator
from products.chess2fight.schemas import BattleMode, BattlePreferences, GenerateRequest, GenerateResponse
from products.chess2fight.style_engine import resolve_style_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chess2fight", tags=["chess2fight"])


def get_orchestrator(provider: AIProvider = Depends(get_ai_provider)) -> FightOrchestrator:
    # Constructing FightOrchestrator is cheap (no I/O) — the expensive
    # part, the AIProvider itself, is already a cached singleton.
    return FightOrchestrator(provider)


def _resolve_preferences(request: GenerateRequest) -> BattlePreferences:
    """Reconciles the two ways a client can express preferences:

    - New clients send `preferences: {...}` explicitly — used as-is.
    - Existing clients send only `pgn` (and optionally the legacy
      top-level `style` string) — a BattlePreferences is built from
      that string, with battle_mode defaulting to duel. This is the
      exact backward-compatibility path: an old request with no
      `preferences` field produces identical behavior to before this
      revision, because `style` still flows through to the same place
      it always did.
    """
    if request.preferences is not None:
        return request.preferences
    return BattlePreferences(
        battle_mode=BattleMode.DUEL,
        style=resolve_style_id(request.style).value,
        combat_intensity="cinematic",
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate_fight(
    request: GenerateRequest,
    orchestrator: FightOrchestrator = Depends(get_orchestrator),
) -> GenerateResponse:
    preferences = _resolve_preferences(request)
    try:
        return await orchestrator.generate_fight(request.pgn, preferences)
    except InvalidPGNError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error generating fight scene.")
        raise HTTPException(
            status_code=500, detail="Something went wrong generating the fight scene."
        ) from None
