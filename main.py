"""FastAPI application entry point. Run with:
uvicorn main:app --host 0.0.0.0 --port 8000
(or see Dockerfile / README for the Render-ready command).
"""

from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import chess2fight
from core.ai_router import get_ai_provider
from core.config import get_settings
from core.exceptions import Chess2FightError

settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Halisako's orchestration backend for Chess2Fight — PGN analysis "
    "and cinematic fight-scene narrative generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(chess2fight.router)


@app.exception_handler(Chess2FightError)
async def domain_error_handler(request: Request, exc: Chess2FightError) -> JSONResponse:
    # Belt-and-suspenders: api/chess2fight.py already catches
    # InvalidPGNError directly, but this catches any Chess2FightError
    # raised from elsewhere so the API never leaks a raw 500 traceback
    # for a domain error.
    logger.warning("Unhandled domain error: %s", exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health", tags=["meta"])
def health() -> dict:
    provider = get_ai_provider()
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "ai_provider_configured": settings.ai_provider,
        "ai_provider_active": type(provider).__name__,
    }


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }
