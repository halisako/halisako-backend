# Halisako Chess2Fight Backend

The first real Halisako orchestration backend: takes a chess PGN,
analyzes the game, and produces a structured cinematic fight
description. No video generation yet — that's the `video_placeholder`
field, deliberately unfilled.

## Quickstart

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn main:app --reload
```

Open http://localhost:8000/docs for interactive API docs, or
http://localhost:8000/health.

No environment variables are required to run this — it defaults to
`AI_PROVIDER=template`, a deterministic fallback with no external
dependencies, so the endpoint works immediately. Copy `.env.example` to
`.env` to configure a real provider.

## Project structure

```
main.py                          FastAPI app, CORS, /health, exception handlers
api/
  chess2fight.py                  POST /api/v1/chess2fight/generate
core/
  config.py                       Settings (env vars), via pydantic-settings
  ai_router.py                    AIProvider abstraction + 5 implementations
  exceptions.py                   Chess2FightError, InvalidPGNError, AIProviderError
products/
  chess2fight/
    schemas.py                    Pydantic request/response models
    pgn_analyzer.py                analyze_game(pgn) — python-chess, no AI involved
    narrative_generator.py         Builds FightStory from GameAnalysis + AIProvider
    orchestrator.py                FightOrchestrator — ties the above together
```

## API

### `POST /api/v1/chess2fight/generate`

Request:
```json
{ "pgn": "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0", "style": "anime" }
```

Response (200):
```json
{
  "status": "completed",
  "game_analysis": {
    "white_player": "...", "black_player": "...", "opening": "...",
    "num_moves": 4, "winner": "white", "is_checkmate": true,
    "checkmate_move_number": 4,
    "captures": [{"move_number": 4, "move_label": "4. Qxf7#", "san": "Qxf7#",
                  "capturing_piece": "queen", "captured_piece": "pawn"}],
    "tactical_moments": [...], "turning_points": [...]
  },
  "fight_story": {
    "winner": "White wins by checkmate", "opening": "...",
    "fight_style": "Blitz Aggression", "best_move": "...",
    "turning_point": "...", "battle_summary": "...",
    "prompt": "SCENE: ...", "estimated_length": "12 sec"
  },
  "video_placeholder": {
    "status": "not_generated",
    "message": "Video rendering is not implemented yet...",
    "estimated_duration_seconds": 12
  }
}
```

400 for an unparseable PGN, 422 for a malformed request body (missing
`pgn`), 500 (logged server-side, generic message to the client) for
anything unexpected.

### `GET /health`

Reports both the *configured* and *actually active* AI provider —
these can differ if, say, `AI_PROVIDER=openai` is set but
`OPENAI_API_KEY` isn't, in which case the service silently falls back
to the template provider rather than crashing, and `/health` is where
that shows up:

```json
{
  "status": "ok", "ai_provider_configured": "openai",
  "ai_provider_active": "TemplateProvider"
}
```

## Design notes

**Facts vs. creative content are deliberately separated.** `winner`,
`opening`, `best_move`, `turning_point`, and `estimated_length` are all
computed directly from the parsed game in `pgn_analyzer.py` — an AI
provider never touches them, so they can't be hallucinated. Only
`fight_style`, `battle_summary`, and the cinematic `prompt` go through
`AIProvider.generate_text()`. See `narrative_generator.py`.

**`pgn_analyzer.py`'s opening and tactical-moment detection are
heuristics, not a chess engine.** There's no Stockfish integration here
— python-chess gives board representation and rules, not positional
evaluation. Opening names come from a small hand-picked book matched
against the first few moves (or PGN `Opening`/`ECO` headers, if
present); "turning points" are the largest material swings plus the
mating move, which is explainable but can misread genuine sacrifices
as material "losses." Adding real engine evaluation
(`chess.engine` + a Stockfish binary in the Dockerfile) is the natural
next step if this needs to get sharper — it wasn't in scope for this
MVP.

**Move numbers follow standard chess notation, not ply count.** Every
`move_number` is the number a PGN actually prints (the "7" in
"7. Nd5#"), and `move_label` disambiguates White's/Black's half of
that move ("7. Nd5#" vs. "7...Bxd1").

**The AI provider is never hardcoded.** `core/ai_router.py` defines the
`AIProvider` interface and five implementations: `TemplateProvider`
(the default — deterministic, no dependencies, parses the same
"ANALYSIS FACTS" block out of the prompt via regex that a real LLM
would read as context), `OpenAIProvider`, `AnthropicProvider`,
`GeminiProvider` (uses the current `google-genai` SDK — the older
`google-generativeai` package is fully end-of-life), and `LocalProvider`
(Ollama-style HTTP, no key required). `get_ai_provider()` reads
`AI_PROVIDER` and falls back to `TemplateProvider` — logging a warning
— if the configured provider can't initialize (e.g. missing key),
rather than crashing the app.

**Every dependency in `requirements.txt` is pinned to a version that
was actually installed and tested** in a clean virtualenv while
building this, not guessed — including catching that
`google-generativeai` is deprecated and switching to `google-genai`
before it ever shipped.

## Deploying to Render

1. Push this `backend/` directory to its own GitHub repo (or a
   subdirectory of a monorepo, setting Render's "Root Directory" to
   `backend`).
2. In Render: **New → Web Service**, connect the repo, environment
   **Docker** (Render will find the `Dockerfile` automatically).
3. Render sets `$PORT` automatically — the Dockerfile already reads it.
4. Add environment variables from `.env.example` as needed. None are
   required to deploy; `AI_PROVIDER=template` works out of the box.
5. Set `CORS_ORIGINS` to include your actual frontend origin (e.g.
   `https://halisako.com`) if it differs from the default.
6. Deploy. Confirm with `curl https://<your-service>.onrender.com/health`.

## What's intentionally not here

Per the brief: no auth, no database, no payments, no video generation.
`video_placeholder` exists in the response shape so the frontend never
has to change when real video rendering is added later — it just
starts getting a real URL instead of `null`.
