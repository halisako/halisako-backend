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

Running the tests:
```bash
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/pytest
```

## Project structure

```
main.py                          FastAPI app, CORS, /health, exception handlers
api/
  chess2fight.py                  POST /api/v1/chess2fight/generate; reconciles
                                   legacy `style` vs. new `preferences` requests
core/
  config.py                       Settings (env vars), via pydantic-settings
  ai_router.py                    AIProvider abstraction + 5 implementations
                                   (present but dormant — see Design notes)
  exceptions.py                   Chess2FightError, InvalidPGNError, AIProviderError
products/
  chess2fight/
    schemas.py                    Pydantic request/response models
    metadata_normalizer.py         Source-agnostic PGN header extraction
    pgn_analyzer.py                analyze_game(pgn) — python-chess, no AI involved
    combat_mapper.py               Chess analysis -> universal combat events
    battle_director.py             Combat events -> battle_arc, combat_style, personalities
    style_engine.py                Battle/combat intelligence -> visual style profile
    battle_mode_engine.py          Battle mode (duel/army) -> presentation metadata (new)
    narrative_generator.py         Battle Screenplay Generator — builds FightStory from
                                    every upstream intelligence layer, deterministically
    orchestrator.py                FightOrchestrator — ties the whole pipeline together
tests/
  test_metadata_normalizer.py      Unit tests, hand-built PGN headers
  test_combat_mapper.py            Unit tests, hand-built GameAnalysis fixtures
  test_battle_director.py          Unit + PGN-driven tests (battle_arc classification)
  test_style_engine.py             Unit tests, vocabulary/determinism/content-safety
  test_battle_mode_engine.py       Unit tests, including the exact required unit mapping
  test_narrative_generator.py      Unit tests, determinism + per-field + duel/army checks
  test_pgn_analyzer.py             Unit + regression tests, real PGNs
  test_api_regression.py           Full-stack tests via FastAPI's TestClient
  test_battle_mode_api.py          Full-stack tests for the 4 required battle-mode scenarios
```

Pipeline, in order:

```
PGN -> Metadata Normalizer -> PGN Analyzer -> Combat Mapper ->
Battle Director -> Style Engine -> Battle Mode Interpreter ->
Narrative Generator
```

Style Engine and the Battle Mode Interpreter both consume only Combat
Intelligence + Battle Intelligence, independently of each other —
neither imports the other, which is what makes genre (style) and
frame (battle mode) fully independent, combinable dimensions rather
than a hierarchy.

## User Preferences

Two independent dimensions control how a battle is presented — neither
affects chess analysis at all, only presentation:

```
Battle Mode
├── Duel   (default) — a 1v1 cinematic fight
└── Army              — a battlefield / war interpretation

Visual Style
├── Anime
├── Fantasy
├── Sci-Fi
├── Modern Warfare
└── Superhero
```

They combine freely — Duel + Anime, Army + Fantasy, Army + Sci-Fi, and
so on are all valid, and Battle Mode deliberately isn't a Style Engine
concept (see `battle_mode_engine.py`'s module docstring): neither
module imports the other, so genre and mode can vary independently by
construction, not just by convention.

Send preferences as a nested object:

```json
{
  "pgn": "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0",
  "preferences": {
    "battle_mode": "army",
    "style": "scifi",
    "combat_intensity": "cinematic"
  }
}
```

Or omit `preferences` entirely and use the legacy flat shape — battle
mode defaults to `duel`, and the top-level `style` field (defaulting
to `"anime"`) is used exactly as it always has been:

```json
{ "pgn": "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0", "style": "fantasy" }
```

`combat_intensity` is accepted and stored on `BattlePreferences` but
not yet wired to any generation behavior — a deliberate placeholder,
like `video_placeholder` was from the start, for a future task.
Documented future additions to preferences (not yet implemented):
age rating, camera preference, realism level, character design,
duration.

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
    "tactical_moments": [...], "turning_points": [...],
    "moves": [{"ply": 1, "move_number": 1, "move_label": "1. e4", "san": "e4",
               "color": "white", "piece_moved": "pawn", "from_square": "e2",
               "to_square": "e4", "is_capture": false, "is_check": false,
               "is_checkmate": false, "is_castle": false}, ...],
    "metadata": { "...": "same shape as top-level game_metadata below" }
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
  },
  "game_metadata": {
    "white_player": "maia5", "black_player": "n1000",
    "white_rating": 1400, "black_rating": 700,
    "opening": "Italian Game", "time_control": "300+3",
    "termination": "Normal", "winner": "white"
  },
  "combat_intelligence": {
    "events": [
      {"event_type": "territorial_advance", "intensity": 2, "attacker": "white",
       "description": "1. e4 pushes forward, claiming ground.",
       "move_number": 1, "move_label": "1. e4"},
      "...",
      {"event_type": "finishing_strike", "intensity": 10, "attacker": "white",
       "description": "4. Qxf7# delivers the finishing blow — the fight is over.",
       "move_number": 4, "move_label": "4. Qxf7#"}
    ],
    "profile": {
      "battle_pace": "fast", "fighter_balance": "veteran vs challenger",
      "ending_type": "checkmate", "winner": "white"
    }
  },
  "battle_intelligence": { "...": "see Design notes below — battle_arc, combat_style, fighter_personality" },
  "style_profile": { "...": "see Design notes below — weapons, powers, environment, visual_effects, finisher" },
  "battle_mode_intelligence": {
    "mode": "duel",
    "scale": "explosive duel",
    "unit_mapping": {
      "pawn": "basic strike", "knight": "flanking strike", "bishop": "ranged technique",
      "rook": "defensive guard", "queen": "signature technique", "king": "final stand"
    },
    "combat_focus": ["total domination", "one-sided mastery", "an overwhelming finish"],
    "environment": "a lightning-fast dueling arena"
  }
}
```

`game_metadata`, `combat_intelligence`, `battle_intelligence`,
`style_profile`, and `battle_mode_intelligence` are all new, additive
top-level fields added across several revisions — `game_analysis`,
`fight_story`, and `video_placeholder` are unchanged in shape from the
very first version of this backend (`fight_story`'s *content* has
gotten much richer, but not its field names). See "Design notes"
below for a full walkthrough of each layer, including worked examples.

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

**Battle Mode is independent of Style, by construction, not by
convention.** `battle_mode_engine.py` and `style_engine.py` both take
only `CombatIntelligence` + `BattleIntelligence` as input — neither
module imports the other or knows the other exists. That's what makes
"Army + Fantasy" and "Duel + Sci-Fi" both simply valid combinations
rather than special cases: there's no shared state or hierarchy to
keep consistent, just two independent presentation lookups that
`narrative_generator.py` combines when building the screenplay.

**Backward compatibility for the new `preferences` field is handled at
the API boundary, not by giving it a clever default.** `GenerateRequest
.preferences` defaults to `None`, not a pre-filled `BattlePreferences`
object — this is what lets `api/chess2fight.py` distinguish "the client
didn't send preferences at all" (build one from the legacy `style`
string, `battle_mode=duel`) from "the client explicitly configured
everything" (use it as-is). A default-factory object would have made
that distinction impossible to recover once the request was parsed.

**`BattlePreferences.style` is a plain string, not the `StyleId` enum
a strict schema would suggest.** Making it an enum would mean an
unrecognized style name gets rejected with a 422 at the request
boundary — a real behavior change from today, where an unrecognized
style string already degrades gracefully to "anime"
(`style_engine.resolve_style_id`). Keeping it a string preserves that
exact leniency for forward-compatibility with style names that don't
exist yet.

**Pipeline (as of this revision):**
```
PGN -> Metadata Normalizer -> Analysis -> Combat Mapper -> Narrative
```
`metadata_normalizer.py` and `combat_mapper.py` are both new. Neither
changes what `narrative_generator.py` receives or produces —
`combat_intelligence` is computed independently and appended to the
response as a new field, so `fight_story` is byte-for-byte identical
to what it was before this revision for the same input (verified in
`tests/test_pgn_analyzer.py` and `tests/test_api_regression.py`).

**Known pre-existing quirk, not introduced by this revision.**
`GameAnalysis.white_player`/`black_player` (the original fields) can
show `"?"` for a PGN with no `White`/`Black` tags at all — python-chess
auto-fills those two specific tags with the PGN spec's own placeholder
`"?"` rather than omitting them, so the existing `headers.get("White",
"White")` fallback never actually engages. This was already true
before this revision; testing the new metadata_normalizer against a
tagless PGN is what surfaced it. The new `game_metadata.white_player`/
`black_player` correctly normalize this to `"Unknown"` — the old field
was deliberately left untouched to avoid bundling an unrelated behavior
change into this PR. See "Risks identified" in the change summary.

**Source-agnostic by construction, not by branching.**
`metadata_normalizer.py` never checks "is this a Chess.com PGN" or "is
this a Lichess PGN" — it just reads a fixed set of standard PGN tag
names (`White`, `Black`, `WhiteElo`, `BlackElo`, `Opening`, `ECO`,
`TimeControl`, `Termination`, `Result`) and falls back to a safe
default for whichever ones happen to be absent. A PGN from a platform
this was never tested against works the same way, as long as it uses
standard PGN tags (nearly all do).

**The Combat Mapper produces intelligence, not stories.** Every
`CombatEvent` has a style-neutral `event_type`
(`breakthrough_attack`, `calculated_sacrifice`, etc.) and a plain
description — no visual style (anime, fantasy, sci-fi, ...) is
hardcoded anywhere in `combat_mapper.py`. A future style engine reads
the same event and renders it however that style calls for; see the
worked example in the module docstring.

**Sacrifice and "coordinated assault" detection are one-ply-lookahead
heuristics, not a static-exchange evaluation.** A move is flagged as a
sacrifice if the piece that just moved is recaptured on the very next
ply for less than it's worth. This catches the common case (a piece
hangs and gets taken) but can't see further-out tactics — Légal's Mate
is a good example of the limit: White's 5.Nxe5 isn't flagged as a
sacrifice by this heuristic (nothing recaptures on e5 next), even
though the whole point of the trap is that White doesn't mind losing
the knight. A real static-exchange evaluation would need a chess
engine; out of scope for this revision, same reasoning as the opening
book below.

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
