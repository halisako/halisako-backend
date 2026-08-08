"""Full-stack API tests via FastAPI's TestClient — exercises the real
route, not just the internal functions. This is the closest thing to
"does the existing API still respond correctly" that runs in CI."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

CHESSCOM_STYLE_PGN = (
    '[Event "Live Chess"]\n[Site "Chess.com"]\n[White "maia5"]\n[Black "n1000"]\n'
    '[Result "1-0"]\n[WhiteElo "1400"]\n[BlackElo "700"]\n[TimeControl "300+3"]\n'
    '[Termination "Normal"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0'
)

LICHESS_STYLE_PGN = (
    '[Event "Rated Blitz game"]\n[Site "https://lichess.org/abc123"]\n'
    '[White "maia5"]\n[Black "n1000"]\n[Result "1-0"]\n[WhiteElo "1400"]\n'
    '[BlackElo "700"]\n[ECO "C50"]\n[Opening "Italian Game"]\n'
    '[TimeControl "300+3"]\n[Termination "Normal"]\n'
    '[WhiteRatingDiff "+8"]\n[BlackRatingDiff "-8"]\n\n'
    "1. e4 e5 2. Nf3 d6 3. Bc4 Bg4 4. Nc3 g6 5. Nxe5 Bxd1 6. Bxf7+ Ke7 7. Nd5# 1-0"
)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_generate_chesscom_style_pgn_full_response_shape():
    res = client.post(
        "/api/v1/chess2fight/generate", json={"pgn": CHESSCOM_STYLE_PGN, "style": "anime"}
    )
    assert res.status_code == 200
    data = res.json()

    assert set(data.keys()) == {
    "status",
    "game_analysis",
    "fight_story",
    "video_placeholder",
    "game_metadata",
    "combat_intelligence",
    "battle_intelligence",
    "style_profile",
    "battle_mode_intelligence",
    "shot_timeline",
    "cinematic_sequence",
    "scene_composition",
    "prompted_timeline",
    }

    # --- existing contract, unchanged ---
    assert data["status"] == "completed"
    assert set(data["game_analysis"].keys()) >= {
        "white_player", "black_player", "opening", "num_moves", "winner",
        "is_checkmate", "checkmate_move_number", "captures", "tactical_moments",
        "turning_points",
    }
    assert set(data["fight_story"].keys()) == {
        "winner", "opening", "fight_style", "best_move", "turning_point",
        "battle_summary", "prompt", "estimated_length",
    }
    assert set(data["video_placeholder"].keys()) == {
        "status", "message", "estimated_duration_seconds",
    }

    # --- new, additive fields ---
    assert data["game_metadata"]["white_rating"] == 1400
    assert data["game_metadata"]["black_rating"] == 700
    assert "combat_intelligence" in data
    assert len(data["combat_intelligence"]["events"]) == data["game_analysis"]["num_moves"] * 2 - 1
    assert data["combat_intelligence"]["profile"]["winner"] == "white"

    # --- v1.2: battle_intelligence, also additive ---
    assert "battle_intelligence" in data
    assert data["battle_intelligence"]["battle_arc"] == "blitz_execution"
    assert set(data["battle_intelligence"].keys()) == {
        "battle_arc", "combat_style", "fighter_personality",
    }
    assert set(data["battle_intelligence"]["fighter_personality"].keys()) == {"white", "black"}
    for side in ("white", "black"):
        profile = data["battle_intelligence"]["fighter_personality"][side]
        assert set(profile.keys()) == {"label", "rationale"}
        assert profile["label"]
        assert profile["rationale"]

    # --- v1.3: style_profile, also additive ---
    assert "style_profile" in data
    sp = data["style_profile"]
    assert set(sp.keys()) == {"style", "weapons", "powers", "environment", "visual_effects", "finisher"}
    assert sp["style"] == "anime"  # request used the default style
    assert isinstance(sp["weapons"], list) and sp["weapons"]
    assert isinstance(sp["powers"], list) and sp["powers"]
    assert isinstance(sp["visual_effects"], list) and sp["visual_effects"]
    assert sp["environment"]
    assert sp["finisher"]
    # --- v3.0: shot_timeline, additive ---
    assert "shot_timeline" in data

    timeline = data["shot_timeline"]

    assert timeline["shot_count"] > 0
    assert timeline["total_duration_seconds"] > 0
    assert len(timeline["shots"]) == timeline["shot_count"]

    first_shot = timeline["shots"][0]

    assert "sequence_order" in first_shot
    assert "shot_type" in first_shot
    assert "camera_angle" in first_shot
    assert "camera_motion" in first_shot
    assert "focus" in first_shot
    assert "duration_seconds" in first_shot


    # --- Sprint 3 Prompt 2: scene composition ---
    assert "scene_composition" in data

    scene = data["scene_composition"]


    assert "scene_continuity" in scene

    continuity = scene["scene_continuity"]

    assert set(continuity.keys()) == {
        "white_fighter",
        "black_fighter",
        "arena",
        "lighting_continuity",
        "cinematic_art_style",
        "color_palette",
}
    white = continuity["white_fighter"]

    assert set(white.keys()) == {
        "hair",
        "facial_features",
        "clothing",
        "armor",
        "weapon",
    }

    black = continuity["black_fighter"]

    assert set(black.keys()) == {
        "hair",
        "facial_features",
        "clothing",
        "armor",
        "weapon",
    }

    arena = continuity["arena"]

    assert set(arena.keys()) == {
        "layout",
        "weather",
        "time_of_day",
}


    first = scene["shots"][0]

    assert "scene" in first

    assert first["scene"] == continuity

    continuity = data["scene_composition"]["scene_continuity"]

    for shot in data["scene_composition"]["shots"]:
        assert shot["scene"] == continuity

    assert len(scene["shots"]) == data["shot_timeline"]["shot_count"]


def test_generate_lichess_style_pgn():
    res = client.post(
        "/api/v1/chess2fight/generate", json={"pgn": LICHESS_STYLE_PGN, "style": "anime"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["game_metadata"]["opening"] == "Italian Game"
    assert len(data["combat_intelligence"]["events"]) == 13
    assert any(
        e["event_type"] == "breakthrough_attack" for e in data["combat_intelligence"]["events"]
    )
    assert data["battle_intelligence"]["battle_arc"] == "gambit_assault"


def test_invalid_pgn_still_returns_400():
    res = client.post(
        "/api/v1/chess2fight/generate", json={"pgn": "not chess at all", "style": "anime"}
    )
    assert res.status_code == 400


def test_missing_pgn_field_still_returns_422():
    res = client.post("/api/v1/chess2fight/generate", json={"style": "anime"})
    assert res.status_code == 422


def test_existing_fields_identical_after_battle_director_addition():
    """Required regression test: every field that existed before the
    Battle Director revision must compute the exact same value after
    it — not just the same shape. Battle Director only adds a new
    top-level field; it must change nothing else."""
    res = client.post(
        "/api/v1/chess2fight/generate", json={"pgn": CHESSCOM_STYLE_PGN, "style": "anime"}
    )
    data = res.json()

    assert data["status"] == "completed"

    ga = data["game_analysis"]
    assert ga["white_player"] == "maia5"
    assert ga["black_player"] == "n1000"
    assert ga["opening"] == "Italian Game (early queen sortie)"
    assert ga["num_moves"] == 4
    assert ga["winner"] == "white"
    assert ga["is_checkmate"] is True
    assert ga["checkmate_move_number"] == 4
    assert len(ga["captures"]) == 1
    assert ga["captures"][0]["move_label"] == "4. Qxf7#"
    assert len(ga["tactical_moments"]) == 1
    assert len(ga["turning_points"]) == 1
    assert len(ga["moves"]) == 7

    fs = data["fight_story"]
    assert fs["winner"] == "White wins by checkmate"
    assert fs["opening"] == "Italian Game (early queen sortie)"
    # fight_style/estimated_length content itself is intentionally
    # improved by a later revision (the Battle Screenplay Generator
    # task) — this test now only pins the facts that predate and
    # survive that change; see
    # test_fight_story_content_upgraded_but_everything_else_identical
    # below for the current-content assertions.
    assert isinstance(fs["fight_style"], str) and fs["fight_style"]
    assert isinstance(fs["estimated_length"], str) and fs["estimated_length"]

    assert data["video_placeholder"]["status"] == "not_generated"
    assert isinstance(data["video_placeholder"]["estimated_duration_seconds"], int)

    gm = data["game_metadata"]
    assert gm["white_rating"] == 1400
    assert gm["black_rating"] == 700
    assert gm["time_control"] == "300+3"
    assert gm["termination"] == "Normal"

    ci = data["combat_intelligence"]
    assert ci["profile"]["battle_pace"] == "fast"
    assert ci["profile"]["fighter_balance"] == "veteran vs challenger"
    assert ci["profile"]["ending_type"] == "checkmate"

    # ...and the one new field is present alongside all of the above, unchanged.
    assert "battle_intelligence" in data


def test_existing_fields_identical_after_style_engine_addition():
    """Same intent as the battle_director regression test above, for
    this revision: style_profile is additive, everything else must be
    byte-identical to before it existed."""
    res = client.post(
        "/api/v1/chess2fight/generate", json={"pgn": CHESSCOM_STYLE_PGN, "style": "anime"}
    )
    data = res.json()

    ga = data["game_analysis"]
    assert ga["white_player"] == "maia5"
    assert ga["num_moves"] == 4
    assert ga["winner"] == "white"
    assert len(ga["moves"]) == 7

    assert isinstance(data["fight_story"]["fight_style"], str) and data["fight_story"]["fight_style"]
    assert data["game_metadata"]["white_rating"] == 1400
    assert data["combat_intelligence"]["profile"]["fighter_balance"] == "veteran vs challenger"
    assert data["battle_intelligence"]["battle_arc"] == "blitz_execution"

    assert "style_profile" in data
    assert "shot_timeline" in data
    


def test_generate_respects_requested_style_field():
    """The existing `style` request field (unchanged type, unchanged
    default) now also selects the style_profile — no new request field
    was added."""
    res = client.post(
        "/api/v1/chess2fight/generate",
        json={"pgn": CHESSCOM_STYLE_PGN, "style": "fantasy"},
    )
    assert res.status_code == 200
    assert res.json()["style_profile"]["style"] == "fantasy"


def test_fight_story_content_upgraded_but_everything_else_identical():
    """The Battle Screenplay Generator revision: fight_story's CONTENT
    is expected and required to change (that's the point of the
    upgrade) — every other field, including video_placeholder's
    derived second count, must not."""
    res = client.post(
        "/api/v1/chess2fight/generate", json={"pgn": CHESSCOM_STYLE_PGN, "style": "anime"}
    )
    assert res.status_code == 200
    data = res.json()

    # --- untouched: identical to every prior revision ---
    ga = data["game_analysis"]
    assert ga["white_player"] == "maia5"
    assert ga["num_moves"] == 4
    assert ga["winner"] == "white"
    assert ga["is_checkmate"] is True
    assert len(ga["moves"]) == 7
    assert data["game_metadata"]["white_rating"] == 1400
    assert data["combat_intelligence"]["profile"]["fighter_balance"] == "veteran vs challenger"
    assert data["battle_intelligence"]["battle_arc"] == "blitz_execution"
    assert data["battle_intelligence"]["combat_style"] == "overwhelming"
    assert data["style_profile"]["style"] == "anime"

    # --- new in this revision: fight_story is now screenplay-grade ---
    fs = data["fight_story"]
    assert set(fs.keys()) == {
        "winner", "opening", "fight_style", "best_move", "turning_point",
        "battle_summary", "prompt", "estimated_length",
    }
    assert fs["winner"] == "White wins by checkmate"  # fact, unchanged
    assert fs["opening"] == "Italian Game (early queen sortie)"  # fact, unchanged
    assert "Predator" in fs["fight_style"] or "overwhelms" in fs["fight_style"]
    assert "4. Qxf7#" in fs["best_move"]
    assert "4. Qxf7#" in fs["turning_point"]
    assert data["battle_intelligence"]["fighter_personality"]["white"]["label"] in fs["battle_summary"]
    assert "-" in fs["estimated_length"] and "sec" in fs["estimated_length"]

    # The screenplay prompt covers every required section. This request
    # has no `preferences`, so it defaults to battle_mode=duel — hence
    # FIGHTERS, not FORCES (see test_battle_mode_engine.py and
    # test_narrative_generator.py for the army-mode equivalents).
    for section in (
        "STYLE", "SETTING", "ENVIRONMENT", "LIGHTING", "CAMERA", "FIGHTERS",
        "White Fighter", "Black Fighter", "PERSONALITIES", "BATTLE SCALE", "VISUAL STYLE",
        "WEAPONS", "POWERS", "VISUAL EFFECTS", "COMBAT CHOREOGRAPHY",
        "ENDING", "FINAL SHOT",
    ):
        assert section in fs["prompt"], f"missing screenplay section: {section}"

    # video_placeholder's seconds are still validly derived from the
    # (now range-shaped) estimated_length, not the old digit-mash bug.
    assert 10 <= data["video_placeholder"]["estimated_duration_seconds"] <= 90


def test_same_pgn_and_style_always_produce_identical_narrative():
    """Required determinism test: same PGN + same style -> identical
    fight_story, across repeated real HTTP requests."""
    responses = [
        client.post(
            "/api/v1/chess2fight/generate", json={"pgn": CHESSCOM_STYLE_PGN, "style": "scifi"}
        ).json()["fight_story"]
        for _ in range(5)
    ]
    assert all(r == responses[0] for r in responses)


def test_same_pgn_different_styles_produce_different_narratives():
    """Required test: same PGN + different styles -> significantly
    different fight_story (not just the style_profile)."""
    stories = {}
    for style in ("anime", "fantasy", "modern_warfare", "superhero", "scifi"):
        res = client.post(
            "/api/v1/chess2fight/generate", json={"pgn": CHESSCOM_STYLE_PGN, "style": style}
        )
        stories[style] = res.json()["fight_story"]

    fight_styles = {s["fight_style"] for s in stories.values()}
    battle_summaries = {s["battle_summary"] for s in stories.values()}
    prompts = {s["prompt"] for s in stories.values()}
    assert len(fight_styles) == 5
    assert len(battle_summaries) == 5
    assert len(prompts) == 5

    # The underlying facts stay the same across every style.
    assert len({s["winner"] for s in stories.values()}) == 1
    assert len({s["opening"] for s in stories.values()}) == 1
