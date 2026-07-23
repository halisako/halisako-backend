"""API-level tests for battle mode preferences — the 4 scenarios this
task's brief explicitly requires, exercised through the real HTTP
route (not just the internal functions, which test_battle_mode_engine.py
and test_narrative_generator.py already cover)."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

SCHOLARS_MATE_PGN = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)


# --- Test 1: default request (PGN only) -> battle_mode=duel ---------------


def test_default_request_pgn_only_defaults_to_duel_mode():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN})
    assert res.status_code == 200
    data = res.json()
    assert data["battle_mode_intelligence"]["mode"] == "duel"
    assert "FIGHTERS" in data["fight_story"]["prompt"]
    # Existing behavior (style default, facts) is unchanged.
    assert data["style_profile"]["style"] == "anime"
    assert data["game_analysis"]["winner"] == "white"


def test_legacy_style_field_still_works_without_preferences():
    """A request shaped exactly like every pre-existing test/frontend
    call (`pgn` + top-level `style`, no `preferences`) must keep
    working identically — this is the core backward-compatibility
    contract for this revision."""
    res = client.post(
        "/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "fantasy"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["style_profile"]["style"] == "fantasy"
    assert data["battle_mode_intelligence"]["mode"] == "duel"


# --- Test 2: same PGN, duel vs army -> different presentation, same facts -


def test_duel_vs_army_preferences_differ_in_presentation_not_facts():
    duel_res = client.post(
        "/api/v1/chess2fight/generate",
        json={"pgn": SCHOLARS_MATE_PGN, "preferences": {"battle_mode": "duel"}},
    )
    army_res = client.post(
        "/api/v1/chess2fight/generate",
        json={"pgn": SCHOLARS_MATE_PGN, "preferences": {"battle_mode": "army"}},
    )
    assert duel_res.status_code == 200
    assert army_res.status_code == 200
    duel, army = duel_res.json(), army_res.json()

    # Different: screenplay, environment, combat interpretation.
    assert duel["fight_story"]["prompt"] != army["fight_story"]["prompt"]
    assert duel["battle_mode_intelligence"]["environment"] != army["battle_mode_intelligence"]["environment"]
    assert duel["battle_mode_intelligence"]["unit_mapping"] != army["battle_mode_intelligence"]["unit_mapping"]

    # Same: winner, moves, chess analysis.
    assert duel["game_analysis"]["winner"] == army["game_analysis"]["winner"]
    assert duel["game_analysis"]["moves"] == army["game_analysis"]["moves"]
    assert duel["game_analysis"]["captures"] == army["game_analysis"]["captures"]
    assert duel["game_metadata"] == army["game_metadata"]
    assert duel["combat_intelligence"] == army["combat_intelligence"]
    assert duel["battle_intelligence"] == army["battle_intelligence"]


def test_preferences_can_set_both_battle_mode_and_style_together():
    res = client.post(
        "/api/v1/chess2fight/generate",
        json={
            "pgn": SCHOLARS_MATE_PGN,
            "preferences": {"battle_mode": "army", "style": "scifi", "combat_intensity": "brutal"},
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["battle_mode_intelligence"]["mode"] == "army"
    assert data["style_profile"]["style"] == "scifi"


# --- Test 3: army mode mapping (API-level confirmation) --------------------


def test_army_mode_unit_mapping_via_api():
    res = client.post(
        "/api/v1/chess2fight/generate",
        json={"pgn": SCHOLARS_MATE_PGN, "preferences": {"battle_mode": "army"}},
    )
    assert res.json()["battle_mode_intelligence"]["unit_mapping"] == {
        "pawn": "infantry",
        "knight": "cavalry",
        "bishop": "mage",
        "rook": "siege",
        "queen": "commander",
        "king": "fortress",
    }


# --- Response shape: additive only -----------------------------------------


def test_battle_mode_intelligence_is_additive_and_all_prior_fields_present():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN})
    data = res.json()
    assert set(data.keys()) == {
        "status", "game_analysis", "fight_story", "video_placeholder",
        "game_metadata", "combat_intelligence", "battle_intelligence",
        "style_profile", "battle_mode_intelligence",
    }
    bmi = data["battle_mode_intelligence"]
    assert set(bmi.keys()) == {"mode", "scale", "unit_mapping", "combat_focus", "environment"}
