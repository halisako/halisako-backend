"""Integration tests: the Shot Timeline as it actually appears in the
real HTTP API response, not just as an object returned by
generate_shot_timeline() directly."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

SCHOLARS_MATE_PGN = (
    '[Event "Example"]\n[White "Halisako"]\n[Black "Guest"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
)


def test_generate_response_includes_shot_timeline():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    assert res.status_code == 200
    data = res.json()
    assert "shot_timeline" in data


def test_shot_timeline_has_correct_shape():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    timeline = res.json()["shot_timeline"]
    assert set(timeline.keys()) == {"shots", "total_duration_seconds", "shot_count"}
    assert timeline["shot_count"] == len(timeline["shots"])

    required_shot_fields = {
        "shot_id", "sequence_order", "shot_type", "camera_angle", "camera_motion",
        "focus", "duration_seconds", "environment", "lighting", "mood",
        "source_moves", "description",
    }
    for shot in timeline["shots"]:
        assert set(shot.keys()) == required_shot_fields


def test_every_pre_existing_response_field_still_present():
    """The Shot Timeline is additive — every field that existed before
    this feature must still be present and correctly populated."""
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "anime"})
    data = res.json()
    for field in (
        "status", "game_analysis", "fight_story", "video_placeholder", "game_metadata",
        "combat_intelligence", "battle_intelligence", "style_profile", "battle_mode_intelligence",
    ):
        assert field in data
    assert data["game_analysis"]["winner"] == "white"
    assert data["fight_story"]["winner"] == "White wins by checkmate"


def test_legacy_style_only_request_still_works_and_includes_shot_timeline():
    """A request shaped exactly like every pre-existing caller (no
    `preferences`, just `pgn` and the legacy top-level `style`) still
    works, and now additionally includes a shot_timeline."""
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "fantasy"})
    assert res.status_code == 200
    data = res.json()
    assert data["style_profile"]["style"] == "fantasy"
    assert data["shot_timeline"]["shot_count"] > 0


def test_shot_timeline_determinism_across_repeated_requests():
    responses = [
        client.post("/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "style": "scifi"}).json()["shot_timeline"]
        for _ in range(5)
    ]
    assert all(r == responses[0] for r in responses)


def test_army_mode_shot_timeline_reflects_different_presentation():
    duel = client.post(
        "/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "preferences": {"battle_mode": "duel"}}
    ).json()
    army = client.post(
        "/api/v1/chess2fight/generate", json={"pgn": SCHOLARS_MATE_PGN, "preferences": {"battle_mode": "army"}}
    ).json()
    assert duel["shot_timeline"]["shots"][0]["environment"] != army["shot_timeline"]["shots"][0]["environment"]
    # Same underlying chess facts regardless of presentation.
    assert duel["game_analysis"]["winner"] == army["game_analysis"]["winner"]


def test_invalid_pgn_still_returns_400_not_a_broken_shot_timeline():
    res = client.post("/api/v1/chess2fight/generate", json={"pgn": "garbage"})
    assert res.status_code == 400
