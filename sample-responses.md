# Chess2Fight — Sample API Responses (post-upgrade)

Both PGNs below are illustrative — the code blocks in the request came
through empty, so no real move data or headers were provided for either
"winded_wayz vs SARDA_SUII" or "maia5 vs n1000". These are short,
representative games built in each platform's header style (Chess.com:
no ECO/Opening tag; Lichess: ECO/Opening/RatingDiff present) and run
through the actual upgraded backend — not hand-written JSON.

---

## 1. Chess.com-style sample — winded_wayz vs SARDA_SUII

Request PGN:
```
[Event "Live Chess"]
[Site "Chess.com"]
[White "winded_wayz"]
[Black "SARDA_SUII"]
[Result "1-0"]
[WhiteElo "1550"]
[BlackElo "1490"]
[TimeControl "600+5"]
[Termination "Normal"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
```

Response:
```json
{
    "status": "completed",
    "game_analysis": {
        "white_player": "winded_wayz",
        "black_player": "SARDA_SUII",
        "opening": "Italian Game (early queen sortie)",
        "num_moves": 4,
        "winner": "white",
        "is_checkmate": true,
        "checkmate_move_number": 4,
        "captures": [
            {
                "move_number": 4,
                "move_label": "4. Qxf7#",
                "san": "Qxf7#",
                "capturing_piece": "queen",
                "captured_piece": "pawn"
            }
        ],
        "tactical_moments": [
            {
                "move_number": 4,
                "move_label": "4. Qxf7#",
                "san": "Qxf7#",
                "description": "Delivers checkmate"
            }
        ],
        "turning_points": [
            {
                "move_number": 4,
                "move_label": "4. Qxf7#",
                "san": "Qxf7#",
                "description": "4. Qxf7# swings the material balance decisively."
            }
        ],
        "moves": [
            {
                "ply": 1, "move_number": 1, "move_label": "1. e4", "san": "e4",
                "color": "white", "piece_moved": "pawn", "from_square": "e2",
                "to_square": "e4", "is_capture": false, "captured_piece": null,
                "is_check": false, "is_checkmate": false, "is_castle": false
            },
            {
                "ply": 2, "move_number": 1, "move_label": "1...e5", "san": "e5",
                "color": "black", "piece_moved": "pawn", "from_square": "e7",
                "to_square": "e5", "is_capture": false, "captured_piece": null,
                "is_check": false, "is_checkmate": false, "is_castle": false
            },
            {
                "ply": 3, "move_number": 2, "move_label": "2. Qh5", "san": "Qh5",
                "color": "white", "piece_moved": "queen", "from_square": "d1",
                "to_square": "h5", "is_capture": false, "captured_piece": null,
                "is_check": false, "is_checkmate": false, "is_castle": false
            },
            {
                "ply": 4, "move_number": 2, "move_label": "2...Nc6", "san": "Nc6",
                "color": "black", "piece_moved": "knight", "from_square": "b8",
                "to_square": "c6", "is_capture": false, "captured_piece": null,
                "is_check": false, "is_checkmate": false, "is_castle": false
            },
            {
                "ply": 5, "move_number": 3, "move_label": "3. Bc4", "san": "Bc4",
                "color": "white", "piece_moved": "bishop", "from_square": "f1",
                "to_square": "c4", "is_capture": false, "captured_piece": null,
                "is_check": false, "is_checkmate": false, "is_castle": false
            },
            {
                "ply": 6, "move_number": 3, "move_label": "3...Nf6", "san": "Nf6",
                "color": "black", "piece_moved": "knight", "from_square": "g8",
                "to_square": "f6", "is_capture": false, "captured_piece": null,
                "is_check": false, "is_checkmate": false, "is_castle": false
            },
            {
                "ply": 7, "move_number": 4, "move_label": "4. Qxf7#", "san": "Qxf7#",
                "color": "white", "piece_moved": "queen", "from_square": "h5",
                "to_square": "f7", "is_capture": true, "captured_piece": "pawn",
                "is_check": true, "is_checkmate": true, "is_castle": false
            }
        ],
        "metadata": {
            "white_player": "winded_wayz",
            "black_player": "SARDA_SUII",
            "white_rating": 1550,
            "black_rating": 1490,
            "opening": "Italian Game (early queen sortie)",
            "time_control": "600+5",
            "termination": "Normal",
            "winner": "white"
        }
    },
    "fight_story": {
        "winner": "White wins by checkmate",
        "opening": "Italian Game (early queen sortie)",
        "fight_style": "Blitz Aggression",
        "best_move": "4. Qxf7# — delivers checkmate",
        "turning_point": "4. Qxf7# — 4. Qxf7# swings the material balance decisively.",
        "battle_summary": "Out of the Italian Game (early queen sortie), White presses the advantage across 4 moves, trading 1 blow before sealing the fight in a decisive final strike.",
        "prompt": "SCENE: Neutral arena, dramatic side lighting.\nFIGHTERS: White and Black, styles shaped by Italian Game (early queen sortie).\nBEATS: 1 escalating exchange building to the final blow.\nFINISH: White lands the decisive strike.\nSTYLE: Sharp linework, fast cuts on impact frames.",
        "estimated_length": "12 sec"
    },
    "video_placeholder": {
        "status": "not_generated",
        "message": "Video rendering is not implemented yet — this is analysis + narrative only.",
        "estimated_duration_seconds": 12
    },
    "game_metadata": {
        "white_player": "winded_wayz",
        "black_player": "SARDA_SUII",
        "white_rating": 1550,
        "black_rating": 1490,
        "opening": "Italian Game (early queen sortie)",
        "time_control": "600+5",
        "termination": "Normal",
        "winner": "white"
    },
    "combat_intelligence": {
        "events": [
            {"event_type": "territorial_advance", "intensity": 2, "attacker": "white", "description": "1. e4 pushes forward, claiming ground.", "move_number": 1, "move_label": "1. e4"},
            {"event_type": "territorial_advance", "intensity": 2, "attacker": "black", "description": "1...e5 pushes forward, claiming ground.", "move_number": 1, "move_label": "1...e5"},
            {"event_type": "power_deployment", "intensity": 3, "attacker": "white", "description": "2. Qh5 commits their strongest fighter to the field.", "move_number": 2, "move_label": "2. Qh5"},
            {"event_type": "tactical_setup", "intensity": 2, "attacker": "black", "description": "2...Nc6 brings a fighter into position.", "move_number": 2, "move_label": "2...Nc6"},
            {"event_type": "tactical_setup", "intensity": 2, "attacker": "white", "description": "3. Bc4 brings a fighter into position.", "move_number": 3, "move_label": "3. Bc4"},
            {"event_type": "tactical_setup", "intensity": 2, "attacker": "black", "description": "3...Nf6 brings a fighter into position.", "move_number": 3, "move_label": "3...Nf6"},
            {"event_type": "finishing_strike", "intensity": 10, "attacker": "white", "description": "4. Qxf7# delivers the finishing blow — the fight is over.", "move_number": 4, "move_label": "4. Qxf7#"}
        ],
        "profile": {
            "battle_pace": "moderate",
            "fighter_balance": "even",
            "ending_type": "checkmate",
            "winner": "white"
        }
    }
}
```

---

## 2. Lichess-style sample — maia5 vs n1000

Request PGN:
```
[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[White "maia5"]
[Black "n1000"]
[Result "1-0"]
[WhiteElo "1400"]
[BlackElo "700"]
[ECO "C50"]
[Opening "Italian Game"]
[TimeControl "300+3"]
[Termination "Normal"]
[WhiteRatingDiff "+8"]
[BlackRatingDiff "-8"]

1. e4 e5 2. Nf3 d6 3. Bc4 Bg4 4. Nc3 g6 5. Nxe5 Bxd1 6. Bxf7+ Ke7 7. Nd5# 1-0
```

Response:
```json
{
    "status": "completed",
    "game_analysis": {
        "white_player": "maia5",
        "black_player": "n1000",
        "opening": "Italian Game",
        "num_moves": 7,
        "winner": "white",
        "is_checkmate": true,
        "checkmate_move_number": 7,
        "captures": [
            {"move_number": 5, "move_label": "5. Nxe5", "san": "Nxe5", "capturing_piece": "knight", "captured_piece": "pawn"},
            {"move_number": 5, "move_label": "5...Bxd1", "san": "Bxd1", "capturing_piece": "bishop", "captured_piece": "queen"},
            {"move_number": 6, "move_label": "6. Bxf7+", "san": "Bxf7+", "capturing_piece": "bishop", "captured_piece": "pawn"}
        ],
        "tactical_moments": [
            {"move_number": 5, "move_label": "5...Bxd1", "san": "Bxd1", "description": "Captures the queen"},
            {"move_number": 6, "move_label": "6. Bxf7+", "san": "Bxf7+", "description": "Delivers check"},
            {"move_number": 7, "move_label": "7. Nd5#", "san": "Nd5#", "description": "Delivers checkmate"}
        ],
        "turning_points": [
            {"move_number": 5, "move_label": "5. Nxe5", "san": "Nxe5", "description": "5. Nxe5 swings the material balance decisively."},
            {"move_number": 7, "move_label": "7. Nd5#", "san": "Nd5#", "description": "7. Nd5# delivers checkmate."}
        ],
        "moves": [
            {"ply": 1, "move_number": 1, "move_label": "1. e4", "san": "e4", "color": "white", "piece_moved": "pawn", "from_square": "e2", "to_square": "e4", "is_capture": false, "captured_piece": null, "is_check": false, "is_checkmate": false, "is_castle": false},
            {"ply": 2, "move_number": 1, "move_label": "1...e5", "san": "e5", "color": "black", "piece_moved": "pawn", "from_square": "e7", "to_square": "e5", "is_capture": false, "captured_piece": null, "is_check": false, "is_checkmate": false, "is_castle": false},
            {"ply": 3, "move_number": 2, "move_label": "2. Nf3", "san": "Nf3", "color": "white", "piece_moved": "knight", "from_square": "g1", "to_square": "f3", "is_capture": false, "captured_piece": null, "is_check": false, "is_checkmate": false, "is_castle": false},
            {"ply": 4, "move_number": 2, "move_label": "2...d6", "san": "d6", "color": "black", "piece_moved": "pawn", "from_square": "d7", "to_square": "d6", "is_capture": false, "captured_piece": null, "is_check": false, "is_checkmate": false, "is_castle": false},
            {"ply": 5, "move_number": 3, "move_label": "3. Bc4", "san": "Bc4", "color": "white", "piece_moved": "bishop", "from_square": "f1", "to_square": "c4", "is_capture": false, "captured_piece": null, "is_check": false, "is_checkmate": false, "is_castle": false},
            {"ply": 6, "move_number": 3, "move_label": "3...Bg4", "san": "Bg4", "color": "black", "piece_moved": "bishop", "from_square": "c8", "to_square": "g4", "is_capture": false, "captured_piece": null, "is_check": false, "is_checkmate": false, "is_castle": false},
            {"ply": 7, "move_number": 4, "move_label": "4. Nc3", "san": "Nc3", "color": "white", "piece_moved": "knight", "from_square": "b1", "to_square": "c3", "is_capture": false, "captured_piece": null, "is_check": false, "is_checkmate": false, "is_castle": false},
            {"ply": 8, "move_number": 4, "move_label": "4...g6", "san": "g6", "color": "black", "piece_moved": "pawn", "from_square": "g7", "to_square": "g6", "is_capture": false, "captured_piece": null, "is_check": false, "is_checkmate": false, "is_castle": false},
            {"ply": 9, "move_number": 5, "move_label": "5. Nxe5", "san": "Nxe5", "color": "white", "piece_moved": "knight", "from_square": "f3", "to_square": "e5", "is_capture": true, "captured_piece": "pawn", "is_check": false, "is_checkmate": false, "is_castle": false},
            {"ply": 10, "move_number": 5, "move_label": "5...Bxd1", "san": "Bxd1", "color": "black", "piece_moved": "bishop", "from_square": "g4", "to_square": "d1", "is_capture": true, "captured_piece": "queen", "is_check": false, "is_checkmate": false, "is_castle": false},
            {"ply": 11, "move_number": 6, "move_label": "6. Bxf7+", "san": "Bxf7+", "color": "white", "piece_moved": "bishop", "from_square": "c4", "to_square": "f7", "is_capture": true, "captured_piece": "pawn", "is_check": true, "is_checkmate": false, "is_castle": false},
            {"ply": 12, "move_number": 6, "move_label": "6...Ke7", "san": "Ke7", "color": "black", "piece_moved": "king", "from_square": "e8", "to_square": "e7", "is_capture": false, "captured_piece": null, "is_check": false, "is_checkmate": false, "is_castle": false},
            {"ply": 13, "move_number": 7, "move_label": "7. Nd5#", "san": "Nd5#", "color": "white", "piece_moved": "knight", "from_square": "c3", "to_square": "d5", "is_capture": false, "captured_piece": null, "is_check": true, "is_checkmate": true, "is_castle": false}
        ],
        "metadata": {
            "white_player": "maia5",
            "black_player": "n1000",
            "white_rating": 1400,
            "black_rating": 700,
            "opening": "Italian Game",
            "time_control": "300+3",
            "termination": "Normal",
            "winner": "white"
        }
    },
    "fight_story": {
        "winner": "White wins by checkmate",
        "opening": "Italian Game",
        "fight_style": "Blitz Aggression",
        "best_move": "5...Bxd1 — captures the queen",
        "turning_point": "5. Nxe5 — 5. Nxe5 swings the material balance decisively.",
        "battle_summary": "Out of the Italian Game, White presses the advantage across 7 moves, trading 3 blows before sealing the fight in a decisive final strike.",
        "prompt": "SCENE: Neutral arena, dramatic side lighting.\nFIGHTERS: White and Black, styles shaped by Italian Game.\nBEATS: 2 escalating exchanges building to the final blow.\nFINISH: White lands the decisive strike.\nSTYLE: Sharp linework, fast cuts on impact frames.",
        "estimated_length": "17 sec"
    },
    "video_placeholder": {
        "status": "not_generated",
        "message": "Video rendering is not implemented yet — this is analysis + narrative only.",
        "estimated_duration_seconds": 17
    },
    "game_metadata": {
        "white_player": "maia5",
        "black_player": "n1000",
        "white_rating": 1400,
        "black_rating": 700,
        "opening": "Italian Game",
        "time_control": "300+3",
        "termination": "Normal",
        "winner": "white"
    },
    "combat_intelligence": {
        "events": [
            {"event_type": "territorial_advance", "intensity": 2, "attacker": "white", "description": "1. e4 pushes forward, claiming ground.", "move_number": 1, "move_label": "1. e4"},
            {"event_type": "territorial_advance", "intensity": 2, "attacker": "black", "description": "1...e5 pushes forward, claiming ground.", "move_number": 1, "move_label": "1...e5"},
            {"event_type": "tactical_setup", "intensity": 2, "attacker": "white", "description": "2. Nf3 brings a fighter into position.", "move_number": 2, "move_label": "2. Nf3"},
            {"event_type": "territorial_advance", "intensity": 2, "attacker": "black", "description": "2...d6 pushes forward, claiming ground.", "move_number": 2, "move_label": "2...d6"},
            {"event_type": "tactical_setup", "intensity": 2, "attacker": "white", "description": "3. Bc4 brings a fighter into position.", "move_number": 3, "move_label": "3. Bc4"},
            {"event_type": "tactical_setup", "intensity": 2, "attacker": "black", "description": "3...Bg4 brings a fighter into position.", "move_number": 3, "move_label": "3...Bg4"},
            {"event_type": "tactical_setup", "intensity": 2, "attacker": "white", "description": "4. Nc3 brings a fighter into position.", "move_number": 4, "move_label": "4. Nc3"},
            {"event_type": "territorial_advance", "intensity": 2, "attacker": "black", "description": "4...g6 pushes forward, claiming ground.", "move_number": 4, "move_label": "4...g6"},
            {"event_type": "attack_landed", "intensity": 4, "attacker": "white", "description": "5. Nxe5 lands an attack, taking the pawn.", "move_number": 5, "move_label": "5. Nxe5"},
            {"event_type": "breakthrough_attack", "intensity": 7, "attacker": "black", "description": "5...Bxd1 lands a heavy blow, taking the queen.", "move_number": 5, "move_label": "5...Bxd1"},
            {"event_type": "breakthrough_attack", "intensity": 7, "attacker": "white", "description": "6. Bxf7+ breaks through enemy defenses with a surprise strike.", "move_number": 6, "move_label": "6. Bxf7+"},
            {"event_type": "strategic_positioning", "intensity": 2, "attacker": "black", "description": "6...Ke7 maneuvers for advantage.", "move_number": 6, "move_label": "6...Ke7"},
            {"event_type": "finishing_strike", "intensity": 10, "attacker": "white", "description": "7. Nd5# delivers the finishing blow — the fight is over.", "move_number": 7, "move_label": "7. Nd5#"}
        ],
        "profile": {
            "battle_pace": "fast",
            "fighter_balance": "veteran vs challenger",
            "ending_type": "checkmate",
            "winner": "white"
        }
    }
}
```

---

## Notes on what these two examples show

- **Opening detection differs by design**: the Chess.com sample has no
  `ECO`/`Opening` tag, so `opening` falls back to the heuristic book
  ("Italian Game (early queen sortie)"). The Lichess sample has an
  explicit `Opening` tag, which is used verbatim ("Italian Game").
- **`fighter_balance` responds to rating gap**: 1550 vs 1490 (60-point
  gap) reads as `"even"`; 1400 vs 700 (700-point gap) reads as
  `"veteran vs challenger"`.
- **`battle_pace` responds to time control**: `600+5` (600s base) reads
  as `"moderate"`; `300+3` (300s base) reads as `"fast"`.
- **5...Bxd1 is classified `breakthrough_attack`, not
  `calculated_sacrifice`**, even though giving up the queen for a
  bishop is the whole point of White's trap — this is exactly the
  known heuristic limitation flagged in the last review: sacrifice
  detection only looks one ply ahead for a same-square recapture, and
  nothing recaptures on d1, so the *real* sacrificial idea (5. Nxe5)
  isn't tagged as one either.
