# Chess2Fight — Real PGN Analysis (post-upgrade)

Both games below are real, complete, and verified legal move-by-move
before submission (checked directly with python-chess, independent of
the API). These supersede the illustrative examples from the previous
message — this is what the upgraded backend actually returns for the
PGNs you provided.

---

## 1. winded_wayz vs SARDA_SUII (Chess.com) — 20 moves, real game

Request PGN:
```
[Event "winded_wayz vs. SARDA_SUII"]
[Site "Chess.com"]
[Date "2026-07-20"]
[White "winded_wayz"]
[Black "SARDA_SUII"]
[Result "1-0"]
[WhiteElo "557"]
[BlackElo "557"]
[TimeControl "300"]
[Termination "winded_wayz won by checkmate"]

1. e4 e5 2. Nf3 Nf6 3. Nxe5 Nxe4 4. Qe2 f5 5. g4 Bd6 6. gxf5 Bxe5 7. Qxe4 d6
8. d4 O-O 9. dxe5 Bxf5 10. Qd5+ Kh8 11. exd6 cxd6 12. Qxb7 Qe8+ 13. Be2 Be4
14. Rg1 Bxb7 15. Nc3 Nc6 16. Be3 g6 17. O-O-O Nb4 18. Bc4 Rc8 19. Bd4+ Rf6
20. Bxf6# 1-0
```

Full response (39 moves/plies, 39 combat events):

```json
{
    "status": "completed",
    "game_analysis": {
        "white_player": "winded_wayz",
        "black_player": "SARDA_SUII",
        "opening": "Open Game (1.e4)",
        "num_moves": 20,
        "winner": "white",
        "is_checkmate": true,
        "checkmate_move_number": 20,
        "captures": [
            {
                "move_number": 3,
                "move_label": "3. Nxe5",
                "san": "Nxe5",
                "capturing_piece": "knight",
                "captured_piece": "pawn"
            },
            {
                "move_number": 3,
                "move_label": "3...Nxe4",
                "san": "Nxe4",
                "capturing_piece": "knight",
                "captured_piece": "pawn"
            },
            {
                "move_number": 6,
                "move_label": "6. gxf5",
                "san": "gxf5",
                "capturing_piece": "pawn",
                "captured_piece": "pawn"
            },
            {
                "move_number": 6,
                "move_label": "6...Bxe5",
                "san": "Bxe5",
                "capturing_piece": "bishop",
                "captured_piece": "knight"
            },
            {
                "move_number": 7,
                "move_label": "7. Qxe4",
                "san": "Qxe4",
                "capturing_piece": "queen",
                "captured_piece": "knight"
            },
            {
                "move_number": 9,
                "move_label": "9. dxe5",
                "san": "dxe5",
                "capturing_piece": "pawn",
                "captured_piece": "bishop"
            },
            {
                "move_number": 9,
                "move_label": "9...Bxf5",
                "san": "Bxf5",
                "capturing_piece": "bishop",
                "captured_piece": "pawn"
            },
            {
                "move_number": 11,
                "move_label": "11. exd6",
                "san": "exd6",
                "capturing_piece": "pawn",
                "captured_piece": "pawn"
            },
            {
                "move_number": 11,
                "move_label": "11...cxd6",
                "san": "cxd6",
                "capturing_piece": "pawn",
                "captured_piece": "pawn"
            },
            {
                "move_number": 12,
                "move_label": "12. Qxb7",
                "san": "Qxb7",
                "capturing_piece": "queen",
                "captured_piece": "pawn"
            },
            {
                "move_number": 14,
                "move_label": "14...Bxb7",
                "san": "Bxb7",
                "capturing_piece": "bishop",
                "captured_piece": "queen"
            },
            {
                "move_number": 20,
                "move_label": "20. Bxf6#",
                "san": "Bxf6#",
                "capturing_piece": "bishop",
                "captured_piece": "rook"
            }
        ],
        "tactical_moments": [
            {
                "move_number": 6,
                "move_label": "6...Bxe5",
                "san": "Bxe5",
                "description": "Captures the knight"
            },
            {
                "move_number": 7,
                "move_label": "7. Qxe4",
                "san": "Qxe4",
                "description": "Captures the knight"
            },
            {
                "move_number": 9,
                "move_label": "9. dxe5",
                "san": "dxe5",
                "description": "Captures the bishop"
            },
            {
                "move_number": 10,
                "move_label": "10. Qd5+",
                "san": "Qd5+",
                "description": "Delivers check"
            },
            {
                "move_number": 12,
                "move_label": "12...Qe8+",
                "san": "Qe8+",
                "description": "Delivers check"
            },
            {
                "move_number": 14,
                "move_label": "14...Bxb7",
                "san": "Bxb7",
                "description": "Captures the queen"
            },
            {
                "move_number": 19,
                "move_label": "19. Bd4+",
                "san": "Bd4+",
                "description": "Delivers check"
            },
            {
                "move_number": 20,
                "move_label": "20. Bxf6#",
                "san": "Bxf6#",
                "description": "Delivers checkmate"
            }
        ],
        "turning_points": [
            {
                "move_number": 20,
                "move_label": "20. Bxf6#",
                "san": "Bxf6#",
                "description": "20. Bxf6# swings the material balance decisively."
            }
        ],
        "moves": [
            {
                "ply": 1,
                "move_number": 1,
                "move_label": "1. e4",
                "san": "e4",
                "color": "white",
                "piece_moved": "pawn",
                "from_square": "e2",
                "to_square": "e4",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 2,
                "move_number": 1,
                "move_label": "1...e5",
                "san": "e5",
                "color": "black",
                "piece_moved": "pawn",
                "from_square": "e7",
                "to_square": "e5",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 3,
                "move_number": 2,
                "move_label": "2. Nf3",
                "san": "Nf3",
                "color": "white",
                "piece_moved": "knight",
                "from_square": "g1",
                "to_square": "f3",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 4,
                "move_number": 2,
                "move_label": "2...Nf6",
                "san": "Nf6",
                "color": "black",
                "piece_moved": "knight",
                "from_square": "g8",
                "to_square": "f6",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 5,
                "move_number": 3,
                "move_label": "3. Nxe5",
                "san": "Nxe5",
                "color": "white",
                "piece_moved": "knight",
                "from_square": "f3",
                "to_square": "e5",
                "is_capture": true,
                "captured_piece": "pawn",
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 6,
                "move_number": 3,
                "move_label": "3...Nxe4",
                "san": "Nxe4",
                "color": "black",
                "piece_moved": "knight",
                "from_square": "f6",
                "to_square": "e4",
                "is_capture": true,
                "captured_piece": "pawn",
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 7,
                "move_number": 4,
                "move_label": "4. Qe2",
                "san": "Qe2",
                "color": "white",
                "piece_moved": "queen",
                "from_square": "d1",
                "to_square": "e2",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 8,
                "move_number": 4,
                "move_label": "4...f5",
                "san": "f5",
                "color": "black",
                "piece_moved": "pawn",
                "from_square": "f7",
                "to_square": "f5",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 9,
                "move_number": 5,
                "move_label": "5. g4",
                "san": "g4",
                "color": "white",
                "piece_moved": "pawn",
                "from_square": "g2",
                "to_square": "g4",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 10,
                "move_number": 5,
                "move_label": "5...Bd6",
                "san": "Bd6",
                "color": "black",
                "piece_moved": "bishop",
                "from_square": "f8",
                "to_square": "d6",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 11,
                "move_number": 6,
                "move_label": "6. gxf5",
                "san": "gxf5",
                "color": "white",
                "piece_moved": "pawn",
                "from_square": "g4",
                "to_square": "f5",
                "is_capture": true,
                "captured_piece": "pawn",
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 12,
                "move_number": 6,
                "move_label": "6...Bxe5",
                "san": "Bxe5",
                "color": "black",
                "piece_moved": "bishop",
                "from_square": "d6",
                "to_square": "e5",
                "is_capture": true,
                "captured_piece": "knight",
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 13,
                "move_number": 7,
                "move_label": "7. Qxe4",
                "san": "Qxe4",
                "color": "white",
                "piece_moved": "queen",
                "from_square": "e2",
                "to_square": "e4",
                "is_capture": true,
                "captured_piece": "knight",
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 14,
                "move_number": 7,
                "move_label": "7...d6",
                "san": "d6",
                "color": "black",
                "piece_moved": "pawn",
                "from_square": "d7",
                "to_square": "d6",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 15,
                "move_number": 8,
                "move_label": "8. d4",
                "san": "d4",
                "color": "white",
                "piece_moved": "pawn",
                "from_square": "d2",
                "to_square": "d4",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 16,
                "move_number": 8,
                "move_label": "8...O-O",
                "san": "O-O",
                "color": "black",
                "piece_moved": "king",
                "from_square": "e8",
                "to_square": "g8",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": true
            },
            {
                "ply": 17,
                "move_number": 9,
                "move_label": "9. dxe5",
                "san": "dxe5",
                "color": "white",
                "piece_moved": "pawn",
                "from_square": "d4",
                "to_square": "e5",
                "is_capture": true,
                "captured_piece": "bishop",
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 18,
                "move_number": 9,
                "move_label": "9...Bxf5",
                "san": "Bxf5",
                "color": "black",
                "piece_moved": "bishop",
                "from_square": "c8",
                "to_square": "f5",
                "is_capture": true,
                "captured_piece": "pawn",
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 19,
                "move_number": 10,
                "move_label": "10. Qd5+",
                "san": "Qd5+",
                "color": "white",
                "piece_moved": "queen",
                "from_square": "e4",
                "to_square": "d5",
                "is_capture": false,
                "captured_piece": null,
                "is_check": true,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 20,
                "move_number": 10,
                "move_label": "10...Kh8",
                "san": "Kh8",
                "color": "black",
                "piece_moved": "king",
                "from_square": "g8",
                "to_square": "h8",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 21,
                "move_number": 11,
                "move_label": "11. exd6",
                "san": "exd6",
                "color": "white",
                "piece_moved": "pawn",
                "from_square": "e5",
                "to_square": "d6",
                "is_capture": true,
                "captured_piece": "pawn",
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 22,
                "move_number": 11,
                "move_label": "11...cxd6",
                "san": "cxd6",
                "color": "black",
                "piece_moved": "pawn",
                "from_square": "c7",
                "to_square": "d6",
                "is_capture": true,
                "captured_piece": "pawn",
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 23,
                "move_number": 12,
                "move_label": "12. Qxb7",
                "san": "Qxb7",
                "color": "white",
                "piece_moved": "queen",
                "from_square": "d5",
                "to_square": "b7",
                "is_capture": true,
                "captured_piece": "pawn",
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 24,
                "move_number": 12,
                "move_label": "12...Qe8+",
                "san": "Qe8+",
                "color": "black",
                "piece_moved": "queen",
                "from_square": "d8",
                "to_square": "e8",
                "is_capture": false,
                "captured_piece": null,
                "is_check": true,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 25,
                "move_number": 13,
                "move_label": "13. Be2",
                "san": "Be2",
                "color": "white",
                "piece_moved": "bishop",
                "from_square": "f1",
                "to_square": "e2",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 26,
                "move_number": 13,
                "move_label": "13...Be4",
                "san": "Be4",
                "color": "black",
                "piece_moved": "bishop",
                "from_square": "f5",
                "to_square": "e4",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 27,
                "move_number": 14,
                "move_label": "14. Rg1",
                "san": "Rg1",
                "color": "white",
                "piece_moved": "rook",
                "from_square": "h1",
                "to_square": "g1",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 28,
                "move_number": 14,
                "move_label": "14...Bxb7",
                "san": "Bxb7",
                "color": "black",
                "piece_moved": "bishop",
                "from_square": "e4",
                "to_square": "b7",
                "is_capture": true,
                "captured_piece": "queen",
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 29,
                "move_number": 15,
                "move_label": "15. Nc3",
                "san": "Nc3",
                "color": "white",
                "piece_moved": "knight",
                "from_square": "b1",
                "to_square": "c3",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 30,
                "move_number": 15,
                "move_label": "15...Nc6",
                "san": "Nc6",
                "color": "black",
                "piece_moved": "knight",
                "from_square": "b8",
                "to_square": "c6",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 31,
                "move_number": 16,
                "move_label": "16. Be3",
                "san": "Be3",
                "color": "white",
                "piece_moved": "bishop",
                "from_square": "c1",
                "to_square": "e3",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 32,
                "move_number": 16,
                "move_label": "16...g6",
                "san": "g6",
                "color": "black",
                "piece_moved": "pawn",
                "from_square": "g7",
                "to_square": "g6",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 33,
                "move_number": 17,
                "move_label": "17. O-O-O",
                "san": "O-O-O",
                "color": "white",
                "piece_moved": "king",
                "from_square": "e1",
                "to_square": "c1",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": true
            },
            {
                "ply": 34,
                "move_number": 17,
                "move_label": "17...Nb4",
                "san": "Nb4",
                "color": "black",
                "piece_moved": "knight",
                "from_square": "c6",
                "to_square": "b4",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 35,
                "move_number": 18,
                "move_label": "18. Bc4",
                "san": "Bc4",
                "color": "white",
                "piece_moved": "bishop",
                "from_square": "e2",
                "to_square": "c4",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 36,
                "move_number": 18,
                "move_label": "18...Rc8",
                "san": "Rc8",
                "color": "black",
                "piece_moved": "rook",
                "from_square": "a8",
                "to_square": "c8",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 37,
                "move_number": 19,
                "move_label": "19. Bd4+",
                "san": "Bd4+",
                "color": "white",
                "piece_moved": "bishop",
                "from_square": "e3",
                "to_square": "d4",
                "is_capture": false,
                "captured_piece": null,
                "is_check": true,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 38,
                "move_number": 19,
                "move_label": "19...Rf6",
                "san": "Rf6",
                "color": "black",
                "piece_moved": "rook",
                "from_square": "f8",
                "to_square": "f6",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 39,
                "move_number": 20,
                "move_label": "20. Bxf6#",
                "san": "Bxf6#",
                "color": "white",
                "piece_moved": "bishop",
                "from_square": "d4",
                "to_square": "f6",
                "is_capture": true,
                "captured_piece": "rook",
                "is_check": true,
                "is_checkmate": true,
                "is_castle": false
            }
        ],
        "metadata": {
            "white_player": "winded_wayz",
            "black_player": "SARDA_SUII",
            "white_rating": 557,
            "black_rating": 557,
            "opening": "Open Game (1.e4)",
            "time_control": "300",
            "termination": "winded_wayz won by checkmate",
            "winner": "white"
        }
    },
    "fight_story": {
        "winner": "White wins by checkmate",
        "opening": "Open Game (1.e4)",
        "fight_style": "Relentless Exchange",
        "best_move": "14...Bxb7 \u2014 captures the queen",
        "turning_point": "20. Bxf6# \u2014 20. Bxf6# swings the material balance decisively.",
        "battle_summary": "Out of the Open Game (1.e4), White presses the advantage across 20 moves, trading 12 blows before sealing the fight in a decisive final strike.",
        "prompt": "SCENE: Neutral arena, dramatic side lighting.\nFIGHTERS: White and Black, styles shaped by Open Game (1.e4).\nBEATS: 6 escalating exchanges building to the final blow.\nFINISH: White lands the decisive strike.\nSTYLE: Sharp linework, fast cuts on impact frames.",
        "estimated_length": "30 sec"
    },
    "video_placeholder": {
        "status": "not_generated",
        "message": "Video rendering is not implemented yet \u2014 this is analysis + narrative only.",
        "estimated_duration_seconds": 30
    },
    "game_metadata": {
        "white_player": "winded_wayz",
        "black_player": "SARDA_SUII",
        "white_rating": 557,
        "black_rating": 557,
        "opening": "Open Game (1.e4)",
        "time_control": "300",
        "termination": "winded_wayz won by checkmate",
        "winner": "white"
    },
    "combat_intelligence": {
        "events": [
            {
                "event_type": "territorial_advance",
                "intensity": 2,
                "attacker": "white",
                "description": "1. e4 pushes forward, claiming ground.",
                "move_number": 1,
                "move_label": "1. e4"
            },
            {
                "event_type": "territorial_advance",
                "intensity": 2,
                "attacker": "black",
                "description": "1...e5 pushes forward, claiming ground.",
                "move_number": 1,
                "move_label": "1...e5"
            },
            {
                "event_type": "tactical_setup",
                "intensity": 2,
                "attacker": "white",
                "description": "2. Nf3 brings a fighter into position.",
                "move_number": 2,
                "move_label": "2. Nf3"
            },
            {
                "event_type": "tactical_setup",
                "intensity": 2,
                "attacker": "black",
                "description": "2...Nf6 brings a fighter into position.",
                "move_number": 2,
                "move_label": "2...Nf6"
            },
            {
                "event_type": "attack_landed",
                "intensity": 4,
                "attacker": "white",
                "description": "3. Nxe5 lands an attack, taking the pawn.",
                "move_number": 3,
                "move_label": "3. Nxe5"
            },
            {
                "event_type": "attack_landed",
                "intensity": 4,
                "attacker": "black",
                "description": "3...Nxe4 lands an attack, taking the pawn.",
                "move_number": 3,
                "move_label": "3...Nxe4"
            },
            {
                "event_type": "power_deployment",
                "intensity": 3,
                "attacker": "white",
                "description": "4. Qe2 commits their strongest fighter to the field.",
                "move_number": 4,
                "move_label": "4. Qe2"
            },
            {
                "event_type": "territorial_advance",
                "intensity": 2,
                "attacker": "black",
                "description": "4...f5 pushes forward, claiming ground.",
                "move_number": 4,
                "move_label": "4...f5"
            },
            {
                "event_type": "territorial_advance",
                "intensity": 2,
                "attacker": "white",
                "description": "5. g4 pushes forward, claiming ground.",
                "move_number": 5,
                "move_label": "5. g4"
            },
            {
                "event_type": "tactical_setup",
                "intensity": 2,
                "attacker": "black",
                "description": "5...Bd6 brings a fighter into position.",
                "move_number": 5,
                "move_label": "5...Bd6"
            },
            {
                "event_type": "attack_landed",
                "intensity": 4,
                "attacker": "white",
                "description": "6. gxf5 lands an attack, taking the pawn.",
                "move_number": 6,
                "move_label": "6. gxf5"
            },
            {
                "event_type": "attack_landed",
                "intensity": 5,
                "attacker": "black",
                "description": "6...Bxe5 lands an attack, taking the knight.",
                "move_number": 6,
                "move_label": "6...Bxe5"
            },
            {
                "event_type": "attack_landed",
                "intensity": 5,
                "attacker": "white",
                "description": "7. Qxe4 lands an attack, taking the knight.",
                "move_number": 7,
                "move_label": "7. Qxe4"
            },
            {
                "event_type": "territorial_advance",
                "intensity": 2,
                "attacker": "black",
                "description": "7...d6 pushes forward, claiming ground.",
                "move_number": 7,
                "move_label": "7...d6"
            },
            {
                "event_type": "territorial_advance",
                "intensity": 2,
                "attacker": "white",
                "description": "8. d4 pushes forward, claiming ground.",
                "move_number": 8,
                "move_label": "8. d4"
            },
            {
                "event_type": "defensive_repositioning",
                "intensity": 3,
                "attacker": "black",
                "description": "8...O-O pulls back into a fortified stance.",
                "move_number": 8,
                "move_label": "8...O-O"
            },
            {
                "event_type": "attack_landed",
                "intensity": 5,
                "attacker": "white",
                "description": "9. dxe5 lands an attack, taking the bishop.",
                "move_number": 9,
                "move_label": "9. dxe5"
            },
            {
                "event_type": "attack_landed",
                "intensity": 4,
                "attacker": "black",
                "description": "9...Bxf5 lands an attack, taking the pawn.",
                "move_number": 9,
                "move_label": "9...Bxf5"
            },
            {
                "event_type": "critical_threat",
                "intensity": 6,
                "attacker": "white",
                "description": "10. Qd5+ presses forward with a direct threat.",
                "move_number": 10,
                "move_label": "10. Qd5+"
            },
            {
                "event_type": "strategic_positioning",
                "intensity": 2,
                "attacker": "black",
                "description": "10...Kh8 maneuvers for advantage.",
                "move_number": 10,
                "move_label": "10...Kh8"
            },
            {
                "event_type": "coordinated_assault",
                "intensity": 6,
                "attacker": "white",
                "description": "11. exd6 joins a rapid exchange of blows.",
                "move_number": 11,
                "move_label": "11. exd6"
            },
            {
                "event_type": "coordinated_assault",
                "intensity": 6,
                "attacker": "black",
                "description": "11...cxd6 joins a rapid exchange of blows.",
                "move_number": 11,
                "move_label": "11...cxd6"
            },
            {
                "event_type": "attack_landed",
                "intensity": 4,
                "attacker": "white",
                "description": "12. Qxb7 lands an attack, taking the pawn.",
                "move_number": 12,
                "move_label": "12. Qxb7"
            },
            {
                "event_type": "critical_threat",
                "intensity": 6,
                "attacker": "black",
                "description": "12...Qe8+ presses forward with a direct threat.",
                "move_number": 12,
                "move_label": "12...Qe8+"
            },
            {
                "event_type": "strategic_positioning",
                "intensity": 2,
                "attacker": "white",
                "description": "13. Be2 maneuvers for advantage.",
                "move_number": 13,
                "move_label": "13. Be2"
            },
            {
                "event_type": "strategic_positioning",
                "intensity": 2,
                "attacker": "black",
                "description": "13...Be4 maneuvers for advantage.",
                "move_number": 13,
                "move_label": "13...Be4"
            },
            {
                "event_type": "strategic_positioning",
                "intensity": 2,
                "attacker": "white",
                "description": "14. Rg1 maneuvers for advantage.",
                "move_number": 14,
                "move_label": "14. Rg1"
            },
            {
                "event_type": "breakthrough_attack",
                "intensity": 7,
                "attacker": "black",
                "description": "14...Bxb7 lands a heavy blow, taking the queen.",
                "move_number": 14,
                "move_label": "14...Bxb7"
            },
            {
                "event_type": "strategic_positioning",
                "intensity": 2,
                "attacker": "white",
                "description": "15. Nc3 maneuvers for advantage.",
                "move_number": 15,
                "move_label": "15. Nc3"
            },
            {
                "event_type": "strategic_positioning",
                "intensity": 2,
                "attacker": "black",
                "description": "15...Nc6 maneuvers for advantage.",
                "move_number": 15,
                "move_label": "15...Nc6"
            },
            {
                "event_type": "strategic_positioning",
                "intensity": 2,
                "attacker": "white",
                "description": "16. Be3 maneuvers for advantage.",
                "move_number": 16,
                "move_label": "16. Be3"
            },
            {
                "event_type": "territorial_advance",
                "intensity": 2,
                "attacker": "black",
                "description": "16...g6 pushes forward, claiming ground.",
                "move_number": 16,
                "move_label": "16...g6"
            },
            {
                "event_type": "defensive_repositioning",
                "intensity": 3,
                "attacker": "white",
                "description": "17. O-O-O pulls back into a fortified stance.",
                "move_number": 17,
                "move_label": "17. O-O-O"
            },
            {
                "event_type": "strategic_positioning",
                "intensity": 2,
                "attacker": "black",
                "description": "17...Nb4 maneuvers for advantage.",
                "move_number": 17,
                "move_label": "17...Nb4"
            },
            {
                "event_type": "strategic_positioning",
                "intensity": 2,
                "attacker": "white",
                "description": "18. Bc4 maneuvers for advantage.",
                "move_number": 18,
                "move_label": "18. Bc4"
            },
            {
                "event_type": "strategic_positioning",
                "intensity": 2,
                "attacker": "black",
                "description": "18...Rc8 maneuvers for advantage.",
                "move_number": 18,
                "move_label": "18...Rc8"
            },
            {
                "event_type": "critical_threat",
                "intensity": 6,
                "attacker": "white",
                "description": "19. Bd4+ presses forward with a direct threat.",
                "move_number": 19,
                "move_label": "19. Bd4+"
            },
            {
                "event_type": "calculated_sacrifice",
                "intensity": 10,
                "attacker": "black",
                "description": "19...Rf6 gives up material on purpose, trading position for a future strike.",
                "move_number": 19,
                "move_label": "19...Rf6"
            },
            {
                "event_type": "finishing_strike",
                "intensity": 10,
                "attacker": "white",
                "description": "20. Bxf6# delivers the finishing blow \u2014 the fight is over.",
                "move_number": 20,
                "move_label": "20. Bxf6#"
            }
        ],
        "profile": {
            "battle_pace": "fast",
            "fighter_balance": "even",
            "ending_type": "checkmate",
            "winner": "white"
        }
    }
}
```

---

## 2. maia5 vs n1000 (Lichess) — 4 moves, real game

Request PGN:
```
[Event "Rated Blitz game"]
[Site "https://lichess.org/abcabcab"]
[Date "2024.04.01"]
[Round "-"]
[White "maia5"]
[Black "n1000"]
[Result "1-0"]
[WhiteElo "1400"]
[BlackElo "700"]
[ECO "A00"]
[Opening "Van't Kruijs Opening"]
[TimeControl "300+3"]
[UTCDate "2024.04.01"]
[UTCTime "12:34:56"]
[Termination "Normal"]
[WhiteRatingDiff "+7"]
[BlackRatingDiff "-4"]

1. e3 e5 2. Bc4 h6 3. Qh5 Nf6 4. Qxf7# 1-0
```

Full response:

```json
{
    "status": "completed",
    "game_analysis": {
        "white_player": "maia5",
        "black_player": "n1000",
        "opening": "Van't Kruijs Opening",
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
                "ply": 1,
                "move_number": 1,
                "move_label": "1. e3",
                "san": "e3",
                "color": "white",
                "piece_moved": "pawn",
                "from_square": "e2",
                "to_square": "e3",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 2,
                "move_number": 1,
                "move_label": "1...e5",
                "san": "e5",
                "color": "black",
                "piece_moved": "pawn",
                "from_square": "e7",
                "to_square": "e5",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 3,
                "move_number": 2,
                "move_label": "2. Bc4",
                "san": "Bc4",
                "color": "white",
                "piece_moved": "bishop",
                "from_square": "f1",
                "to_square": "c4",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 4,
                "move_number": 2,
                "move_label": "2...h6",
                "san": "h6",
                "color": "black",
                "piece_moved": "pawn",
                "from_square": "h7",
                "to_square": "h6",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 5,
                "move_number": 3,
                "move_label": "3. Qh5",
                "san": "Qh5",
                "color": "white",
                "piece_moved": "queen",
                "from_square": "d1",
                "to_square": "h5",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 6,
                "move_number": 3,
                "move_label": "3...Nf6",
                "san": "Nf6",
                "color": "black",
                "piece_moved": "knight",
                "from_square": "g8",
                "to_square": "f6",
                "is_capture": false,
                "captured_piece": null,
                "is_check": false,
                "is_checkmate": false,
                "is_castle": false
            },
            {
                "ply": 7,
                "move_number": 4,
                "move_label": "4. Qxf7#",
                "san": "Qxf7#",
                "color": "white",
                "piece_moved": "queen",
                "from_square": "h5",
                "to_square": "f7",
                "is_capture": true,
                "captured_piece": "pawn",
                "is_check": true,
                "is_checkmate": true,
                "is_castle": false
            }
        ],
        "metadata": {
            "white_player": "maia5",
            "black_player": "n1000",
            "white_rating": 1400,
            "black_rating": 700,
            "opening": "Van't Kruijs Opening",
            "time_control": "300+3",
            "termination": "Normal",
            "winner": "white"
        }
    },
    "fight_story": {
        "winner": "White wins by checkmate",
        "opening": "Van't Kruijs Opening",
        "fight_style": "Blitz Aggression",
        "best_move": "4. Qxf7# \u2014 delivers checkmate",
        "turning_point": "4. Qxf7# \u2014 4. Qxf7# swings the material balance decisively.",
        "battle_summary": "Out of the Van't Kruijs Opening, White presses the advantage across 4 moves, trading 1 blow before sealing the fight in a decisive final strike.",
        "prompt": "SCENE: Neutral arena, dramatic side lighting.\nFIGHTERS: White and Black, styles shaped by Van't Kruijs Opening.\nBEATS: 1 escalating exchange building to the final blow.\nFINISH: White lands the decisive strike.\nSTYLE: Sharp linework, fast cuts on impact frames.",
        "estimated_length": "12 sec"
    },
    "video_placeholder": {
        "status": "not_generated",
        "message": "Video rendering is not implemented yet \u2014 this is analysis + narrative only.",
        "estimated_duration_seconds": 12
    },
    "game_metadata": {
        "white_player": "maia5",
        "black_player": "n1000",
        "white_rating": 1400,
        "black_rating": 700,
        "opening": "Van't Kruijs Opening",
        "time_control": "300+3",
        "termination": "Normal",
        "winner": "white"
    },
    "combat_intelligence": {
        "events": [
            {
                "event_type": "territorial_advance",
                "intensity": 2,
                "attacker": "white",
                "description": "1. e3 pushes forward, claiming ground.",
                "move_number": 1,
                "move_label": "1. e3"
            },
            {
                "event_type": "territorial_advance",
                "intensity": 2,
                "attacker": "black",
                "description": "1...e5 pushes forward, claiming ground.",
                "move_number": 1,
                "move_label": "1...e5"
            },
            {
                "event_type": "tactical_setup",
                "intensity": 2,
                "attacker": "white",
                "description": "2. Bc4 brings a fighter into position.",
                "move_number": 2,
                "move_label": "2. Bc4"
            },
            {
                "event_type": "territorial_advance",
                "intensity": 2,
                "attacker": "black",
                "description": "2...h6 pushes forward, claiming ground.",
                "move_number": 2,
                "move_label": "2...h6"
            },
            {
                "event_type": "power_deployment",
                "intensity": 3,
                "attacker": "white",
                "description": "3. Qh5 commits their strongest fighter to the field.",
                "move_number": 3,
                "move_label": "3. Qh5"
            },
            {
                "event_type": "tactical_setup",
                "intensity": 2,
                "attacker": "black",
                "description": "3...Nf6 brings a fighter into position.",
                "move_number": 3,
                "move_label": "3...Nf6"
            },
            {
                "event_type": "finishing_strike",
                "intensity": 10,
                "attacker": "white",
                "description": "4. Qxf7# delivers the finishing blow \u2014 the fight is over.",
                "move_number": 4,
                "move_label": "4. Qxf7#"
            }
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

## What this run surfaced

- **Both castling moves detected correctly from real notation**:
  `8...O-O` (Black, kingside) and `17. O-O-O` (White, queenside) both
  show `"is_castle": true` and classify as `defensive_repositioning`.
  This specific path — real PGN castling text through
  `board.is_castling()` — hadn't been exercised by a real PGN in
  testing before (the unit tests set `is_castle=True` directly on
  hand-built fixtures); this PGN is the first real confirmation it
  works end-to-end.
- **`TimeControl "300"` (no increment) parsed correctly** — the format
  without a `+` was handled by the existing fallback branch in
  `_infer_battle_pace`, resulting in `"fast"` (300s base).
- **The verbose Chess.com termination string
  `"winded_wayz won by checkmate"` passes through as-is** in
  `game_metadata.termination` — it doesn't match `"Normal"` or any
  known enum, which is fine, since `ending_type` is derived from
  `is_checkmate` first and only falls back to parsing `termination`
  text when checkmate isn't detected.
- **A more precise example of the known sacrifice-heuristic
  limitation**: move `19...Rf6` is tagged `calculated_sacrifice`
  (intensity 10). I checked the actual position — after `19.Bd4+`,
  Black's *only* two legal moves were `Rf6` or `Qe5` (no king move was
  available at all). So this wasn't a "calculated" sacrifice in the
  sense of a chosen tactical risk; it was a forced choice between
  losing the rook or putting the queen on a square that may also have
  been unsafe. The heuristic can't currently distinguish "voluntarily
  offered material" from "the least-bad of two forced, already-losing
  options" — both look identical to a one-ply-lookahead check. Worth
  keeping in mind if this label is ever surfaced to users as implying
  deliberate strategy.
- **`Van't Kruijs Opening` (with the apostrophe) round-trips cleanly**
  through the header, the schema, and JSON serialization — no escaping
  issues.
