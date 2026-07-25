"""
Halisako Cinematic Event (HCE)

The universal cinematic representation between:
Chess Intelligence
and
Rendering Engines.

No renderer logic exists here.

This layer describes:
- what happens
- who acts
- camera behaviour
- visual effects
- audio direction
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel
from typing import Optional


class CameraShot(str, Enum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE = "extreme_close"


class CameraMovement(str, Enum):
    STATIC = "static"
    PUSH_IN = "push_in"
    PAN = "pan"
    TRACK = "tracking"
    SHAKE = "impact_shake"


class ActorAction(str, Enum):
    ADVANCE = "advance"
    ATTACK = "attack"
    DEFEND = "defend"
    RETREAT = "retreat"
    DESTROY = "destroy"
    FINISH = "finish"


class CinematicActor(BaseModel):
    entity: str
    side: str
    role: str
    action: ActorAction


class CameraDirection(BaseModel):
    shot: CameraShot
    movement: CameraMovement
    description: Optional[str] = None


class VisualEffect(BaseModel):
    effect_type: str
    intensity: int


class AudioDirection(BaseModel):
    impact: Optional[str] = None
    atmosphere: Optional[str] = None


class HCEEvent(BaseModel):
    """
    Single cinematic moment.

    Example:
    Knight charges.
    Camera follows.
    Explosion occurs.
    """

    event_id: str

    timestamp_start: float
    duration_seconds: float

    chess_move: str

    actors: list[CinematicActor]

    camera: CameraDirection

    effects: list[VisualEffect] = []

    audio: AudioDirection

    narrative_intent: str


class CinematicSequence(BaseModel):
    """
    Complete cinematic timeline.
    """

    title: str

    total_duration_seconds: float

    events: list[HCEEvent]