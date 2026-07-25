"""
Halisako Cinematic Engine

Transforms combat intelligence into
Halisako Cinematic Events (HCE).

This is renderer-agnostic.
It does not generate video.

It creates the timeline blueprint.
"""

from __future__ import annotations

from products.chess2fight.schemas import CombatIntelligence
from products.chess2fight.schemas import StyleProfile
from products.cinema.schemas import (
    HCEEvent,
    CinematicSequence,
    CinematicActor,
    CameraDirection,
    CameraShot,
    CameraMovement,
    ActorAction,
    VisualEffect,
    AudioDirection,
)


class CinematicEngine:


    def generate(
        self,
        combat: CombatIntelligence,
        style: StyleProfile,
    ) -> CinematicSequence:


        events = []

        timestamp = 0


        for index, combat_event in enumerate(combat.events):

            action = self._map_action(
                combat_event.event_type.value
            )


            actor = CinematicActor(
                entity=self._piece_entity(
                    combat_event.move_label
                ),
                side=combat_event.attacker,
                role="fighter",
                action=action
            )


            event = HCEEvent(

                event_id=f"hce_{index+1}",

                timestamp_start=timestamp,

                duration_seconds=3,

                chess_move=combat_event.move_label,

                actors=[actor],

                camera=CameraDirection(
                    shot=self._camera(combat_event.intensity),
                    movement=self._movement(
                        combat_event.intensity
                    )
                ),

                effects=[
                    VisualEffect(
                        effect_type="energy_trail",
                        intensity=combat_event.intensity
                    )
                ],

                audio=AudioDirection(
                    impact="battle_clash",
                    atmosphere="war_ambience"
                ),

                narrative_intent=
                combat_event.description
            )


            events.append(event)

            timestamp += 3


        return CinematicSequence(

            title="Halisako Battle Sequence",

            total_duration_seconds=timestamp,

            events=events
        )


    def _map_action(self,event):

        if "finishing" in event:
            return ActorAction.FINISH

        if "attack" in event:
            return ActorAction.ATTACK

        if "defensive" in event:
            return ActorAction.DEFEND

        return ActorAction.ADVANCE



    def _camera(self,intensity):

        if intensity >= 9:
            return CameraShot.EXTREME_CLOSE

        if intensity >=6:
            return CameraShot.CLOSE_UP

        return CameraShot.WIDE



    def _movement(self,intensity):

        if intensity >=8:
            return CameraMovement.SHAKE

        return CameraMovement.TRACK



    def _piece_entity(self,move):

        if "N" in move:
            return "knight"

        if "B" in move:
            return "bishop"

        if "Q" in move:
            return "queen"

        if "R" in move:
            return "rook"

        return "pawn"