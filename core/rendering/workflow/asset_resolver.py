"""Resolves which assets a workflow needs, from style and sequence data.

Every method here returns *semantic identifiers* — never file paths.
Turning a semantic identifier like "anime_diffusion_base" or
"pose_controlnet" into an actual file location is a concern for
whatever asset catalog/manifest sits below this layer (or a renderer
adapter), not for the Workflow Builder. This module's job is entirely
about *which* identifiers are needed and keeping that list
deduplicated and well-organized by category — never about *where*
those assets live on disk.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from core.rendering.workflow.workflow_templates import (
    ActorActionType,
    CinematicSequence,
    StyleProfile,
    WorkflowTemplate,
)

# Which animation clip identifier each action maps to. Owned here
# (rather than in the Animation Builder stage) because "what asset
# does this action need" is squarely an asset-resolution question —
# the Animation Builder stage calls `animation_clip_for_action` rather
# than keeping its own copy of this table.
_ACTION_TO_ANIMATION_CLIP: dict[ActorActionType, str] = {
    ActorActionType.ADVANCE: "clip_advance",
    ActorActionType.ATTACK: "clip_attack",
    ActorActionType.BLOCK: "clip_block",
    ActorActionType.ROLL: "clip_roll",
    ActorActionType.JUMP: "clip_jump",
    ActorActionType.COUNTER: "clip_counter",
    ActorActionType.SPECIAL: "clip_special",
    ActorActionType.DEATH: "clip_death",
    ActorActionType.FINISH: "clip_finish",
}


class ResolvedAssets(BaseModel):
    """Every asset identifier a workflow needs, organized by category
    and deduplicated within each category."""

    models: list[str] = Field(default_factory=list)
    loras: list[str] = Field(default_factory=list)
    vaes: list[str] = Field(default_factory=list)
    controlnets: list[str] = Field(default_factory=list)
    ip_adapters: list[str] = Field(default_factory=list)
    animations: list[str] = Field(default_factory=list)
    textures: list[str] = Field(default_factory=list)
    hdris: list[str] = Field(default_factory=list)
    weapons: list[str] = Field(default_factory=list)
    environment_assets: list[str] = Field(default_factory=list)


class AssetResolver:
    """Determines every asset a CinematicSequence + StyleProfile needs,
    given the style's WorkflowTemplate."""

    def resolve(
        self, sequence: CinematicSequence, style: StyleProfile, template: WorkflowTemplate
    ) -> ResolvedAssets:
        """Resolves the complete set of assets needed to render `sequence`
        in the given style.

        Args:
            sequence: The cinematic sequence being rendered.
            style: The style profile in effect.
            template: That style's WorkflowTemplate (for its default
                model/VAE).
        """
        return ResolvedAssets(
            models=self._resolve_models(sequence, template),
            loras=self._resolve_loras(sequence),
            vaes=[template.default_vae],
            controlnets=self._resolve_controlnets(sequence),
            ip_adapters=self._resolve_ip_adapters(sequence),
            animations=self._resolve_animations(sequence),
            textures=self._environment_derived(sequence, suffix="texture_set"),
            hdris=self._environment_derived(sequence, suffix="hdri"),
            weapons=self._resolve_weapons(sequence),
            environment_assets=[sequence.environment],
        )

    def animation_clip_for_action(self, action: ActorActionType) -> str:
        """Returns the animation clip identifier for a given action —
        the single source of truth the Animation Builder stage reads
        from, so this mapping is never duplicated elsewhere."""
        return _ACTION_TO_ANIMATION_CLIP[action]

    def _resolve_models(self, sequence: CinematicSequence, template: WorkflowTemplate) -> list[str]:
        models = [template.default_model]
        for actor in sequence.actors:
            if actor.character_model and actor.character_model != template.default_model:
                models.append(actor.character_model)
        return self._dedupe(models)

    def _resolve_loras(self, sequence: CinematicSequence) -> list[str]:
        return self._dedupe([actor.lora for actor in sequence.actors if actor.lora])

    def _resolve_controlnets(self, sequence: CinematicSequence) -> list[str]:
        # A pose reference implies pose-guided generation is needed.
        return self._dedupe(["pose_controlnet" for actor in sequence.actors if actor.pose_reference])

    def _resolve_ip_adapters(self, sequence: CinematicSequence) -> list[str]:
        # A facial preset implies identity/facial-consistency conditioning.
        return self._dedupe(["facial_ip_adapter" for actor in sequence.actors if actor.facial_preset])

    def _resolve_animations(self, sequence: CinematicSequence) -> list[str]:
        clips = [
            self.animation_clip_for_action(action.action)
            for beat in sequence.beats
            for action in beat.actions
        ]
        return self._dedupe(clips)

    def _resolve_weapons(self, sequence: CinematicSequence) -> list[str]:
        return self._dedupe([actor.weapon for actor in sequence.actors if actor.weapon])

    def _environment_derived(self, sequence: CinematicSequence, suffix: str) -> list[str]:
        """Derives an environment-specific asset identifier by naming
        convention (`{environment}_{suffix}`) — unlike a style's model
        or VAE, which come from explicit per-style configuration,
        environments are open-ended, so a documented naming convention
        is used instead of an exhaustive per-environment config table.
        """
        return [f"{sequence.environment}_{suffix}"]

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        """Deduplicates while preserving first-seen order, since node
        construction order can matter for readability/traceability."""
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
