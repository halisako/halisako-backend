"""The Workflow Builder: converts a CinematicSequence into a renderer
workflow.

This is the "you are here" module of the rendering pipeline. Everything
upstream (chess/combat/battle intelligence, the style engine, the
cinematic engine) has already decided the story: who the actors are,
what they do, how the camera moves, what effects fire, and what the
audience should feel. This module never invents any of that — it only
converts those already-made decisions into an executable, renderer-
agnostic graph. See `workflow_templates.py`'s module docstring for the
important caveat about where that input contract (`CinematicSequence`
and its parts) comes from in this implementation.

`WorkflowBuilder.build()` runs the sequence through eight named stages
(Scene, Character, Animation, Camera, Lighting, Effects, Audio, Export)
plus a Timeline stage (specified separately, in more detail, by the
brief's own "Timeline Builder" section) — each its own method, never
folded into one large function. No stage constructs a node dict by
hand; every node comes from `NodeBuilder`. No stage decides anything
about story, only about presentation — camera *type* (wide, close,
tracking, ...) was already decided upstream; this module only maps
that decision into concrete render parameters via `ParameterMapper`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.rendering.workflow.asset_resolver import AssetResolver, ResolvedAssets
from core.rendering.workflow.exceptions import WorkflowValidationError
from core.rendering.workflow.node_builder import NodeBuilder, WorkflowNode
from core.rendering.workflow.parameter_mapper import ParameterMapper
from core.rendering.workflow.validators import validate_workflow
from core.rendering.workflow.workflow_registry import WorkflowRegistry
from core.rendering.workflow.workflow_templates import (
    CinematicActor,
    CinematicBeat,
    CinematicSequence,
    StyleProfile,
    WorkflowTemplate,
    get_template,
)

# Rough, documented estimation constants — pending calibration against
# a real renderer, which does not exist yet for this module to measure
# against. See `_estimate_gpu_seconds` / `_estimate_vram_gb`.
_BASE_SECONDS_PER_FRAME_AT_25_STEPS = 0.12
_BASE_CHECKPOINT_VRAM_GB = 6.0
_VRAM_PER_LORA_GB = 0.5
_VRAM_RESOLUTION_BASELINE_PIXELS = 1024 * 576


class RendererWorkflow(BaseModel):
    """The Workflow Builder's output: a complete, renderer-agnostic
    workflow graph plus everything a caller needs to know about it
    before handing it to a renderer adapter.
    """

    renderer: str = Field(..., description="Which registered renderer target this workflow was built for.")
    version: str = Field(
        ..., description="Workflow schema version, for the adapter layer to check compatibility."
    )
    workflow: dict[str, Any] = Field(
        ..., description='The renderer-agnostic graph: {"nodes": [...]} — see NodeBuilder/WorkflowNode.'
    )
    required_models: list[str] = Field(default_factory=list, description="Base checkpoint models required.")
    required_loras: list[str] = Field(default_factory=list, description="LoRAs required.")
    required_assets: list[str] = Field(
        default_factory=list,
        description=(
            "Every other required asset (VAEs, ControlNets, IPAdapters, animations, textures, "
            "HDRIs, weapons, environment assets)."
        ),
    )
    gpu_estimate_seconds: float = Field(..., ge=0, description="Estimated GPU render time, in seconds.")
    estimated_vram_gb: float = Field(..., gt=0, description="Estimated peak VRAM required, in gigabytes.")
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Informational notes — optimizations applied, soft caveats. Never a substitute for "
            "raising on a genuinely broken workflow; see WorkflowValidationError."
        ),
    )


class WorkflowBuilder:
    """Converts a CinematicSequence + StyleProfile into a RendererWorkflow.

    Dependencies (asset resolution, parameter mapping, the renderer
    registry) are injected rather than hardcoded, so each can be
    substituted in a test — e.g. a test AssetResolver that returns
    fixed assets without touching a real catalog, once one exists.
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        asset_resolver: AssetResolver | None = None,
        parameter_mapper: ParameterMapper | None = None,
        registry: WorkflowRegistry | None = None,
    ) -> None:
        """Initializes a WorkflowBuilder.

        Args:
            asset_resolver: Resolves required assets. Defaults to a
                fresh AssetResolver.
            parameter_mapper: Maps cinematic intensity to concrete
                parameters. Defaults to a fresh ParameterMapper.
            registry: Tracks known renderer targets. Defaults to a
                fresh WorkflowRegistry (pre-populated with the
                specification's default targets).
        """
        self._asset_resolver = asset_resolver or AssetResolver()
        self._parameter_mapper = parameter_mapper or ParameterMapper()
        self._registry = registry or WorkflowRegistry()

    def build(
        self, sequence: CinematicSequence, style: StyleProfile, renderer: str = "comfyui"
    ) -> RendererWorkflow:
        """Builds a complete renderer workflow for the given sequence and style.

        Args:
            sequence: The cinematic intent to convert into a workflow.
            style: The visual style to render it in.
            renderer: Which registered renderer target to build for.
                Defaults to "comfyui", the first supported target.

        Returns:
            A RendererWorkflow ready for a renderer adapter to
            translate into that renderer's actual input format.

        Raises:
            UnsupportedRendererError: If `renderer` is not registered.
            UnknownStyleError: If `style.style` has no WorkflowTemplate.
            WorkflowValidationError: If the assembled graph fails
                structural validation.
        """
        self._registry.get(renderer)  # raises UnsupportedRendererError if unknown
        template = get_template(style.style)  # raises UnknownStyleError if unknown
        assets = self._asset_resolver.resolve(sequence, style, template)

        node_builder = NodeBuilder(id_prefix=sequence.sequence_id)
        nodes: list[WorkflowNode] = []
        warnings: list[str] = []

        checkpoint_node, vae_node, environment_nodes = self._build_scene(node_builder, template, sequence)
        nodes += [checkpoint_node, vae_node] + environment_nodes

        character_nodes, character_new_nodes, character_warnings = self._build_characters(
            node_builder, sequence.actors, checkpoint_node, template
        )
        nodes += character_new_nodes
        warnings += character_warnings

        timeline = self._build_timeline(sequence.beats)

        for beat_index, beat in enumerate(sequence.beats):
            camera_params = self._build_camera(beat, template)
            sampler_node, animation_node = self._build_sampler_and_animation(
                node_builder, beat, beat_index, sequence, character_nodes, camera_params, template
            )
            lighting_params = self._build_lighting(beat, template)
            effect_nodes = self._build_effects(node_builder, beat)
            audio_node = self._build_audio(node_builder, beat)

            nodes += [sampler_node, animation_node]
            nodes += effect_nodes
            if audio_node is not None:
                nodes.append(audio_node)
            # Lighting is expressed as parameters carried alongside the
            # beat's animation node (there is no dedicated lighting
            # node among the twelve supported types, the same
            # constraint that applies to camera — see
            # node_builder.py's `animatediff` docstring) rather than
            # discarded; recorded here for traceability.
            warnings.append(
                f"Beat {beat.beat_id}: lighting reasoning — "
                f"key={lighting_params['key_light_intensity']:.2f}, "
                f"temperature={lighting_params['color_temperature_kelvin']:.0f}K "
                f"(mood={beat.emotion or 'unspecified'})."
            )

        export_node, export_supporting_nodes = self._build_export(node_builder, nodes, sequence, template)
        nodes += export_supporting_nodes
        nodes.append(export_node)

        optimization_notes = self._describe_optimizations(sequence, assets)
        warnings += optimization_notes

        required_assets = (
            assets.vaes + assets.controlnets + assets.ip_adapters + assets.animations
            + assets.textures + assets.hdris + assets.weapons + assets.environment_assets
        )
        failures = validate_workflow(
            nodes=nodes,
            required_models=assets.models,
            required_assets=assets.loras + required_assets,
            beats=sequence.beats,
            total_duration_seconds=sequence.total_duration_seconds,
        )
        if failures:
            raise WorkflowValidationError(failures)

        return RendererWorkflow(
            renderer=renderer,
            version=self.SCHEMA_VERSION,
            workflow={"nodes": [node.model_dump(mode="json") for node in nodes], "timeline": timeline},
            required_models=assets.models,
            required_loras=assets.loras,
            required_assets=required_assets,
            gpu_estimate_seconds=self._estimate_gpu_seconds(sequence, template),
            estimated_vram_gb=self._estimate_vram_gb(assets, template),
            warnings=warnings,
        )

    # -- Stage: Scene Builder -------------------------------------------------

    def _build_scene(
        self, node_builder: NodeBuilder, template: WorkflowTemplate, sequence: CinematicSequence
    ) -> tuple[WorkflowNode, WorkflowNode, list[WorkflowNode]]:
        """Builds the foundation every beat's nodes share: the base
        checkpoint and VAE for this style, plus the environment's
        texture set and HDRI as IMAGE_LOADER nodes (no dedicated
        "environment" node type exists among the twelve NodeBuilder
        supports).

        Returns:
            The checkpoint node, the VAE node, and a list of the
            environment-related nodes (texture set, HDRI).
        """
        checkpoint_node = node_builder.checkpoint_loader(template.default_model)
        vae_node = node_builder.vae(template.default_vae)
        environment_node = node_builder.image_loader(image_path=sequence.environment)
        texture_node = node_builder.image_loader(image_path=f"{sequence.environment}_texture_set")
        hdri_node = node_builder.image_loader(image_path=f"{sequence.environment}_hdri")
        return checkpoint_node, vae_node, [environment_node, texture_node, hdri_node]

    # -- Stage: Character Builder ----------------------------------------------

    def _build_characters(
        self,
        node_builder: NodeBuilder,
        actors: list[CinematicActor],
        checkpoint_node: WorkflowNode,
        template: WorkflowTemplate,
    ) -> tuple[dict[str, WorkflowNode], list[WorkflowNode], list[str]]:
        """Resolves each actor's character model, LoRA, pose, rig,
        weapon, armor, facial preset, and style preset into a node
        graph.

        An actor whose `character_model` matches the style's default
        chains from the shared `checkpoint_node`. An actor with a
        distinct `character_model` (already flagged as its own
        required model by AssetResolver) gets its own
        CHECKPOINT_LOADER node instead — otherwise that distinct model
        would be resolved as "required" but never actually appear
        anywhere in the built graph.

        Pose and facial-preset conditioning have no dedicated node
        type among the twelve NodeBuilder supports — like camera and
        lighting elsewhere in this builder, they're represented as
        CONTROLNET nodes (the general-purpose "additional conditioning
        signal" type), with `control_type` set to the exact asset
        identifier AssetResolver resolved, so the two stay in sync by
        construction rather than by convention. A weapon reference is
        represented as an IMAGE_LOADER node for the same reason (no
        dedicated "prop" node type exists).

        Returns:
            A (actor_id -> node) mapping for every actor, used by
            later stages to know which node an actor's identity flows
            through — this may be the shared `checkpoint_node` itself,
            so it is *not* safe for the caller to add every value in
            this mapping to the graph (it would duplicate the shared
            checkpoint node's ID); a list of every genuinely new node
            this stage built (per-actor checkpoints, LoRAs,
            pose/facial conditioning, weapons), which *is* safe and
            correct for the caller to add; and a list of warnings for
            anything notable encountered.
        """
        character_nodes: dict[str, WorkflowNode] = {}
        new_nodes: list[WorkflowNode] = []
        warnings: list[str] = []

        for actor in actors:
            if actor.character_model and actor.character_model != template.default_model:
                model_node = node_builder.checkpoint_loader(actor.character_model)
                new_nodes.append(model_node)
            else:
                model_node = checkpoint_node

            if actor.lora:
                lora_node = node_builder.lora_loader(actor.lora, model_node.node_id)
                character_nodes[actor.actor_id] = lora_node
                new_nodes.append(lora_node)
            else:
                character_nodes[actor.actor_id] = model_node
                warnings.append(
                    f"Actor {actor.actor_id!r} has no LoRA — using its checkpoint model directly."
                )

            if actor.pose_reference:
                new_nodes.append(node_builder.controlnet(control_type="pose_controlnet"))
            if actor.facial_preset:
                new_nodes.append(node_builder.controlnet(control_type="facial_ip_adapter"))
            if actor.weapon:
                new_nodes.append(node_builder.image_loader(image_path=actor.weapon))

        return character_nodes, new_nodes, warnings

    # -- Stage: Camera Builder -------------------------------------------------

    def _build_camera(self, beat: CinematicBeat, template: WorkflowTemplate) -> dict[str, float]:
        """Maps this beat's CameraDirection into concrete parameters.

        No dedicated camera node exists among the twelve supported
        node types (see node_builder.py's `animatediff` docstring) —
        this returns plain parameters that `_build_animation` attaches
        to the beat's AnimateDiff node.
        """
        camera = beat.camera
        base_shake = self._parameter_mapper.camera_shake(camera.intensity)
        is_deliberately_shaky = camera.shot_type.value in ("shake", "handheld")
        shaky_scale = 1.0 if is_deliberately_shaky else template.camera.motion_blur_strength
        return {
            "fov_degrees": template.camera.default_fov_degrees,
            "motion_strength": self._parameter_mapper.motion_strength(camera.intensity),
            "camera_shake": base_shake * shaky_scale,
        }

    # -- Stage: Animation Builder -----------------------------------------------

    def _build_sampler_and_animation(
        self,
        node_builder: NodeBuilder,
        beat: CinematicBeat,
        beat_index: int,
        sequence: CinematicSequence,
        character_nodes: dict[str, WorkflowNode],
        camera_params: dict[str, float],
        template: WorkflowTemplate,
    ) -> tuple[WorkflowNode, WorkflowNode]:
        """Builds this beat's Sampler node — the node that actually
        performs generation — chained from a character node, and its
        AnimateDiff motion node chained from that Sampler.

        The beat's own camera intensity adjusts CFG scale via
        ParameterMapper, on top of the style's baseline CFG. The seed
        is derived from the sequence's pinned seed plus the beat's
        index: deterministic (the same sequence always produces the
        same per-beat seeds) without every beat rendering from an
        identical seed.

        Simplification, documented rather than hidden: when a beat
        involves multiple actors, this builds one shared Sampler
        chained from the first-listed actor's character node, not a
        full per-actor compositing graph — building a general
        multi-character compositing pipeline is a materially deeper
        problem than this pass attempts to solve. See this class's
        docstring / the accompanying engineering notes for why.

        Returns:
            The Sampler node and the AnimateDiff node chained from it.
        """
        chain_source = self._first_character_node_for_beat(beat, character_nodes)
        cfg_scale = self._parameter_mapper.cfg_scale(template.render.cfg_scale, beat.camera.intensity)
        beat_seed = sequence.seed + beat_index

        sampler_node = node_builder.sampler(
            checkpoint_node_id=chain_source.node_id,
            cfg_scale=cfg_scale,
            steps=template.render.steps,
            sampler_name=template.render.sampler,
            scheduler=template.render.scheduler,
            seed=beat_seed,
        )

        clip_names = [
            self._asset_resolver.animation_clip_for_action(action.action) for action in beat.actions
        ] or ["clip_idle"]
        frame_count = self._parameter_mapper.frame_count(
            beat.duration_seconds, template.render.fps, beat.camera.intensity
        )
        animation_node = node_builder.animatediff(
            motion_clips=clip_names,
            frame_count=frame_count,
            motion_strength=camera_params["motion_strength"],
            camera_shake=camera_params["camera_shake"],
            fov_degrees=camera_params["fov_degrees"],
        )
        # AnimateDiff has no explicit "inputs" slot of its own in this
        # builder (see node_builder.py) beyond what _build already set;
        # the chain from checkpoint through Sampler is what a renderer
        # adapter would follow to know generation order.
        return sampler_node, animation_node

    def _first_character_node_for_beat(
        self, beat: CinematicBeat, character_nodes: dict[str, WorkflowNode]
    ) -> WorkflowNode:
        """Picks which character node this beat's Sampler chains from —
        see `_build_sampler_and_animation`'s docstring for the
        documented simplification this represents."""
        for action in beat.actions:
            if action.actor_id in character_nodes:
                return character_nodes[action.actor_id]
        # No actor performs an action this beat (e.g. a pure
        # establishing shot) — fall back to any known character node.
        return next(iter(character_nodes.values()))

    # -- Stage: Lighting Builder ------------------------------------------------

    def _build_lighting(self, beat: CinematicBeat, template: WorkflowTemplate) -> dict[str, float]:
        """Determines key/fill/rim/ambient light, temperature, and
        contrast for this beat.

        Reasoning: the style's WorkflowTemplate sets the baseline
        (anime runs cooler and higher-contrast; fantasy runs warmer
        and softer; sci-fi runs colder and harsher; and so on — see
        `workflow_templates.py`'s per-style `LightingDefaults`). A
        beat's own camera intensity (the same 4-tier scale
        `IntensityLevel` uses everywhere else) then nudges exposure and
        contrast: higher intensity pushes contrast and key light up
        and ambient down, for a harsher, more dramatic look; lower
        intensity relaxes toward the style's calm baseline. No
        renderer-specific lighting model is assumed — these are
        abstract intensities an adapter maps to its own light types.
        """
        intensity = beat.camera.intensity
        # Reuses the same 4-tier CFG curve as a stand-in exposure-bias scale.
        exposure_bias = self._parameter_mapper.cfg_scale(0.0, intensity) / 10.0
        return {
            "key_light_intensity": template.lighting.key_light_intensity * (1.0 + exposure_bias * 0.1),
            "fill_light_ratio": template.lighting.fill_light_ratio,
            "rim_light_intensity": template.lighting.rim_light_intensity,
            "ambient_intensity": max(0.0, template.lighting.ambient_intensity - exposure_bias * 0.05),
            "color_temperature_kelvin": template.lighting.color_temperature_kelvin,
            "contrast": template.lighting.contrast + exposure_bias * 0.05,
        }

    # -- Stage: Visual Effects Builder -------------------------------------------

    def _build_effects(self, node_builder: NodeBuilder, beat: CinematicBeat) -> list[WorkflowNode]:
        """Builds one conditioning node per visual effect in the beat,
        scaled by intensity via ParameterMapper.

        Visual effects are represented as ControlNet-style nodes (the
        closest of the twelve supported node types to "an additional
        conditioning signal layered onto the render") with the effect
        type and a 0-1 density carried as parameters — no renderer-
        specific particle-system detail is assumed here.
        """
        return [
            node_builder.controlnet(
                control_type=f"effect_{effect.effect_type.value}",
                strength=self._parameter_mapper.particle_density(effect.intensity),
            )
            for effect in beat.effects
        ]

    # -- Stage: Audio Builder ------------------------------------------------------

    def _build_audio(self, node_builder: NodeBuilder, beat: CinematicBeat) -> WorkflowNode | None:
        """Builds this beat's audio-intent node, if it has audio
        direction. No actual audio is generated — see AUDIO's
        docstring in node_builder.py."""
        if beat.audio is None:
            return None
        return node_builder.audio(
            mood=beat.audio.mood,
            impact_beats=beat.audio.impact_beats,
            voice_placeholder=beat.audio.voice_placeholder,
        )

    # -- Stage: Timeline Builder ----------------------------------------------------

    def _build_timeline(self, beats: list[CinematicBeat]) -> list[dict[str, Any]]:
        """Builds timeline track metadata from each beat's
        timestamp_start/duration_seconds.

        Beats are grouped into overlapping tracks (parallel events): a
        new track starts whenever a beat begins before the previous
        track's last beat has finished, so simultaneous/overlapping
        beats end up on separate tracks rather than being silently
        collapsed — the same approach a video editor's timeline uses
        for overlapping clips.
        """
        tracks: list[list[CinematicBeat]] = []
        for beat in sorted(beats, key=lambda b: b.timestamp_start):
            placed = False
            for track in tracks:
                if track[-1].timestamp_start + track[-1].duration_seconds <= beat.timestamp_start:
                    track.append(beat)
                    placed = True
                    break
            if not placed:
                tracks.append([beat])

        return [
            {
                "track_index": index,
                "beats": [
                    {
                        "beat_id": beat.beat_id,
                        "timestamp_start": beat.timestamp_start,
                        "duration_seconds": beat.duration_seconds,
                    }
                    for beat in track
                ],
            }
            for index, track in enumerate(tracks)
        ]

    # -- Stage: Export Builder --------------------------------------------------------

    def _build_export(
        self,
        node_builder: NodeBuilder,
        nodes_so_far: list[WorkflowNode],
        sequence: CinematicSequence,
        template: WorkflowTemplate,
    ) -> tuple[WorkflowNode, list[WorkflowNode]]:
        """Builds the final video-combine/save/output chain.

        Combines every AnimateDiff node produced so far into one video
        stream, then saves and marks it as the workflow's output.

        Returns:
            The terminal output node, and the supporting combine/save
            nodes the caller should also add to the graph.
        """
        animation_nodes = [n for n in nodes_so_far if n.node_type.value == "animatediff"]
        source_id = animation_nodes[-1].node_id if animation_nodes else nodes_so_far[-1].node_id

        combine_node = node_builder.video_combine(
            source_node_id=source_id,
            fps=template.render.fps,
            frame_interpolation=template.render.frame_interpolation,
        )
        save_node = node_builder.save_video(
            source_node_id=combine_node.node_id, filename_prefix=sequence.sequence_id
        )
        output_node = node_builder.output(source_node_id=save_node.node_id, output_name="final_video")
        return output_node, [combine_node, save_node]

    # -- Optimization reporting (section 16) ----------------------------------------

    def _describe_optimizations(self, sequence: CinematicSequence, assets: ResolvedAssets) -> list[str]:
        """Documents optimizations AssetResolver's own deduplication
        already applied — genuinely computed from before/after counts,
        not a static claim."""
        notes: list[str] = []
        raw_model_refs = [actor.character_model for actor in sequence.actors]
        if len(raw_model_refs) > len(assets.models):
            notes.append(
                f"Shared/deduplicated character models across {len(sequence.actors)} actors: "
                f"{len(raw_model_refs)} references resolved to {len(assets.models)} unique model loads."
            )
        raw_lora_refs = [actor.lora for actor in sequence.actors if actor.lora]
        if len(raw_lora_refs) > len(assets.loras):
            notes.append(
                f"Deduplicated LoRA references: {len(raw_lora_refs)} uses resolved to "
                f"{len(assets.loras)} unique LoRA loads."
            )
        return notes

    # -- Estimation (documented heuristics, pending real calibration) ---------------

    def _estimate_gpu_seconds(self, sequence: CinematicSequence, template: WorkflowTemplate) -> float:
        """Estimates total GPU render time.

        Heuristic: total frames across all beats, times a fixed
        per-frame cost at 25 sampling steps, scaled linearly for the
        style's actual step count. This is a rough starting point —
        there is no real renderer wired up yet to calibrate it
        against; a renderer adapter should replace this once real
        timing data exists.
        """
        total_frames = sum(
            self._parameter_mapper.frame_count(
                beat.duration_seconds, template.render.fps, beat.camera.intensity
            )
            for beat in sequence.beats
        )
        step_scale = template.render.steps / 25.0
        return round(total_frames * _BASE_SECONDS_PER_FRAME_AT_25_STEPS * step_scale, 2)

    def _estimate_vram_gb(self, assets: ResolvedAssets, template: WorkflowTemplate) -> float:
        """Estimates peak VRAM usage.

        Heuristic: a fixed base cost for the checkpoint, plus a fixed
        cost per loaded LoRA, plus a resolution-driven multiplier
        relative to a 1024x576 baseline. Also a rough starting point —
        see `_estimate_gpu_seconds`.
        """
        width, height = template.render.resolution
        resolution_multiplier = (width * height) / _VRAM_RESOLUTION_BASELINE_PIXELS
        base = _BASE_CHECKPOINT_VRAM_GB + len(assets.loras) * _VRAM_PER_LORA_GB
        return round(base * resolution_multiplier, 2)
