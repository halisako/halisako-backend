"""Validates a fully assembled workflow graph before it's handed to a
renderer adapter.

Deliberately operates on primitives (a list of `WorkflowNode`, plain
lists of required asset identifiers, a list of `CinematicBeat`) rather
than importing `RendererWorkflow` from `workflow_builder.py` — that
keeps this module's dependency graph one-directional
(`workflow_builder.py` imports this module, never the reverse), which
is what avoids a circular import between the two.

Every check here is about the *graph's own internal consistency* —
node IDs, edges, declared requirements actually being used, timing
against the sequence's own stated duration. None of it can verify that
a referenced model or asset *actually exists* on disk; that's an asset
catalog's concern, not this module's.
"""

from __future__ import annotations

from core.rendering.workflow.node_builder import NodeType, WorkflowNode
from core.rendering.workflow.workflow_templates import CinematicBeat

# Node types that count as a valid terminal output for the "missing
# outputs" check — SAVE_VIDEO and PREVIEW are both legitimate ways a
# workflow can expose a result, not just OUTPUT specifically.
_OUTPUT_LIKE_TYPES = frozenset({NodeType.OUTPUT, NodeType.SAVE_VIDEO, NodeType.PREVIEW})

# The parameter keys NodeBuilder always sets for each node type — used
# to catch a node that claims a type but is missing what that type
# requires (most likely from a node constructed by hand, bypassing
# NodeBuilder, rather than a NodeBuilder bug).
_EXPECTED_PARAMETER_KEYS: dict[NodeType, frozenset[str]] = {
    NodeType.IMAGE_LOADER: frozenset({"image_path"}),
    NodeType.CHECKPOINT_LOADER: frozenset({"model"}),
    NodeType.LORA_LOADER: frozenset({"lora", "strength"}),
    NodeType.SAMPLER: frozenset({"cfg_scale", "steps", "sampler_name", "scheduler", "seed"}),
    NodeType.CONTROLNET: frozenset({"control_type", "strength"}),
    NodeType.ANIMATEDIFF: frozenset(
        {"motion_clips", "frame_count", "context_length", "motion_strength", "camera_shake", "fov_degrees"}
    ),
    NodeType.VAE: frozenset({"vae_model"}),
    NodeType.VIDEO_COMBINE: frozenset({"fps", "frame_interpolation"}),
    NodeType.AUDIO: frozenset({"mood", "impact_beats", "voice_placeholder"}),
    NodeType.SAVE_VIDEO: frozenset({"filename_prefix"}),
    NodeType.PREVIEW: frozenset(),
    NodeType.OUTPUT: frozenset({"output_name"}),
}


def validate_workflow(
    nodes: list[WorkflowNode],
    required_models: list[str],
    required_assets: list[str],
    beats: list[CinematicBeat],
    total_duration_seconds: float,
) -> list[str]:
    """Validates an assembled workflow graph.

    Args:
        nodes: Every node in the assembled graph.
        required_models: Model/LoRA/VAE identifiers the AssetResolver
            determined are required (any category — this check only
            confirms each one is referenced somewhere, not which
            category).
        required_assets: Every other required asset identifier
            (animations, textures, HDRIs, weapons, environment assets).
        beats: The sequence's beats, for timing validation.
        total_duration_seconds: The sequence's stated total duration.

    Returns:
        A list of human-readable validation failure descriptions.
        Empty if the workflow is valid. Every check runs regardless of
        whether earlier checks failed, so a caller sees every problem
        in one pass rather than one at a time.
    """
    failures: list[str] = []
    failures.extend(_check_duplicate_node_ids(nodes))
    failures.extend(_check_broken_graph(nodes))
    failures.extend(_check_missing_outputs(nodes))
    failures.extend(_check_unsupported_parameters(nodes))
    failures.extend(_check_required_identifiers_referenced(nodes, required_models, "model"))
    failures.extend(_check_required_identifiers_referenced(nodes, required_assets, "asset"))
    failures.extend(_check_invalid_timing(beats, total_duration_seconds))
    return failures


def _check_duplicate_node_ids(nodes: list[WorkflowNode]) -> list[str]:
    seen: set[str] = set()
    failures: list[str] = []
    for node in nodes:
        if node.node_id in seen:
            failures.append(f"Duplicate node ID: {node.node_id!r}.")
        seen.add(node.node_id)
    return failures


def _check_broken_graph(nodes: list[WorkflowNode]) -> list[str]:
    known_ids = {node.node_id for node in nodes}
    failures: list[str] = []
    for node in nodes:
        for slot, upstream_id in node.inputs.items():
            if upstream_id not in known_ids:
                failures.append(
                    f"Node {node.node_id!r} input {slot!r} references unknown node "
                    f"{upstream_id!r}."
                )
    return failures


def _check_missing_outputs(nodes: list[WorkflowNode]) -> list[str]:
    if not any(node.node_type in _OUTPUT_LIKE_TYPES for node in nodes):
        return ["Workflow has no output node (expected at least one OUTPUT, SAVE_VIDEO, or PREVIEW)."]
    return []


def _check_unsupported_parameters(nodes: list[WorkflowNode]) -> list[str]:
    failures: list[str] = []
    for node in nodes:
        expected = _EXPECTED_PARAMETER_KEYS.get(node.node_type)
        if expected is None:
            continue
        missing = expected - node.parameters.keys()
        if missing:
            failures.append(
                f"Node {node.node_id!r} ({node.node_type.value}) is missing expected "
                f"parameters: {sorted(missing)}."
            )
    return failures


def _check_required_identifiers_referenced(
    nodes: list[WorkflowNode], required: list[str], label: str
) -> list[str]:
    referenced_values: set[str] = set()
    for node in nodes:
        for value in node.parameters.values():
            if isinstance(value, (list, tuple)):
                referenced_values.update(str(item) for item in value)
            else:
                referenced_values.add(str(value))

    failures: list[str] = []
    for identifier in required:
        if identifier not in referenced_values:
            failures.append(f"Required {label} {identifier!r} is not referenced by any node.")
    return failures


def _check_invalid_timing(beats: list[CinematicBeat], total_duration_seconds: float) -> list[str]:
    failures: list[str] = []
    for beat in beats:
        beat_end = beat.timestamp_start + beat.duration_seconds
        if beat_end > total_duration_seconds + 1e-6:
            failures.append(
                f"Beat {beat.beat_id!r} ends at {beat_end:.3f}s, after the sequence's stated "
                f"total_duration_seconds ({total_duration_seconds:.3f}s)."
            )
    return failures
