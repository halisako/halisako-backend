"""ReferenceMethodCalibrationRunner — Sprint 4 Prompt 16.

Isolates exactly one variable: `reference_latents_method` (the value
`FluxKontextMultiReferenceLatentMethod` sets on both conditioning
branches), for the single known-difficult shot from the Prompt 15.1
live GPU result — timeline index 2, where a duplicated diagonal
spear/polearm still appeared alongside the correct dragon-headed
halberd, despite Prompt 15.1's strengthened, fully generic
identity-preservation language.

    an EXISTING anchor PNG (never generated here)
    + the EXACT Prompt 15.1 shot-2 reference-edit prompt (never
      reworded here)
    + the EXACT Prompt 15.1 shot-2 seed, 981216397 (never re-derived
      here)
    -> three reference-conditioned FLUX generations, one per candidate
       method (offset, uxo/uno, index_timestep_zero), each through its
       own experimental workflow file (see
       workflows/README-reference-method-sweep.md) — all three
       otherwise byte-for-byte identical to the unmodified production
       reference workflow.

The already-paid Prompt 15.1 shot-2 result is the implicit "index"
control — recorded in the manifest, never regenerated here. No T2I
anchor generation, no Wan, no VideoBuilder — exactly 3 ComfyUI jobs
total, the smallest possible test of this one variable.

Reuses `ReferenceSeedCalibrationRunner.prepare()` for anchor validation
and prompt composition (via a repeated-index call — see `prepare()`'s
own docstring for why this is correct, not a workaround), rather than
duplicating that logic. Reuses `ComfyUIImageProvider` directly (its
`reference_workflow_path` constructor argument — already existing,
unmodified — is exactly the seam this experiment needs: one provider
instance per candidate workflow file, all sharing the same pinned-seed
override mechanism Sprint 4 Prompt 15.1 already established).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core.exceptions import ImageProviderError
from core.image_providers.comfyui import ComfyUIImageProvider
from products.chess2fight.rendering.asset_manager import AssetManager
from products.chess2fight.rendering.multi_shot_acceptance import SeedEvidenceMismatchError
from products.chess2fight.rendering.reference_seed_calibration import (
    CalibrationAnchor,
    ReferenceSeedCalibrationRunner,
)
from products.chess2fight.schemas import BattlePreferences

logger = logging.getLogger(__name__)

# The three candidates this sweep generates. "index" is deliberately
# excluded — the already-paid Prompt 15.1 shot-2 result is that
# control; this task's own explicit instruction is not to regenerate
# it.
CANDIDATE_METHODS: tuple[str, str, str] = ("offset", "uxo/uno", "index_timestep_zero")

_METHOD_WORKFLOW_FILENAMES: dict[str, str] = {
    "offset": "flux2_klein_reference_method_offset_4b.json",
    "uxo/uno": "flux2_klein_reference_method_uxo_4b.json",
    "index_timestep_zero": "flux2_klein_reference_method_index_timestep_zero_4b.json",
}

_WORKFLOWS_DIR = Path(__file__).resolve().parent / "workflows"


def method_workflow_path(method: str) -> str:
    """Returns the experimental workflow file path for a given
    candidate method. Raises `KeyError` (with the method named) for
    anything outside `CANDIDATE_METHODS` — never silently falls back
    to the production workflow or a different candidate's file."""
    return str(_WORKFLOWS_DIR / _METHOD_WORKFLOW_FILENAMES[method])


class ExistingControlInfo(BaseModel):
    """Sprint 4 Prompt 16 — records the already-paid Prompt 15.1
    shot-2 result as this experiment's implicit "index" control,
    without regenerating it. `control_image_path`/`control_image_sha256`
    are optional: supplying the actual prior output lets the manifest
    carry real evidence of what's being compared against, but this
    experiment's own generation count and safety guarantees don't
    depend on it being provided."""

    model_config = ConfigDict(frozen=True)

    method: str = Field(default="index", frozen=True)
    generated_this_run: bool = Field(default=False, frozen=True)
    seed: int = Field(..., description="The control's own actual seed — 981216397 for the sample fight.")
    control_image_path: str | None = Field(default=None)
    control_image_sha256: str | None = Field(default=None)


class MethodCandidatePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str = Field(..., min_length=1)
    workflow_path: str = Field(..., min_length=1)


class ReferenceMethodCalibrationPlan(BaseModel):
    """Computed with zero ComfyUI/network calls — safe for dry-run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., min_length=1)
    anchor: CalibrationAnchor = Field(...)
    timeline_index: int = Field(..., ge=0)
    shot_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, description="The exact, unmodified Prompt 15.1 shot-2 reference-edit prompt.")
    planned_seed: int = Field(..., description="The single seed every candidate must resolve to — never re-derived per method.")
    control: ExistingControlInfo = Field(...)
    candidates: list[MethodCandidatePlan] = Field(..., min_length=1)
    expected_comfyui_jobs: int = Field(..., description="Always len(candidates) — the control is never regenerated.")


class MethodCandidateResult(BaseModel):
    method: str
    workflow_path: str
    planned_seed: int
    actual_seed: int
    output_path: str


class ReferenceMethodCalibrationResult(BaseModel):
    plan: ReferenceMethodCalibrationPlan
    candidate_results: list[MethodCandidateResult] = Field(..., min_length=1)


class ReferenceMethodCalibrationRunner:
    """Drives the reference-latent method sweep for a single shot."""

    def __init__(self, ai_provider, asset_manager: AssetManager | None = None) -> None:
        self._seed_runner = ReferenceSeedCalibrationRunner(ai_provider, asset_manager=asset_manager)
        self._asset_manager = asset_manager or AssetManager()

    async def prepare(
        self,
        pgn: str,
        preferences: BattlePreferences,
        anchor_path: str,
        anchor_original_seed: int,
        seed: int,
        style: str,
        battle_mode: str,
        timeline_index: int = 2,
        control_image_path: str | None = None,
        control_image_sha256: str | None = None,
        candidate_methods: tuple[str, ...] = CANDIDATE_METHODS,
    ) -> ReferenceMethodCalibrationPlan:
        """Validates the supplied anchor and plans all three candidate
        generations, all sharing the identical anchor, prompt, and
        seed — only `reference_latents_method` (via each candidate's
        own workflow file) differs between them.

        Reuses `ReferenceSeedCalibrationRunner.prepare()` internally,
        called with `shot_indices=(timeline_index, timeline_index)`
        (the same index twice) and `explicit_seeds=(seed, seed)` (the
        same seed twice) — that method's own contract accepts any two
        indices/seeds without requiring them to differ; passing the
        same value twice deliberately produces two identical shot
        plans, of which only the first is used here. This gets anchor
        validation and prompt composition for free, correctly and
        without duplicating either, rather than reimplementing them
        for a genuinely single-shot case.

        Args:
            seed: The single seed every candidate must use — 981216397
                for the sample fight's shot 2. Never re-derived per
                candidate; this is Sprint 4 Prompt 16's own explicit
                "do not derive a new seed from workflow method"
                requirement.
            timeline_index: Which single shot to reference-condition.
                Defaults to 2 — the known-difficult Prompt 15.1 shot.
            control_image_path: Optional path to the already-paid
                Prompt 15.1 shot-2 output, for manifest evidence only
                — never read, validated, or regenerated by this method.
            control_image_sha256: Optional SHA256 of that same file,
                if the caller already has it computed — never computed
                here (this method never opens `control_image_path` at
                all).
            candidate_methods: Which `reference_latents_method` values
                to plan. Defaults to `CANDIDATE_METHODS` (all three
                non-"index" values this sweep exists to test).

        Returns:
            A ReferenceMethodCalibrationPlan.

        Raises:
            AnchorValidationError: If the anchor is missing, unreadable,
                or not exactly 1280x704 — before any paid job.
        """
        underlying_plan = await self._seed_runner.prepare(
            pgn, preferences, anchor_path=anchor_path, anchor_original_seed=anchor_original_seed,
            style=style, battle_mode=battle_mode, shot_indices=(timeline_index, timeline_index),
            explicit_seeds=(seed, seed),
        )
        shot = underlying_plan.shots[0]

        candidates = [
            MethodCandidatePlan(method=method, workflow_path=method_workflow_path(method))
            for method in candidate_methods
        ]

        return ReferenceMethodCalibrationPlan(
            run_id=underlying_plan.run_id,
            anchor=underlying_plan.anchor,
            timeline_index=timeline_index,
            shot_id=shot.shot_id,
            prompt=shot.prompt,
            planned_seed=seed,
            control=ExistingControlInfo(
                seed=seed, control_image_path=control_image_path, control_image_sha256=control_image_sha256,
            ),
            candidates=candidates,
            expected_comfyui_jobs=len(candidates),
        )

    async def execute(self, plan: ReferenceMethodCalibrationPlan) -> ReferenceMethodCalibrationResult:
        """Runs all three candidate generations, sequentially, each
        through its own `ComfyUIImageProvider` instance (constructed
        with that candidate's own `reference_workflow_path`), all
        pinned to `plan.planned_seed` via the same
        `build_plan_seed_override`-style resolution
        `ReferenceSeedCalibrationRunner` already established — here,
        simplified to a plain constant-returning lambda, since there
        is exactly one (prompt, seed) pair for this entire plan, not a
        multi-shot mapping to disambiguate.

        Raises:
            ImageProviderError: If any candidate's generation fails.
                Never falls back to another method, never retries — a
                failed candidate stops this method immediately (its
                exception propagates), preserving whichever earlier
                candidates already succeeded (their output files
                remain on disk — the loop is sequential).
            SeedEvidenceMismatchError: If any candidate's actual
                provider-reported seed disagrees with `plan.planned_seed`.
        """
        output_dir = self._asset_manager.storage_root / "reference_method_calibration" / plan.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        candidate_results = []
        for candidate in plan.candidates:
            provider = ComfyUIImageProvider(
                reference_workflow_path=candidate.workflow_path,
                seed_override=lambda _prompt, _seed=plan.planned_seed: _seed,
            )
            try:
                result = await provider.generate_reference_conditioned_image(
                    plan.prompt, plan.anchor.path, width=plan.anchor.width, height=plan.anchor.height,
                )
            except ImageProviderError:
                raise  # no fallback, no retry, no substitute method — see this method's own docstring

            actual_seed = result.metadata.get("seed")
            if actual_seed != plan.planned_seed:
                raise SeedEvidenceMismatchError(
                    f"Method candidate {candidate.method!r} (timeline index {plan.timeline_index}): "
                    f"planned seed {plan.planned_seed} does not match the actual provider-reported seed "
                    f"{actual_seed}."
                )

            method_slug = candidate.method.replace("/", "_")
            output_path = output_dir / f"shot{plan.timeline_index}_{method_slug}.png"
            output_path.write_bytes(Path(result.image_path).read_bytes())

            candidate_results.append(
                MethodCandidateResult(
                    method=candidate.method, workflow_path=candidate.workflow_path,
                    planned_seed=plan.planned_seed, actual_seed=actual_seed, output_path=str(output_path),
                )
            )
            logger.info("Method sweep: candidate %r generated at %s (seed %d).", candidate.method, output_path, actual_seed)

        return ReferenceMethodCalibrationResult(plan=plan, candidate_results=candidate_results)
