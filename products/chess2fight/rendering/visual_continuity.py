"""Visual continuity: fight-level base seed derivation and seed
policy — Sprint 4 Prompt 12.

Two entirely separate "seed" concepts exist in this codebase already,
confirmed directly against source before this module was written:

1. `scene_composer.py`'s own `_seed(battle)`/`_pick()` — a
   cinematic-layer, deterministic *vocabulary selection* seed (which
   hair/clothing/weapon phrase to pick from each style's word pool).
   Computed once per fight, already reused across every shot via one
   shared `SceneContinuity` object — already fully stable, needing no
   change here (see `prompt_generator.py`'s `_stable_continuity_block`
   docstring for the direct verification of this).

2. `core/image_providers/comfyui.py` / `core/animation_providers/comfyui.py`'s
   `_derive_seed(prompt)` — a ComfyUI-layer *image-generation noise*
   seed, independently derived per call from a SHA256 hash of whatever
   prompt text is given. This is the seed the real three-shot GPU
   evidence showed varying per shot (2192948747, 1709036070,
   1265274475) — not because the fighter identity text varied (it was
   confirmed byte-for-byte identical across all three shots in that
   real run — see this same evidence, independently re-verified before
   this module was written), but because each shot's *overall* prompt
   text differs (different action/camera/mood fragments), and
   `_derive_seed` hashes the whole string.

This module concerns itself only with (2) — offering an explicit,
testable, opt-in alternative to "always hash the current prompt" for
FLUX specifically, so a controlled experiment can test whether a
shared or base-seed-derived FLUX seed improves visual identity
consistency across independently-generated shots that already share
identical identity text. Wan's own seed derivation is untouched
entirely — `AnimationPipeline._build_instruction()` still sets
`AnimationInstruction.prompt = shot.image_prompt` unchanged, so
`ComfyUIAnimationProvider` continues deriving its own seed from each
shot's own prompt exactly as before this module existed.

Per this task's explicit instruction: a shared FLUX seed is offered
here as a controlled experiment to run and observe, not asserted to
solve consistency by itself.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from enum import Enum


class VisualSeedPolicy(str, Enum):
    """How a multi-shot acceptance run resolves each selected shot's
    FLUX seed. Wan's own seed is never affected by this policy — see
    this module's own docstring."""

    DEFAULT = "default"
    """Unchanged, pre-Prompt-12 behavior: each shot's FLUX seed is
    `_derive_seed(shot.image_prompt)`, independently, exactly as
    `ComfyUIImageProvider` has always computed it. No fight-level base
    seed is computed or used at all under this policy."""

    SHARED = "shared"
    """Every selected shot's FLUX seed is the same fight-level base
    seed, regardless of that shot's own prompt text. Policy A in this
    task's own framing — a controlled experiment testing whether one
    shared seed alone improves visual identity consistency, given that
    the identity text itself is already stable across shots."""

    DERIVED = "derived"
    """Every selected shot's FLUX seed is deterministically derived
    from the fight-level base seed combined with that shot's own
    prompt text — still varies per shot (unlike SHARED), but is
    reproducible from the same (base_seed, prompt) pair rather than
    depending solely on the prompt's own hash. Policy B in this task's
    own framing."""


def derive_fight_base_visual_seed(pgn: str, style: str, battle_mode: str) -> int:
    """A fight's deterministic base visual seed — the one explicit,
    testable value every per-shot FLUX seed resolution (SHARED or
    DERIVED policy) is ultimately grounded in.

    Deterministic and pure: the exact same (pgn, style, battle_mode)
    always yields the exact same base seed — no random process state,
    no clock, no external input. Different inputs typically (not
    guaranteed, given hashing) yield different seeds; this function
    never forces unrelated fights toward the same value — it's a
    property of the specific fight's own inputs, not a global
    constant.

    Args:
        pgn: The game's PGN text — the primary source of a fight's
            identity for this purpose (the same game text always
            describes the same fight).
        style: The visual style requested (e.g. "anime") — included so
            the same PGN rendered in two different styles gets
            different base seeds, matching that a different style
            genuinely produces a different-looking fight.
        battle_mode: The battle mode requested (e.g. "duel") — same
            reasoning as `style`.

    Returns:
        A deterministic non-negative integer seed.
    """
    combined = f"{pgn}\x00{style}\x00{battle_mode}"
    return int(hashlib.sha256(combined.encode("utf-8")).hexdigest()[:8], 16)


def derive_shot_seed(base_seed: int, prompt: str) -> int:
    """Deterministically derives one shot's FLUX seed from a fight's
    base visual seed and that shot's own prompt text — the DERIVED
    policy's own resolution rule.

    Deliberately still varies per shot (combines `prompt` into the
    hash), unlike SHARED, which ignores `prompt` entirely — see
    `build_seed_override`'s own docstring for exactly where that
    distinction is applied. Deterministic and pure: the same
    (base_seed, prompt) pair always yields the same result.
    """
    combined = f"{base_seed}\x00{prompt}"
    return int(hashlib.sha256(combined.encode("utf-8")).hexdigest()[:8], 16)


def build_seed_override(policy: VisualSeedPolicy, base_seed: int) -> Callable[[str], int] | None:
    """Builds the `seed_override` callable `ComfyUIImageProvider`
    accepts (see that class's own constructor docstring), for a given
    policy and base seed.

    Returns:
        - `None` for `VisualSeedPolicy.DEFAULT` — no override at all;
          `ComfyUIImageProvider` falls back to its own unchanged
          `_derive_seed(prompt)` behavior, exactly as before this
          module existed.
        - For `SHARED`: a callable that ignores its `prompt` argument
          entirely and always returns `base_seed` — every selected
          shot's FLUX seed is identical.
        - For `DERIVED`: a callable that combines `base_seed` with
          each call's own `prompt` via `derive_shot_seed` — varies per
          shot, but is reproducible from (base_seed, prompt) rather
          than depending solely on the prompt's own hash.
    """
    if policy == VisualSeedPolicy.DEFAULT:
        return None
    if policy == VisualSeedPolicy.SHARED:
        return lambda prompt: base_seed
    if policy == VisualSeedPolicy.DERIVED:
        return lambda prompt: derive_shot_seed(base_seed, prompt)
    raise ValueError(f"Unknown VisualSeedPolicy: {policy!r}")
