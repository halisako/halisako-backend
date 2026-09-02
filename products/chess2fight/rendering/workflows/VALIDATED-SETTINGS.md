# ComfyUI providers — validated runtime requirements

This documents the exact, experimentally-validated configuration behind
`ComfyUIImageProvider` and `ComfyUIAnimationProvider` (Sprint 4 Prompts
4–6). Both were proven end to end on a real GPU: a FLUX-generated
keyframe was fed directly into the Wan provider and produced real,
verified motion. This development environment does not have GPU/ComfyUI
access itself — see `core/animation_providers/comfyui.py`'s own
environment audit for that — so everything below reflects the supplied,
externally-validated artifacts, not anything run in this environment.

## Live milestone status (Sprint 4 Prompt 13)

**PROVEN LIVE** — real RTX 4090, real ComfyUI history inspected directly:

- Single-shot Chess2Fight: 1 FLUX job + 1 Wan job, real end-to-end
  success. `products/chess2fight/rendering/single_shot_acceptance.py`
  via `scripts/render_single_shot.py`.
- Three-shot Chess2Fight, independent (per-prompt) FLUX seeds: 3 FLUX
  jobs + 3 Wan jobs, real ComfyUI history confirmed exactly 6
  generation jobs, local concatenation via
  `VideoBuilder.concatenate_clips()` into one real 51-frame, 6.375s
  final MP4. This same real evidence also exposed prompt-composition
  and visual-identity-drift findings addressed in Sprint 4 Prompt 12.
- Three-shot Chess2Fight, shared FLUX seed: same 6-job/6.375s
  generation-count and duration contract, `visual_seed_policy=shared`,
  `fight_base_visual_seed=1697950441` confirmed injected into all
  three FLUX jobs' `RandomNoise` node and recorded correctly as both
  `planned_flux_seed` and `actual_flux_seed` for all three shots.
  **Result: a material but insufficient improvement** — face/palette
  visibly more stable across shots, but hairstyle, armor construction,
  and especially weapon geometry (three genuinely different weapon
  designs observed across three shots, for the same fighter, same
  weapon *name* in every prompt) still drifted substantially. Directly
  viewed and confirmed against the real generated keyframes, not just
  the task's own description of them — this finding is what motivated
  Sprint 4 Prompt 13's reference-conditioning work below.

**NOT YET PROVEN**:

- Reference-conditioned FLUX generation (Sprint 4 Prompt 13) — built,
  not yet GPU-tested. The new workflow file
  (`flux2_klein_reference_4b.json`) is a well-grounded, research-based
  extension of the proven T2I graph, not itself a supplied, validated
  export like the other workflow files in this directory — see that
  file's own README before trusting its exact graph shape. The first
  real run against it is the validation step.
- Whether reference-conditioning actually solves the identity-drift
  problem the shared-seed experiment left unsolved is a genuinely open
  empirical question for that first real run.
- All 8 timeline shots / a complete fight render: not attempted, not
  built.

## Validated environment

| | |
|---|---|
| ComfyUI version | 0.33.1 |
| GPU | RTX 4090 (24GB VRAM) |

## FLUX.2 Klein 4B (distilled) — image generation

| Setting | Value |
|---|---|
| Diffusion model | `flux-2-klein-4b.safetensors` |
| Text encoder | `qwen_3_4b.safetensors` |
| VAE | `flux2-vae.safetensors` |
| Resolution | 1280 × 704 |
| Steps | 4 |
| CFG | 1 |
| Sampler | euler |
| Workflow file | `products/chess2fight/rendering/workflows/flux2_klein_t2i_4b.json` |

Distilled branch specifically — not the "base" (non-distilled) FLUX.2
Klein variant, which uses different settings (CFG 5, more steps) and a
different checkpoint (`flux-2-klein-base-4b.safetensors`). The supplied
UI-export companion file
(`flux2_klein_t2i_4b.ui_export.json`) contains both branches as
ComfyUI subgraphs; the API-format file used by the provider is the
distilled one, confirmed by its CFG=1/steps=4 values matching exactly.

## Runtime vs. reference workflow files

Only two files under `products/chess2fight/rendering/workflows/` are
ever loaded at runtime — `wan22_i2v_5b.json` and `wan22_t2v_5b.json`
(both API-format, referenced by `settings.comfyui_workflow_path` /
`.comfyui_t2v_workflow_path`), plus `flux2_klein_t2i_4b.json` for the
image provider. Every `*.ui_export.json` file (the regular "Save"
format, with node positions/UI metadata/subgraphs) is reference/debug
material only — useful for opening in ComfyUI's editor to inspect or
modify visually, never read by any provider. No setting points at a
`.ui_export.json` path; this is enforced by construction, not by a
runtime check.

## Wan 2.2 TI2V 5B — image-to-video (I2V, primary mode)

| Setting | Value |
|---|---|
| Diffusion model | `wan2.2_ti2v_5B_fp16.safetensors` |
| Text encoder | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| VAE | `wan2.2_vae.safetensors` |
| Resolution | 832 × 480 |
| Frames | 17 (≈2.125s at 8fps) |
| FPS | 8 |
| Steps | 8 |
| CFG | 5 |
| Sampler | uni_pc |
| Scheduler | simple |
| Denoise | 1 |
| Workflow file | `products/chess2fight/rendering/workflows/wan22_i2v_5b.json` |

**Sprint 4 Prompt 8 superseded these values** — a newer live validation
on a RunPod RTX 4090 host used 832×480/8fps/17 frames/8 steps, differing
from Prompt 4's earlier 640×352/24fps/49-frame/20-step proof. Both are
genuine, live-validated data points from different sessions; this
backend uses the more recent one as its default. `settings.comfyui_default_fps`
is `8`, not `24`.

The workflow file was updated to the version actually used in a live
FLUX→Wan run (the `LoadImage` node's default value references the real
generated keyframe filename from that run — always overwritten by the
provider's own upload step before submission, never read as-is).

**A discrepancy worth recording plainly**: the artifacts supplied for
Prompt 8 labeled one file "the I2V API-format workflow... the most
important artifact," and another as merely useful for "understanding
the working node structure" of "the small Wan 2.2 test." The actual
JSON structure showed the reverse — the file called "the I2V workflow"
has no `LoadImage` node and no `start_image` wiring at all (structurally
text-to-video); the file described as a small test has the real,
wired image-conditioning connection (structurally image-to-video).
`wan22_i2v_5b.json` and `wan22_t2v_5b.json` were built from the
structurally-correct files, not the labels — see
`core/animation_providers/comfyui.py`'s module docstring for the full
account.

## Wan 2.2 TI2V 5B — text-to-video (T2V, secondary mode)

| Setting | Value |
|---|---|
| Diffusion model | `wan2.2_ti2v_5B_fp16.safetensors` (same as I2V) |
| Text encoder | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (same as I2V) |
| VAE | `wan2.2_vae.safetensors` (same as I2V) |
| Resolution | 832 × 480 |
| Frames | 17 |
| FPS | 8 |
| Steps | 8 |
| CFG | 5 |
| Workflow file | `products/chess2fight/rendering/workflows/wan22_t2v_5b.json` |

Set `AnimationInstruction.animation_type = AnimationType.TEXT_TO_VIDEO`
and omit `source_image_path` to use this mode.
`ComfyUIAnimationProvider` selects between the I2V and T2V workflow
files automatically based on this field — no separate provider class,
no separate router registration. `MockAnimationProvider` does not
support T2V (it has no image to hold static) and returns a clear
failure if given a text-to-video instruction; Chess2Fight's own
production pipeline (`AnimationPipeline`) always uses I2V, since it
always renders a still frame first — T2V exists for direct/standalone
provider use.

## Provider selection

Set via the existing settings (environment-variable overridable, same
Pydantic Settings mechanism as every other setting in `core/config.py`):

```
image_provider=comfyui
animation_provider=comfyui
comfyui_base_url=http://<your-comfyui-host>:8188
```

Both default to `mock` — ordinary tests and `/generate`/`/render` never
require GPU, ComfyUI, or these model files unless explicitly configured
otherwise.

## Running the live tests

Neither runs as part of `pytest tests/`. Each requires a real, running
ComfyUI server with the respective models installed:

```bash
COMFYUI_LIVE_TEST=1 pytest tests/test_comfyui_live_integration.py -v -s
COMFYUI_IMAGE_LIVE_TEST=1 pytest tests/test_comfyui_image_live_integration.py -v -s
```

`COMFYUI_LIVE_TEST=1` now exercises both animation modes in one run —
I2V (against `wan22_i2v_5b.json`) and T2V (against `wan22_t2v_5b.json`),
added Sprint 4 Prompt 8.

## Standalone ComfyUI provider smoke test (Sprint 4 Prompt 9)

Before running the gated pytest above, or the full Chess2Fight
single-shot path below, this is the narrowest possible real check —
it exercises only `ComfyUIAnimationProvider` directly, with a
manually-supplied image and prompt, no PGN or chess analysis involved
at all.

### Required model files and directories

ComfyUI must have these three files already installed in the
directories shown (paths are ComfyUI's own standard model layout):

```
ComfyUI/
├── models/
│   ├── diffusion_models/
│   │   └── wan2.2_ti2v_5B_fp16.safetensors
│   ├── text_encoders/
│   │   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors
│   └── vae/
│       └── wan2.2_vae.safetensors
```

This document does not include, link to, or bundle any model binary —
only the filenames and expected directories above.

### Exact command

```bash
python scripts/comfyui_single_shot_smoke.py \
    --base-url http://<comfyui-host>:8188 \
    --image path/to/reference.png \
    --prompt "Cinematic battle animation. The character comes alive and moves forward with controlled aggressive motion."
```

Defaults to the validated baseline if not overridden: 832×480, 8fps,
17 frames (≈2.125s), steps=8, CFG=5, sampler=uni_pc — printed explicitly
before submission, along with the resolved prompt/job ID once queued.

### What constitutes success

The script itself prints `SUCCESS` and exits `0` only if all of the
following actually happened — never fabricated:

1. the local image was found and uploaded to ComfyUI;
2. the mutated workflow was accepted (`/prompt` didn't return
   `node_errors`);
3. polling `/history/{prompt_id}` reached a non-error completed state;
4. the resulting video was located and downloaded from ComfyUI;
5. the downloaded file passed local verification (exists, non-empty,
   `ffprobe`-readable, valid duration/dimensions).

Any failure at any step prints the real error message to stderr and
exits `1` — no fallback file, no partial success.

## Expected ComfyUI job count for one acceptance run (Sprint 4 Prompt 10.1)

Selecting exactly one cinematic shot causes **two** ComfyUI generation
jobs, not one:

1. one FLUX `/prompt` submission (the reference keyframe, 1280×704), and
2. one Wan `/prompt` submission (the animation, 832×480).

"Exactly one Wan `/prompt` submission" is the correct statement about
the animation step specifically. "Exactly one `/prompt` submission
total" would be wrong — it would incorrectly suggest FLUX generation
doesn't count, or didn't happen.

## Single-shot acceptance path (Sprint 4 Prompt 7)

Before attempting a full render, prove the real Chess2Fight rendering
architecture (not just the providers standalone) works for one real
shot. See `products/chess2fight/rendering/single_shot_acceptance.py`
and `scripts/render_single_shot.py`.

### Dry run (no ComfyUI needed, works anywhere)

```bash
python scripts/render_single_shot.py --sample --shot-index 0 --dry-run
```

### Exact command for the next real RTX 4090 host

```bash
export IMAGE_PROVIDER=comfyui
export ANIMATION_PROVIDER=comfyui
export COMFYUI_BASE_URL=http://<your-comfyui-host>:8188

python scripts/render_single_shot.py --sample --shot-index 0
```

This selects shot 0 (the "establishing" shot) from the bundled sample
game, renders its real `image_prompt` through the validated FLUX
workflow, and animates the result through the validated Wan workflow —
printing both the resulting image and video paths on success.

### Low-cost GPU smoke test (Sprint 4 Prompt 7.1, values updated Prompt 10)

Before spending GPU time on a shot's full real duration (shot 0's real
duration is 7.75s → 61 Wan frames at the current 8fps default), first
prove the FLUX → Wan handoff works with a capped, low-cost animation
duration:

```bash
export IMAGE_PROVIDER=comfyui
export ANIMATION_PROVIDER=comfyui
export COMFYUI_BASE_URL=http://127.0.0.1:8188

python scripts/render_single_shot.py \
    --sample \
    --shot-index 0 \
    --max-animation-seconds 2
```

This uses the real shot's real `image_prompt` (unmodified), generates
a real FLUX keyframe (1280×704 — `settings.comfyui_image_default_width`
/`.height`, independent of the animation resolution below), and
animates it for a capped ~2 seconds — resolving to exactly **17
frames at 8fps (2.125s effective duration)**, the current validated
Wan baseline, at 832×480. `--max-animation-seconds` never changes the
shot's own real cinematic duration — only the duration used for the
animation step in this one acceptance run. (An earlier version of this
document, written before Sprint 4 Prompt 8's live validation changed
the default from 24fps to 8fps, said "49 frames at 24fps" here — that
was this feature's original, now-superseded reference point, not a
claim about the current default.)

Once that succeeds, run the subsequent full-duration command (no cap):

```bash
python scripts/render_single_shot.py \
    --sample \
    --shot-index 0
```

This requests the shot's full real duration — approximately 7.75
seconds, resolving to 61 frames at the current 8fps default (not the
185 frames a stale 24fps assumption would have suggested).

### Windows portability note

Workflow JSON loading in both `core/image_providers/comfyui.py` and
`core/animation_providers/comfyui.py` uses explicit
`Path.read_text(encoding="utf-8")` — required for
`wan22_i2v_5b.json`, whose negative prompt (node 7) is non-ASCII
Chinese text that Windows' default locale encoding (cp1252) cannot
decode correctly.

### Live acceptance test (same path, as a gated pytest)

```bash
HALISAKO_SINGLE_SHOT_LIVE_TEST=1 \
IMAGE_PROVIDER=comfyui ANIMATION_PROVIDER=comfyui \
COMFYUI_BASE_URL=http://<your-comfyui-host>:8188 \
pytest tests/test_single_shot_live_acceptance.py -v -s
```

Ordinary `pytest tests/` never runs this — it requires the environment
variable above and a real, reachable ComfyUI server with both
validated model sets installed.


## Three-shot (capped multi-shot) acceptance path (Sprint 4 Prompt 11)

Generalizes the single-shot acceptance path above from exactly one
shot to a capped range of shots — same production wiring
(`RenderPipeline`, `AnimationPipeline`, `AnimationRouter`,
`ImageRouter`), plus a final local `VideoBuilder.concatenate_clips()`
step. Default and safety-capped at 3 shots — see
`products/chess2fight/rendering/multi_shot_acceptance.py`'s own
`ShotCountExceedsAcceptanceCapError` for why exceeding it requires an
explicit `--allow-more-than-cap` flag, never a convenience default.

### Generation count contract

For 3 selected shots: **3 FLUX jobs + 3 Wan jobs = 6 ComfyUI generation
jobs total**, plus exactly 1 local concatenation (never a 7th ComfyUI
job) and 0 additional AI generation jobs of any kind. "Exactly one Wan
`/prompt` submission" describes one shot; for the whole 3-shot run, the
correct statement is "exactly 3 Wan `/prompt` submissions."

### Dry run (no ComfyUI needed, works anywhere)

```bash
python scripts/render_multi_shot_acceptance.py --sample --dry-run --max-animation-seconds 2
```

### Exact first paid three-shot command

```bash
export IMAGE_PROVIDER=comfyui
export ANIMATION_PROVIDER=comfyui
export COMFYUI_BASE_URL=http://127.0.0.1:8188

python scripts/render_multi_shot_acceptance.py \
    --sample \
    --start-shot-index 0 \
    --shot-count 3 \
    --max-animation-seconds 2
```

Selects real timeline shots 0, 1, 2 (never fabricated, never a
different image_prompt), caps each shot's animation to ~2s (17 frames
at the current 8fps default per shot — the same validated baseline as
the single-shot path), renders 3 real FLUX keyframes at 1280x704,
animates 3 real Wan clips at 832x480, and concatenates them via the
real `VideoBuilder.concatenate_clips()` into one final MP4 — expected
duration approximately 3 x 2.125s = 6.375s (before any
container/encoding rounding on the real measured value). Writes
`multi_shot_acceptance_manifest.json` (path configurable via
`--manifest-path`) recording every shot's prompt, durations, frame
count, and artifact paths, in timeline order.

### Live acceptance test (same path, as a gated pytest)

```bash
COMFYUI_MULTI_SHOT_LIVE_TEST=1 \
IMAGE_PROVIDER=comfyui ANIMATION_PROVIDER=comfyui \
COMFYUI_BASE_URL=http://<your-comfyui-host>:8188 \
pytest tests/test_multi_shot_live_acceptance.py -v -s
```

Ordinary `pytest tests/` never runs this. Do not run this — or the
paid CLI command above — until the single-shot path (already
GPU-proven) has been re-confirmed working on whatever host will run
this; see "Cost control" below.

### Cost control — do not run until three-shot acceptance passes

Do not attempt: a `--shot-count` above 3 (even with the override
flag), the full 8-shot timeline, uncapped per-shot animation duration,
or a full `/render` fight — until the capped 3-shot command above
succeeds and its evidence (6 ComfyUI history entries, 3 valid FLUX
keyframes, 3 valid Wan clips, 1 valid concatenated MP4) has been
manually reviewed.


## Visual continuity foundation (Sprint 4 Prompt 12)

The real three-shot GPU evidence's own manifest was inspected directly
(not just the task's description of it) and confirmed: fighter
identity text (hair, facial features, clothing, armor, weapon) was
already byte-for-byte identical across all three shots for both
fighters — `scene_composer.py`'s `compose_scene()` builds exactly one
`SceneContinuity` per fight and assigns the same object to every shot,
confirmed directly against source. The observed visual drift is
therefore not explained by unstable prompt text. Two things were
confirmed genuinely real instead:

1. **A concrete prompt-composition bug**: the real evidence contained
   `"...arena., deep dusk..."` — a duplicate-punctuation defect.
   Reproduced from first principles (not assumed) by running the
   actual pipeline: `shot.description` for the establishing shot ends
   in a literal period (`timeline_engine.py`'s own f-string), and the
   prior inline join only stripped trailing commas, not periods. Fixed
   generally in `products/chess2fight/cinematic/prompt_composer.py`
   (strips *any* trailing punctuation, not just the one observed
   case). The task's three other named example defects
   ("keylightis", "dynamicspeed", "thescene") were searched for
   directly in the real evidence and were **not** found there —
   reported honestly as illustrative examples of a defect class, not
   literal strings this specific run produced.
2. **Each shot's FLUX seed varies** (confirmed against the real
   ComfyUI history: 2192948747, 1709036070, 1265274475) — not because
   identity text varies, but because `_derive_seed()` hashes the
   *entire* prompt string, which does vary shot-to-shot (different
   action/camera/mood text). A shared or base-seed-derived FLUX seed
   is therefore a genuinely untested variable, independent of the
   prompt-hygiene fix above — see `visual_continuity.py`.

### Prompt composition contract

`prompt_generator.py`'s `_build_prompt()` now explicitly builds four
named blocks — stable continuity, shot action, shot camera, global
style — joined via `compose_prompt_from_blocks()` rather than one flat
clause list joined by fragile raw string concatenation.

### Visual seed policy

```bash
python scripts/render_multi_shot_acceptance.py --sample --dry-run --visual-seed-policy shared
python scripts/render_multi_shot_acceptance.py --sample --dry-run --visual-seed-policy derived
```

`default` (unchanged pre-Prompt-12 behavior), `shared` (Policy A —
every selected shot gets the identical fight-level base FLUX seed),
`derived` (Policy B — deterministically derived per shot from the same
base seed, still varies per shot). Wan's own seed derivation is
**never** affected by this flag under any policy — still
`_derive_seed(shot.image_prompt)`, exactly as before. Do not claim a
shared seed guarantees identity consistency — this is a controlled
experiment to run and observe, not a verified fix.

### Reference-conditioning audit (researched, not implemented)

FLUX.2 Klein — the model family Halisako already uses — natively
supports reference-image conditioning (up to 4 reference images,
guiding identity/style/composition), confirmed via current
documentation and community workflows, not assumed. This is a
**native model capability**, not a ControlNet/IP-Adapter-style add-on
— the least invasive option among the families the task asked about.

Two concrete caveats, both confirmed via research:

- The reference-conditioned (image-edit) ComfyUI workflow graph is
  **structurally different** from the plain text-to-image graph
  Halisako currently uses (different nodes: multi-reference-latent
  conditioning, a different sampler chain) — this would require a new
  workflow file, not a parameter tweak to the existing one.
- The 4B image-editing variant's documented example specifically uses
  the **undistilled base** model, not the **distilled 4-step** model
  Halisako currently runs (`flux-2-klein-4b.safetensors`, CFG=1,
  steps=4). Distilled vs. base generation time differs by roughly an
  order of magnitude in vendor-published benchmarks (~1.2s vs. ~17s on
  a 5090) — reference-conditioning may carry a substantial per-shot
  cost increase, not just an architectural one.
- Community reports (not vendor-verified) describe multi-reference
  results as sometimes inconsistent — face vs. outfit vs. background
  can drift independently even with reference conditioning — relevant
  given Halisako needs two distinct fighters' identities preserved
  simultaneously in most shots, not one subject.

**Implemented in Prompt 12**: prompt-composition hygiene, the explicit
composition contract, the FLUX seed policy (shared/derived).
**Possible next escalation if prompt+seed consistency proves
insufficient**: reference-conditioned FLUX generation, requiring a new
workflow file, likely the undistilled base model, and a mechanism to
generate/reuse canonical per-fighter reference images across shots —
none of this is implemented, evaluated only.


## Reference-conditioned visual continuity (Sprint 4 Prompt 13)

**Correction to the Prompt 12 section above**: it speculated reference
conditioning would "likely" require the undistilled base model. Fresh,
targeted research (Sprint 4 Prompt 13's own explicit instruction not
to assume this) found a dedicated, official "Flux.2 [Klein] 4B
Distilled: Image Edit" ComfyUI workflow exists, using the exact same
distilled model (`flux-2-klein-4b.safetensors`) Halisako's T2I
workflow already uses — confirmed via multiple independent sources,
including the same docs.comfy.org guide Prompt 12 used, whose second
link (the distilled variant) was missed the first time. No model
switch is needed for this experiment.

### Mechanism

`ReferenceLatent` — a standard, core ComfyUI node (by ComfyAnonymous,
the core maintainer, not a third-party custom node) — is the
documented mechanism: encode the reference image through the model's
own VAE, feed that latent plus the prompt's own text conditioning into
`ReferenceLatent`, and use its output as the sampler's positive
conditioning instead of the plain text conditioning. Confirmed
additive: bypassing `ReferenceLatent` is documented as producing plain
T2I behavior from the same graph.

### The new workflow file

`workflows/flux2_klein_reference_4b.json` — see that file's own
`README-reference-conditioning.md` for its full provenance. In short:
constructed by extending the proven `flux2_klein_t2i_4b.json` graph
with exactly three new nodes (`ref:1` LoadImage, `ref:2` VAEEncode,
`ref:3` ReferenceLatent) and one redirected wire (`CFGGuider`'s
`positive` input now points at `ref:3` instead of the plain
`CLIPTextEncode` output). Every other node, model filename, and
setting is byte-identical to the T2I workflow. Unlike every other
workflow file in this directory, this one has **not** been proven on
real hardware — the first real run against it is the validation step,
not a formality.

### Architecture

`ComfyUIImageProvider.generate_reference_conditioned_image(prompt,
reference_image_path, width, height)` — a new capability on the
concrete provider class only; `ImageProvider.generate_image()`'s
generic signature is completely unchanged (verified via a dedicated
test). `products/chess2fight/rendering/reference_continuity_acceptance.py`
orchestrates the experiment: shot 0 through the real, unchanged
`RenderPipeline.render()`; shots 1/2 directly through the new provider
capability, both referencing the identical anchor path (verified
directly, not just asserted — a dedicated test confirms shot 2 never
receives shot 1's own output as its reference).

### Reference-edit prompt contract

`compose_reference_edit_prompt()` in `prompt_generator.py` builds an
explicit PRESERVE/CHANGE ONLY structure, reusing `_character_clause`
and `_camera_clause` directly (not reimplemented) so the preserved
identity text is always textually identical to what the T2I anchor
prompt itself used.

### Next paid command

```bash
export IMAGE_PROVIDER=comfyui
export ANIMATION_PROVIDER=comfyui
export COMFYUI_BASE_URL=http://127.0.0.1:8188

python scripts/render_reference_continuity_acceptance.py --sample
```

Generation-count contract: 1 T2I anchor + 2 reference-conditioned FLUX
+ 3 Wan = 6 ComfyUI jobs, 1 local concatenation — identical to Prompt
11/12's own contract. No new model files required — same distilled 4B
stack as the T2I workflow.

## Reference-latent method sweep (Sprint 4 Prompt 16)

Following the Prompt 15.1 live GPU result (character identity strong,
but timeline shot 2 still produced a duplicated diagonal spear/polearm
alongside the correct dragon-headed halberd — RGB SSIM anchor→shot1
0.951, anchor→shot2 0.799, shot1→shot2 0.813), this experiment isolates
`reference_latents_method` for that one shot only. The production
workflow implicitly uses ComfyUI's own current FLUX.2 default,
`"index"` — confirmed via research, not assumed — so Prompt 13 through
15.1's entire live history is the "index" control. Three new,
experimental-only workflow files add `FluxKontextMultiReferenceLatentMethod`
on both conditioning branches (see
`workflows/README-reference-method-sweep.md`): `offset`, `uxo/uno`,
`index_timestep_zero`. The production `flux2_klein_reference_4b.json`
is verified byte-for-byte unchanged (checksum confirmed in this
prompt's own test suite). Not yet run on real hardware — the next paid
run is exactly 3 ComfyUI jobs (one per candidate), reusing the same
anchor, prompt, and seed (981216397) as the already-paid Prompt 15.1
shot-2 result, which is never regenerated.

## Production reference method decision record (Sprint 4 Prompt 18)

**Decision: `reference_latents_method = "offset"` is now Halisako's
explicit production reference-conditioning method.** Not an implicit
ComfyUI default — set explicitly on both conditioning branches in
`flux2_klein_reference_4b.json`, which is no longer byte-identical to
its Sprint 4 Prompt 16 form (checksum
`738ad1818a72a2ac21c5f7ddf69e23c7ead867515a609b3520e07a6c6fe14a9b`
before this promotion; `b2ff7c9e1024abc76361c363601c68870148f9924dc3ca7c022c81ecfdb10b3d`
after).

**Why "index" (the prior implicit default) was rejected**: real GPU
evidence (Prompt 16, timeline shot 2) showed strong character identity
but a duplicated diagonal polearm alongside the correct dragon-headed
halberd — an equipment-topology failure — plus (per Prompt 14/15.1's
own earlier evidence) excessive anchor/composition lock under a shared
seed.

**Why "offset" was selected**: real GPU evidence across both known
reference-conditioned shots. **Correction (Sprint 4 Prompt 18.1)**:
the shot-2 SSIM values originally recorded here had been mixed up with
shot 1's own — independently recomputed from the saved live evidence:
- Shot 2 (Prompt 16): duplicate polearm removed, coherent single
  weapon, identity/arena/style retained, useful shot variation
  retained (RGB SSIM anchor→index 0.799134, anchor→offset 0.750260,
  index→offset 0.677935 — meaningful variation from index, not simply
  reverting to anchor lock).
- Shot 1 (Prompt 17): fighter identities, wardrobe/armor, and the
  dragon weapon's singular coherent form all preserved; twin-weapon
  configuration retained; more variation from anchor than index (RGB
  SSIM anchor→index 0.950738, anchor→offset 0.761308, index→offset
  0.774020).

SSIM here is supporting evidence for relative image variation, not a
proof of visual quality — the qualitative visual review (identity,
equipment coherence, arena/style) is what actually grounds the
decision.

The other two candidates tested in Prompt 16 (`uxo/uno`,
`index_timestep_zero`) both showed major identity/equipment/
environment drift or mutation and were not selected.

**This does not prove full video continuity is solved.** No Wan
animation or video concatenation has been evaluated with this method;
weapon/equipment coherence across exactly two isolated reference shots
is what's been validated, not a complete rendered fight.

**Next acceptance gate (documented, not executed by this prompt)**:
canonical T2I anchor + two derived-seed FLUX offset reference shots +
Wan I2V for all three shots + VideoBuilder concatenation. See this
prompt's own deliverable report for the exact contract.
