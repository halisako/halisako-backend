# ComfyUI providers — validated runtime requirements

This documents the exact, experimentally-validated configuration behind
`ComfyUIImageProvider` and `ComfyUIAnimationProvider` (Sprint 4 Prompts
4–6). Both were proven end to end on a real GPU: a FLUX-generated
keyframe was fed directly into the Wan provider and produced real,
verified motion. This development environment does not have GPU/ComfyUI
access itself — see `core/animation_providers/comfyui.py`'s own
environment audit for that — so everything below reflects the supplied,
externally-validated artifacts, not anything run in this environment.

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
