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

## Wan 2.2 TI2V 5B — image-to-video

| Setting | Value |
|---|---|
| Diffusion model | `wan2.2_ti2v_5B_fp16.safetensors` |
| Text encoder | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| VAE | `wan2.2_vae.safetensors` |
| Resolution | 640 × 352 |
| Frames | 49 (≈2.04s at 24fps) |
| FPS | 24 |
| Steps | 20 |
| CFG | 5 |
| Sampler | uni_pc |
| Scheduler | simple |
| Denoise | 1 |
| Workflow file | `products/chess2fight/rendering/workflows/wan22_i2v_5b.json` |

The workflow file was updated in this task to the version actually used
in the validated FLUX→Wan run (the `LoadImage` node's default value
references the real generated keyframe filename from that run,
`halisako_flux2_keyframe_proof_01.png.png` — this default is always
overwritten by the provider's own upload step before submission, never
read as-is; it's meaningful only as evidence of what was actually run,
not as a required input filename).

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
