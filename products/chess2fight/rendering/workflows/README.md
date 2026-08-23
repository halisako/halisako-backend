# Wan 2.2 TI2V-5B ComfyUI workflow

> **PARTIALLY SUPERSEDED (Sprint 4 Prompt 8, corrected Prompt 10):** the
> `wan22_i2v_5b.json` file in this directory was replaced in Sprint 4
> Prompt 8 with a newer, separately-validated RunPod RTX 4090 run —
> 832×480, 8fps, 17 frames, 8 steps — superseding the 640×352/24fps/
> 49-frame values this document originally described. See
> `VALIDATED-SETTINGS.md` for the current values. The node map below
> (which nodes exist and what each is for) is still accurate — only
> the specific numbers, and the fps row's own claim, were stale; both
> are corrected inline below rather than rewriting this whole document.

`wan22_i2v_5b.json` in this directory is the **experimentally-validated**
ComfyUI API-format workflow for Wan 2.2 TI2V-5B image-to-video generation.
It was supplied (not authored or guessed) as part of Sprint 4 Prompt 4,
after being manually run successfully in ComfyUI on:

```
NVIDIA RTX 4090, 24 GB VRAM, CUDA 12.8, PyTorch 2.10.0+cu128
```

producing a real 640x352, 24fps, 49-frame (2.04s) MP4 with five distinct
sampled-frame hashes -- confirmed genuine generated motion, not a static
frame repeated. **(That specific run is now historical — see the notice
above; the file at `wan22_i2v_5b.json` itself no longer matches these
exact numbers.)**

This supersedes Prompt 3's version of this README, which explained why
*no* workflow file existed yet. That gap is now closed for the workflow
structure itself; see this feature's engineering report for what's still
unverified (the HTTP-driving code has not been exercised against a real
server -- no ComfyUI installation exists in the environment this was
built in).

## Model files this workflow requires

```
wan2.2_ti2v_5B_fp16.safetensors        (diffusion model, node 37)
umt5_xxl_fp8_e4m3fn_scaled.safetensors (text encoder, node 38)
wan2.2_vae.safetensors                 (VAE, node 39)
```

## Node map -- what `core/animation_providers/comfyui.py` injects where

| Node ID | `class_type`              | Purpose                            | Injected from `AnimationInstruction`                       |
|---------|----------------------------|-------------------------------------|--------------------------------------------------------------|
| 56      | `LoadImage`                | Reference image                     | Uploaded filename (from `source_image_path`)                 |
| 6       | `CLIPTextEncode`           | Positive prompt                     | `prompt`                                                      |
| 7       | `CLIPTextEncode`           | Negative prompt                     | `negative_prompt`, **only if provided** -- see below          |
| 3       | `KSampler`                 | Seed                                 | `seed`, or a deterministic fallback derived from `prompt`      |
| 55      | `Wan22ImageToVideoLatent`  | Width / height / frame length        | `width`, `height` (normalized), `duration_seconds` -> frames    |
| 57      | `CreateVideo`              | FPS (currently 8 in the active workflow, see VALIDATED-SETTINGS.md) | **is** injected — `instruction.fps` or `settings.comfyui_default_fps` (Sprint 4 Prompt 6 fix; this row previously said "not injected," which stopped being true once that fix landed) |
| 58      | `SaveVideo`                | Output node                          | not injected -- read from, not written to                       |

Nodes 37/38/39 (model/VAE/text-encoder loaders) and node 48
(`ModelSamplingSD3`, `shift=8`) are left as workflow configuration and are
never modified per-request, per the task's own guidance.

**Negative prompt handling, specifically:** this workflow ships with a
real, substantial, tuned negative prompt on node 7 (a long list of
artifacts to avoid). The provider only overwrites it when
`AnimationInstruction.negative_prompt` is explicitly set -- an unset
negative prompt leaves the workflow's own tuned value in place rather
than blanking it to an empty string. (Prompt 3's version of this logic
did the latter; that was a bug, now fixed -- see the engineering report.)

## Constraints applied before injection

- **Width/height** are rounded to the nearest multiple of 16 --
  `Wan22ImageToVideoLatent`'s own documented requirement (confirmed
  directly from ComfyUI's official node reference page for this exact
  node).
- **Frame count** (`length`) is snapped to the nearest value satisfying
  `(length - 1) % 4 == 0` (1, 5, 9, ..., 49, ...). This is inferred from
  converging evidence -- the closely-related `WanImageToVideo` node's own
  ComfyUI source defines `length` with `step=4, min=1`; multiple
  independent tutorials for `Wan22ImageToVideoLatent` specifically cite
  default lengths of 41/81/121, all matching the pattern; and this
  workflow's own validated 49-frame value matches it too (49 = 4x12+1) --
  but it is not confirmed from `Wan22ImageToVideoLatent`'s own source
  directly, since no ComfyUI installation was available to check
  against. See `core/animation_providers/comfyui.py`'s module docstring
  for the full sourcing notes.

## Known open risk: SaveVideo output resolution via `/history`

A ComfyUI community forum thread (`forum.comfy.org/t/comfy-ui-api-automation/4251`,
unanswered as of when this was written) reports that `SaveVideo`'s output
sometimes doesn't appear in `/history/{prompt_id}` via polling, even
though the video was generated and saved successfully. This provider's
`_extract_output_reference` checks multiple plausible output keys
(`images`, `videos`, `gifs`) and falls back to scanning every node's
output, but this is a real, documented, unresolved risk that cannot be
fully ruled out without a live ComfyUI instance to test against.
