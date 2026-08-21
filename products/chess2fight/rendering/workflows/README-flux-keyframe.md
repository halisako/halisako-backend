# FLUX keyframe ComfyUI workflow — not present in this repository

> **SUPERSEDED (Sprint 4 Prompt 6):** the real, experimentally-validated
> workflow now exists at `flux2_klein_t2i_4b.json`, and the provider
> targets its exact node IDs — see `VALIDATED-SETTINGS.md` for the
> current state. This document is kept for its research record (FLUX.1
> vs FLUX.2, the CLIPLoader ambiguity, the resolution reasoning) — all
> now resolved, but the reasoning trail is still accurate and may be
> useful context for future workflow changes.

`core/image_providers/comfyui.py`'s `ComfyUIImageProvider` expects a
ComfyUI **API-format** workflow JSON at the path configured by
`settings.comfyui_image_workflow_path` (default:
`products/chess2fight/rendering/workflows/flux_keyframe.json`).

**That file does not exist, and this document is here instead of a
fabricated one.** Unlike the Wan 2.2 5B animation workflow (Prompt 4),
no FLUX workflow was supplied with this task, and this environment has
no ComfyUI installation, no GPU, and no network path to model-hosting
domains (unchanged since the Prompt 3 environment audit — see
`core/animation_providers/comfyui.py`'s docstring). Assembling a
JSON file that *looks* like a real, exported workflow — with specific
node IDs and confident-looking structure — without anything to
validate it against would misrepresent research as verified fact.

## What was actually researched, and how confident each part is

**Model choice: FLUX.2 [klein]** (the 4B or 9B distilled variant),
over FLUX.1. Reasoning, not just a guess:
- FLUX.2 [klein] is reported (multiple independent sources, current as
  of mid-2026) to run in ~13GB VRAM with sub-second-to-few-second
  generation on consumer GPUs — well within a 24GB RTX 4090's budget,
  leaving headroom for the same machine to also run Wan.
  Apache 2.0 licensed (the first fully commercial-friendly FLUX line),
  which matters for a product, not just a research exercise.
- This directly matches the task's own stated priorities: "good visual
  quality, manageable VRAM, reliable ComfyUI support, fast enough for
  MVP keyframe generation."
- **Not certain**: FLUX.1 dev/schnell remains viable and more
  extensively documented; if FLUX.2 [klein] turns out to be
  unavailable or underperforms on quality, FLUX.1 is the fallback with
  no code changes needed beyond swapping the workflow file and model
  filenames the workflow's loader nodes reference.

**Node structure**: `UNETLoader` (or `Load Diffusion Model`) + a CLIP
loader + `VAELoader` + `CLIPTextEncode` (prompt) + a latent-image node
(width/height) + a sampler (seed) + `SaveImage`, matching the pattern
multiple independent sources describe consistently for FLUX generally.

**Genuinely unresolved, found while researching, not glossed over**:
one real FLUX.2 [klein] workflow JSON snippet found during research
uses a single `CLIPLoader`, while other sources describe FLUX.2 using
`DualCLIPLoader` (a Qwen or Mistral-based encoder, still "dual" in
some configurations). This wasn't a case of finding one clear answer —
different real sources disagree, likely reflecting different FLUX.2
sub-variants or ComfyUI versions. `_PARAMETER_NODE_MAP` in
`comfyui.py` deliberately doesn't hard-code which loader type is used,
since the provider never touches the loader nodes at all (they're
static workflow configuration, like Wan's UNETLoader/CLIPLoader/
VAELoader in Prompt 4) — this ambiguity affects workflow *authoring*,
not this provider's *injection logic*, so it doesn't block the code
here from being correct, only the workflow file's own construction.

**Resolution**: default `1280×704` — exactly 2x the experimentally
validated Wan 2.2 5B resolution (640×352, Prompt 4), so a FLUX keyframe
needs no cropping or distortion before Wan conditioning. Divisible by
16 (Wan's own confirmed alignment rule) and independently cited during
Prompt 4's own research as a commonly-used Wan resolution — not just
derived by doubling.

**Sampler settings**: not hard-coded into the provider at all — steps/
CFG/sampler/scheduler are workflow configuration (matching how Wan's
KSampler.steps/cfg/sampler_name/scheduler are left untouched in Prompt
4), left for whoever authors the real workflow to tune. Sources
consulted suggest steps≈4 and CFG≈1.0–1.5 for FLUX.2 [klein]'s
distilled variant specifically (few-step, guidance-distilled models
behave very differently from FLUX.1 dev's typical 20–30 steps/CFG
3–5) — worth setting correctly in the real workflow, but not something
this provider needs to know about or inject.

## What's needed to supply the real file

1. A working ComfyUI installation with a FLUX.2 [klein] (or FLUX.1)
   checkpoint, text encoder(s), and VAE installed.
2. Build (or obtain) a working text-to-image workflow in ComfyUI's UI
   using that model; confirm it actually generates images manually.
3. Rename the nodes that should receive dynamic values to match the
   titles in the table below (or edit `_PARAMETER_NODE_MAP` in
   `comfyui.py` to match whatever titles the real workflow uses).
4. Export via **"Save (API Format)"** to
   `products/chess2fight/rendering/workflows/flux_keyframe.json`.

| Provider value | Expected node title | Input key |
|---|---|---|
| `prompt` | `Halisako: Positive Prompt` | `text` |
| seed (derived from prompt) | `Halisako: Sampler` | `seed` |
| `width` (normalized to nearest ×16) | `Halisako: Latent Image` | `width` |
| `height` (normalized to nearest ×16) | `Halisako: Latent Image` | `height` |

Negative prompt is never injected by this provider — the existing
`ImageProvider.generate_image(prompt, width, height)` interface has no
parameter for one. If the real workflow has a negative-prompt node,
whatever it ships with is left untouched.

If a target node/title isn't found in the loaded workflow, the
provider logs a warning and leaves that input at the workflow's own
default — same behavior as the Wan provider, never a hard failure over
a missing optional node.

## Output key uncertainty

`_VIDEO_OUTPUT_KEYS` (misleadingly named — kept consistent with the
animation provider's naming, since both scan the same candidate keys)
checks `"images"` first, since `SaveImage` is ComfyUI's standard,
built-in output node and has reported under that key consistently
across every source checked — meaningfully more certain than the
Wan animation provider's `SaveVideo` output-key situation, which has a
documented, unresolved community-reported gap (Prompt 4's report).
