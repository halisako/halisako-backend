# FLUX.2 Klein 4B (distilled) — reference-conditioned image editing

`flux2_klein_reference_4b.json` in this directory — Sprint 4 Prompt 13,
topology corrected Sprint 4 Prompt 13.1.

## Provenance — read this before trusting this file the way you can
## trust the other workflows in this directory

Every other workflow file in this directory (`flux2_klein_t2i_4b.json`,
`wan22_i2v_5b.json`, `wan22_t2v_5b.json`) was **supplied directly** —
a real, exported API-format JSON from an actual validated ComfyUI
session on real hardware. This file is **not that**. No real
reference-conditioned run has been performed against a live ComfyUI
server for Halisako as of Sprint 4 Prompt 13.1.

This file was constructed by directly extending the known-valid,
already-proven `flux2_klein_t2i_4b.json` graph (same model loaders,
same sampler chain, same scheduler, same width/height/seed nodes —
byte-identical except where noted below), based on current research
into FLUX.2 Klein's official reference-conditioning support:

- A dedicated, official "Flux.2 [Klein] 4B Distilled: Image Edit"
  ComfyUI workflow exists (docs.comfy.org's own FLUX.2 Klein guide
  links both a base-model and a distilled-model image-editing
  workflow — the distilled one was missed in Sprint 4 Prompt 12's
  earlier research pass; this task's own explicit instruction not to
  assume the base model was necessary led to finding it in Prompt 13).
- `ReferenceLatent` is confirmed as a **standard, core ComfyUI node**
  (authored by ComfyAnonymous, the core ComfyUI maintainer — not a
  third-party custom node).
- **Sprint 4 Prompt 13.1 correction**: fresh research into the exact
  official topology found the reference latent is blended into
  **both** the positive and negative conditioning branches, not just
  the positive one — Prompt 13's own first version of this file only
  did the latter. A directly relevant source (a Flux.2 Klein 9B KV
  Image Edit workflow writeup) states explicitly: "the subgraph
  Reference Conditioning... encodes the primary image with the VAE and
  blends its latent features into the positive **and negative**
  conditionings" — with the negative branch's own neutral signal
  (`ConditioningZeroOut`) then also passed through its own
  `ReferenceLatent`, using the *same* encoded reference latent as the
  positive branch. Corrected below.
- Reference support remains documented as **additive**: bypassing the
  `ReferenceLatent` node(s) is described elsewhere as producing plain
  text-to-image behavior from the same underlying graph — supporting
  the design choice here (extend the T2I graph minimally, don't build
  a parallel one).

## Exactly what changed vs. `flux2_klein_t2i_4b.json`

Every node ID, model filename, and setting from the validated T2I
workflow is unchanged. Four new nodes were added (namespaced `ref:*`
to keep them visually distinct from the T2I graph's own `77:*`
subgraph-derived IDs), and two existing wires were redirected:

| Node ID | `class_type`     | Purpose                                                                |
|---------|-------------------|-------------------------------------------------------------------------|
| `ref:1` | `LoadImage`       | The reference image — the fight's canonical visual anchor                 |
| `ref:2` | `VAEEncode`       | Encodes it through the same VAE (`77:89`) the T2I graph already loads     |
| `ref:3` | `ReferenceLatent` | Positive branch: combines `ref:2`'s latent with `77:92`'s (the prompt's) conditioning |
| `ref:4` | `ReferenceLatent` | Negative branch: combines the SAME `ref:2` latent with `77:91`'s (`ConditioningZeroOut`) output |

The two redirected wires: node `77:90` (`CFGGuider`)'s `positive` input
now points at `ref:3` (not `77:92` directly), and its `negative` input
now points at `ref:4` (not `77:91` directly). `ref:3` and `ref:4` both
receive the identical `ref:2` latent — the single canonical anchor,
never two different images. Everything downstream of `77:90` (the
sampler, scheduler, VAE decode, save) is untouched.

```
LoadImage (ref:1) -> VAEEncode (ref:2) -> reference latent
                                              |         |
CLIPTextEncode (77:92) -----------------------+         |
       |                                      v         |
       |                          ReferenceLatent (ref:3)|
       |                                      |          |
       |                                  positive        |
       |                                                  |
       +--> ConditioningZeroOut (77:91)                   |
                       |                                   v
                       +----------------------> ReferenceLatent (ref:4)
                                                       |
                                                    negative

positive + negative -> CFGGuider (77:90) -> existing validated sampler path
```

## What this means for the first paid run

Do not trust this file's exact node IDs, parameter names, or graph
shape the way the T2I/Wan workflows can be trusted — those were
proven; this is a well-grounded, best-effort construction from
documented patterns, not yet fire-tested. A workflow structure
mismatch here (an expected node ID missing) raises
`ComfyUIImageRequestError`, the same internal exception type
`generate_image` already uses — caught and re-raised as
`ImageProviderError` by `generate_reference_conditioned_image`, same
as every other failure mode. Should be treated as an expected
possibility for this file specifically, not a bug in the request
itself. The first real run against this file **is** the validation
step. Before that run, `validate_reference_workflow_topology()`
(Sprint 4 Prompt 13.1, in `acceptance_preflight.py`) statically
confirms this exact graph shape (VAEEncode present, both
ReferenceLatent nodes present and sharing one latent source,
CFGGuider's positive/negative both pointing at a ReferenceLatent) —
zero network calls, safe in ordinary tests.

## Model files and weight-file clarification (Sprint 4 Prompt 13.1)

Three distinct claims, kept explicitly separate — conflating them was
flagged as a documentation risk worth correcting precisely:

**A. Model architecture/capability** — Black Forest Labs documents
that the FLUX.2 Klein 4B architecture itself (both distilled and base
variants) supports text-to-image, image editing, and multi-reference
editing. This is a capability claim about the model family, not about
any specific weight file.

**B. The official ComfyUI distilled image-edit template's advertised
weight** — the current official "Flux.2 [Klein] 4B Distilled: Image
Edit" ComfyUI template specifically names
`flux-2-klein-4b-fp8.safetensors` (an fp8-quantized weight file).

**C. Halisako's actual experimental weight** —
`flux-2-klein-4b.safetensors` (not explicitly confirmed fp8 or a
different precision) — the exact file already live-proven for
Halisako's T2I path (Sprint 4 Prompt 6, real RTX 4090 run). **The
reference/edit path with this exact weight file is not yet
live-proven** — only the model *architecture's* documented capability
(claim A) supports the expectation that it should work, not a
confirmed run with this specific file against the official edit
template.

This file's node `77:87` (`UNETLoader`) still names
`flux-2-klein-4b.safetensors` (claim C, unchanged from the T2I
workflow) — not `flux-2-klein-4b-fp8.safetensors` (claim B) — since
this task does not switch models. No additional model file is
currently believed necessary, but this has not been confirmed by
source-level compatibility analysis beyond the architecture-level
claim (A) and is a genuine open question for the first real run, not
a settled fact.

```
flux-2-klein-4b.safetensors        (diffusion model, node 77:87 — Halisako's own experimental weight, claim C above)
qwen_3_4b.safetensors              (text encoder, node 77:88)
flux2-vae.safetensors              (VAE, node 77:89 — also encodes the reference)
```
