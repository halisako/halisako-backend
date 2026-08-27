# Reference-latent method sweep — experimental workflow variants

Sprint 4 Prompt 16. Three files, all in this directory:

```
flux2_klein_reference_method_offset_4b.json
flux2_klein_reference_method_uxo_4b.json
flux2_klein_reference_method_index_timestep_zero_4b.json
```

## Provenance

Each is generated programmatically from the production
`flux2_klein_reference_4b.json` (verified byte-for-byte unchanged by
this task — see that file's own checksum, unmodified) by inserting
exactly two new nodes and rewiring exactly two existing inputs. No
other node, connection, or setting differs between these three files
or from the production workflow.

`FluxKontextMultiReferenceLatentMethod` is confirmed as a standard,
core ComfyUI node (`comfy_extras/nodes_flux.py`, not a third-party
custom node) that takes a `CONDITIONING` input and a
`reference_latents_method` combo value, and outputs a modified
`CONDITIONING` — a simple, two-input/one-output node, confirmed via
multiple independent sources including a real workflow JSON export
showing its exact input/output shape. The four supported values —
`offset`, `index`, `uxo/uno`, `index_timestep_zero` — are confirmed
identically across every source consulted.

## Exact topology change (identical across all three files)

```
positive:  77:92 -> ref:3 (ReferenceLatent) -> method:1 (FluxKontextMultiReferenceLatentMethod) -> 77:90.positive
negative:  77:91 -> ref:4 (ReferenceLatent) -> method:2 (FluxKontextMultiReferenceLatentMethod) -> 77:90.negative
```

`method:1` and `method:2` always carry the identical
`reference_latents_method` value within a single file — never mixed
between the two branches. Confirmed directly (not just constructed
correctly): each file's `method:1` and `method:2` nodes were checked
programmatically after generation.

| File                                                    | `reference_latents_method` |
|----------------------------------------------------------|-----------------------------|
| `flux2_klein_reference_method_offset_4b.json`             | `offset`                    |
| `flux2_klein_reference_method_uxo_4b.json`                 | `uxo/uno`                   |
| `flux2_klein_reference_method_index_timestep_zero_4b.json` | `index_timestep_zero`       |

The production workflow (`flux2_klein_reference_4b.json`, unmodified)
implicitly uses ComfyUI's own current FLUX.2 default —
`reference_latents_method="index"` — since it never explicitly sets
this value at all. Prompt 13/14/15/15.1's entire live GPU history is
therefore this implicit "index" method; there is no separate "index"
variant file here because the already-paid Prompt 15.1 shot-2 result
**is** that control, and this task's own explicit instruction is not
to regenerate it.

## What this means for the next paid run

Same caveat as every other workflow file in this directory that isn't
a directly-supplied, validated export: this is a well-grounded,
research-based construction, not yet fire-tested on real hardware. The
first real run against each of these three files is the validation
step for that file specifically.
