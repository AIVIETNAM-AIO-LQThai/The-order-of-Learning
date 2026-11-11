# Medium World — Refined Frozen Shape Probes

This package **extends the existing representation-diagnostics experiment on `experiment/phase-aware-lr`**. The original trained ViT, dataset, 30 checkpoints, and all earlier diagnostic files remain unchanged. It does not add a new curriculum, change the architecture, or train the ViT again.

## What changed

The original Shape probe spatially averaged visual tokens, potentially discarding object geometry. The refined analysis fits a **matched 2×2 set of linear Shape decoders**:

| Decoder training scenes | 5×5 mean-pooled tokens | 5×5 spatially ordered tokens |
|---|---|---|
| Original training distribution | Mean / original | Spatial / original |
| Training scenes recolored with balanced Shape×Color weights | Mean / balanced | Spatial / balanced |

- All decoders use **exactly the same fixed 5×5 object-centered sampling grid**. No Shape mask or target label is used for crop geometry.
- Recoloring modifies only **probe training images**, never model weights. All 20 Shape×Color combinations receive equal total decoder-training weight; the same original train scenes supply the geometry.
- A random frozen encoder, a shuffled-Shape-label decoder, and a raw foreground-silhouette decoder are diagnostic controls.
- Train/probe-evaluation scenes remain separate. Both ID-dev and OOD-dev are reported. **OOD-dev combinations are unseen by the ViT but not necessarily unseen by the recolored probe**, so the latter cannot establish novel-combination generalization by the probe.
- All **5 seeds × 2 curricula × 3 checkpoints** are enabled. Only the small diagnostic decoders are fitted.

## Files

- `src/medium_visual_world/representation_probing.py` — original code plus refined extraction, recoloring, balancing, and probe functions.
- `notebooks/medium_world/representation_probe_refinement.ipynb` — **run this first**; uses the 30 saved checkpoints and writes new `refined_probe_*` results.
- `notebooks/medium_world/representation_probing_analysis.ipynb` — original analysis plus refined five-seed summaries, paired comparisons, Shape×Color tables, and a plot.
- `notebooks/medium_world/representation_evidence_analysis.ipynb` — retains old reproducible figures and adds figures/tables about refined probing and trained-versus-random feature accessibility.
- `notebooks/medium_world/representation_probing.ipynb` — the original behavioral/Network Dissection/SpRAy experiment, with the project-root line corrected. **No need to rerun it** if its existing outputs are complete.

## Run

Extract the ZIP into the repository root, replacing files with the same paths. Open notebooks with the working directory set to `notebooks/medium_world`. They use exactly:

```python
PROJECT_ROOT = Path.cwd().resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

Run **Restart Kernel and Run All** in this order:

1. `representation_probe_refinement.ipynb`
2. `representation_probing_analysis.ipynb`
3. `representation_evidence_analysis.ipynb`

Do not rerun checkpoint training or original probing. Results are saved alongside the existing experiment under `outputs/medium_world/experiments/representation_diagnostics/five_seeds/` and its `evidence_tables/` and `evidence_figures/` subfolders. No environment provenance JSON files are created; use your existing repository README for machine information.

## Interpretation

The **matched new 2×2 probes**, not the old ROI mean probe, isolate the effects of retaining spatial layout and changing probe-training color associations. A trained-minus-random gain suggests ViT training made Shape information more accessible *to that decoder*. It is not evidence that the original prediction head uses that information causally. The spatial readout has more input dimensions, so stronger accuracy can also reflect greater decoder capacity; the random-encoder, silhouette, and shuffled-label controls help contextualize that.

## Testing completed before delivery

- All notebooks structurally validated and all code cells parsed.
- Refined extraction, recoloring, balanced sample weights, three frozen feature stages, linear decoding, and pixel silhouette extraction executed against synthetic scenes and a compatible PyTorch model.
- The complete refinement notebook's run and CSV-saving flow executed against a small synthetic checkpoint/data fixture.
- The added sections of both analysis notebooks executed against a synthetic **30-checkpoint file inventory** (five seeds × two curricula × three steps).

These are implementation tests. The complete trained-checkpoint analysis on your CUDA device has not been executed here, and **no new empirical Shape findings are claimed**.
