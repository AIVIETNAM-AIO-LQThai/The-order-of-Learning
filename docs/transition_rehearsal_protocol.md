# Transition rehearsal: shared-checkpoint protocol

**Project:** The Order of Learning · Medium World  
**Source curriculum:** `prereq_mixed_pair_focus`  
**Status:** Design frozen before training · Five paired seeds (42–46)

## Research question

When a ViT has recently trained on same-object Binding, does the **order of reintroducing simpler tasks** affect retention of Binding and of earlier skills? Compare an abrupt task switch, a gradual bridge, and greater *late* Binding exposure.

This is a continuation of the completed Mixed → Pair experiment, **not** a change in dataset or model. Keep its model `AttributeBindingViT(dim=48, patch_size=4, depth=1)`, training dataset (10,000 questions, 2,500 per task), ID-dev/OOD-dev, AdamW, batch size 50, LR `1e-4`, weight decay 0, gradient clipping and image augmentation. The notebook checks the original dataset file hashes before starting.

## Common parent

For each seed, train **once** with the exact initial blocks from `prereq_mixed_pair_focus`:

| Steps | Parent training | Example presentations |
|---|---|---:|
| 0–5,400 | Color, Shape, Conjunction mixed; 36 passes of each | 270,000 |
| 5,400–6,300 | Binding only; 18 passes | 45,000 |

At step **6,300**, save weights, **full AdamW state**, Python/NumPy/PyTorch CPU/CUDA RNG states, phase history, and aligned prediction traces. Reload those identical states independently for each suffix; check the shared parent weight SHA-256 for each seed.

## Three suffix strategies

One pass means each of that task's **2,500 distinct training questions** appears exactly once. Let **P** = Binding, **E** = each of Color, Shape and Conjunction.

| Strategy | 6,300–6,600 | 6,600–7,200 | 7,200–9,000 |
|---|---|---|---|
| `abrupt` | P 100% | P 100% | P 25%, E 25% each |
| `gradual_bridge` | P 100% | P 75%, E 8⅓% each | P 33⅓%, E 22.22% each |
| `late_binding` | P 100% | P 25%, E 25% each | P 50%, E 16⅔% each |

**Exact epoch allocations**, which take precedence over rounded percentages:

| Strategy | 6,300–6,600: 300 updates | 6,600–7,200: 600 updates | 7,200–9,000: 1,800 updates |
|---|---|---|---|
| `abrupt` | P6 | P12 | P9 + E9 each |
| `gradual_bridge` | P6 | P9 + E1 each | P12 + E8 each |
| `late_binding` | P6 | P3 + E3 each | P18 + E6 each |

For all three: suffix totals are **P27 and E9 each**, or 67,500 Binding presentations and 22,500 for each earlier task. Combined with their common prefix, that yields exactly **45 presentations of each individual training question**, i.e. 112,500 presentations per task, 450,000 total presentations, and **9,000 updates**. Within each task, the **individual-question order is unchanged across conditions**; the abrupt condition exactly reproduces the original baseline sequence. Random augmentation is sampled from an identical state at branching, not coupled per example across branches.

### What is — and is not — isolated

- `gradual_bridge − abrupt`: total exposure and starting state are matched; the suffix schedule differs. If the bridge does better, the result supports its **whole task-order intervention**, not smoothness alone.
- `gradual_bridge − late_binding`: both introduce earlier tasks by step 6,600, but the late-Binding control concentrates more Binding practice near the final evaluation. This tests two timing patterns, **not a perfect single-variable smoothness manipulation**.
- Step **7,200 is not Binding-exposure matched** across these strategies (P18 / P15 / P9 suffix epochs for abrupt / gradual / late). Treat this as a diagnostic stage boundary; compare fixed-final outcomes at 9,000 when exposure totals match.

## Evaluation and decision rules

- **Primary:** paired-seed difference in final step-9,000 **ID-dev Binding** between `gradual_bridge` and `abrupt`.
- **Secondary:** final OOD-dev Binding; `late_binding` comparisons; ID-dev earlier-task retention; Binding AULC from step 6,300; positive recall, negative specificity, predicted-positive rate, and per-question prediction changes.
- Evaluate every **100 updates** (0–9,000); the notebook records 91 checkpoints and uses identical question IDs.
- Report the exact transitions at steps **6,300, 6,600, 7,200, 9,000**. Examine changes over 6,300→9,000, 7,200→7,300, and 7,200→9,000; note the exposure difference at 7,200.
- Compare seeds as paired replicates; report their differences, mean, sample SD, and paired 95% t-interval (n=5).
- OOD-dev informs interpretation **only**, not training-schedule tuning. Preserve ID-test/OOD-test without new evaluation unless predeclared.

## Implementation, outputs and checks

Training: `notebooks/medium_world/transition_rehearsal.ipynb`; analysis: `notebooks/medium_world/transition_rehearsal_analysis.ipynb`. Schedule helper: `src/medium_visual_world/transition_control.py`.

Outputs: `outputs/medium_world/experiments/transition_rehearsal/five_seeds/`, with `schedule_comparison.json`, 15 branch `.npz` traces, five `prefix_seed_*.npz` traces, checkpoint `.pt` files, figures, and CSV tables. The Git ignore rule already excludes `outputs/**/experiments/**/checkpoints/`; **do not commit those checkpoints**.

Before interpreting outcomes verify: exact per-question exposure; common first 6,300 steps; identical parent SHA per seed; complete traces; correct dataset hashes; and equal final task presentation counts. The results are not precomputed.
