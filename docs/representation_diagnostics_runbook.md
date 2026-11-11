# Visual Representation Diagnostics — Full Five-Seed Experiment

Add this ZIP **at the root of the existing `The-order-of-Learning` repository** on branch `experiment/phase-aware-lr`. The ZIP adds a diagnostic module and three notebooks; it does **not** replace the dataset, existing model, or LR-reset experiments.

## Exact run scope

- **5 seeds:** 42, 43, 44, 45, 46 (all enabled by default).
- **2 curricula:** `prereq_mixed_pair_focus` and `mixed` (Constant LR, unchanged model).
- **3 checkpoints per trained model:** updates 5,400, 7,200, 9,000.
- **10 training runs and 30 model checkpoints** (not 30 separate trainings).
- **30 trained-model diagnostic evaluations**, with random-encoder and shuffled-label controls.
- **Data:** the existing immutable Medium World train, ID-dev, OOD-dev splits. ID-test and OOD-test remain untouched.

No pilot stage is required or configured. The preflight audit below is **not training**.

## Run order

1. From the repository root run `python scripts/medium_world/audit_representation_probing.py`. This verifies frozen manifests, oracle labels, exact image rendering, scene separation, and diagnostic utilities.
2. Open `notebooks/medium_world/representation_checkpoints.ipynb` in its own directory and **Run All**. It reproduces the 10 Constant-LR training runs and saves the 30 checkpoints. The last cell compares their predictions against archived LR-reset traces. If some exact old trajectories differ, the audit records the difference rather than claiming exact reproduction.
3. Open `notebooks/medium_world/representation_probing.ipynb` and **Run All**. It evaluates every required checkpoint, saves CSV/NPZ evidence separately for each seed, schedule, checkpoint, and split, and checks output completeness.
4. Open `notebooks/medium_world/representation_probing_analysis.ipynb` and **Run All**. This analyzes saved evidence only (no training), with subgroup support, controls, confusion matrices, paired five-seed summaries, concept alignment, and explanation clusters.

All new results are isolated in `outputs/medium_world/experiments/representation_diagnostics/five_seeds/`. Do not rename or overwrite existing LR-reset files. If a checkpoint group has only some of its 3 files, the checkpoint notebook stops; restore the missing files or deliberately remove the group's incomplete checkpoints before rerunning.

## Files

| Path | Function |
|---|---|
| `src/medium_visual_world/representation_probing.py` | Diagnostic oracle/questions, counterfactuals, probes, concept masks, spectral explanations |
| `scripts/medium_world/audit_representation_probing.py` | Preflight integrity + random-model sanity checks; **no training** |
| `notebooks/medium_world/representation_checkpoints.ipynb` | 5-seed Constant-LR checkpoint reproduction and archived-trace audit |
| `notebooks/medium_world/representation_probing.ipynb` | Five-seed same-scene, counterfactual, probe, Network Dissection and SpRAy-style runs |
| `notebooks/medium_world/representation_probing_analysis.ipynb` | Read-only analysis, subgroup and paired-seed summaries, visual inspection |
| `docs/representation_probing_protocol.md` | Research design, controls, limitations, interpretation decisions |

Requires the same environment as the previous notebook plus `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `Pillow`, and `nbformat`. The checkpoints and diagnostics are computationally substantial; CUDA is recommended. To use CPU, change `DEVICE = 'cuda'` to `DEVICE = 'cpu'` in both training and probing notebooks.

## Methodological precision

- Question grid asks all four question types about the same scene and target combination. **Single-attribute questions repeat across pairs**; the analysis deduplicates them per seed/schedule/checkpoint but keeps the full grid for logical relations. ID-dev extended questions are **not** the original ID-dev benchmark.
- Object probes use oracle object centers and **fixed square** spatial pooling. They measure feature accessibility, not deployable detection or causal reliance.
- ViT **Network Dissection-style** alignment thresholds individual patch channels on train scenes at **99% and 99.5% activation quantiles** and evaluates IoU with renderer masks on disjoint dev scenes. It is not the original Broden/CNN implementation.
- **SpRAy-inspired** clustering uses signed gradient × (image − background), not original Layer-wise Relevance Propagation (LRP). Cluster IDs are not comparable across seeds.
- Checkpoint reproduction is audited by exact question IDs, final predictions, and probabilities; GPU nondeterminism may prevent perfect replay.
- Five seeds give replication, but small differences and low-support subgroups remain exploratory. Do not interpret these as causal proof about internal mechanisms.

**No experimental findings are included; no full GPU training was executed by the assistant.**
