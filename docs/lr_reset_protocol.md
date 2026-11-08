# Phase-Aware Learning Rate Restarts — Experimental Protocol

**Status:** Completed — all 100 planned runs (four curricula × five LR policies × five seeds) were executed and analyzed. This document preserves the original experimental protocol; see [Phase-Aware LR Findings](lr_reset_findings.md) for outcomes and interpretation.

## Research questions

1. Does restarting an LR schedule at a **genuine task-distribution change** improve final in-distribution same-object Binding accuracy compared with a global LR schedule?
2. Does the gain (if any) generalize to held-out color–shape combinations, or reflect shifts in positive recall versus negative specificity?
3. Does restarting LR interfere with Color, Shape, or Conjunction performance after a transition? How quickly do previously correct predictions recover?
4. Are changes plausibly caused by **more LR-weighted updates** rather than by a scheduler-history reset? The current study compares *policies*, not isolated scheduler mechanics.

## Frozen setup

- Existing Medium World dataset: 10,000 training questions (2,500 per task), 1,024 ID-dev questions (256 per task), and 400 OOD-dev Binding questions.
- Existing `AttributeBindingViT` unchanged: dimension 48; 4×4 patches; depth 1. AdamW with weight decay 0; batch 50; gradient clip 1; existing image augmentations.
- **Five paired seeds:** 42–46. Every run uses 9,000 updates and exactly 45 presentations of **every training question**. No best-checkpoint selection: assess **step 9,000**.
- Keep model initialization and per-seed question orders matched across LR policies. Evaluate every 100 global steps, genuine boundaries, and extra policy monitoring points. Bind all paired analyses to the *same seed, curriculum, and question IDs*.
- **Primary outcome:** final ID-dev Binding accuracy. **Secondary:** final OOD-dev Binding accuracy; final Shape and Conjunction; Binding AULC over the shared global 100-step grid; per-question error dynamics and confidence proxies; held-out pair subgroup accuracy.
- OOD-dev has been consulted extensively in earlier investigations. Treat it as exploratory rather than a sealed final test.

## Curricula retained

Schedules are produced using existing `make_spiral_blocks` and frozen within-task streams. Adjacent blocks sharing identical task proportions are merged for LR restarts.

| Curriculum | Real task-distribution transitions |
|---|---|
| Prerequisite-Mixed → Pair Focus | 5,400; 7,200 |
| Staged Rehearsal | 3,000; 4,950; 7,200 |
| Scaffolded Spiral | 2,800; 4,450; 6,100; 7,200 |
| Fully Mixed | **None** |

**No LR restart at 6,300** for Prerequisite-Mixed: `pair_early` and `pair_focus` both contain 100% Pair. Fully Mixed has no LR restart at 7,200, despite the `common_mixed` label.

## Learning-rate policies

| Policy | Initial behavior | At genuine transitions | Final Mixed transition |
|---|---|---|---|
| `constant` | `1e-4` for every update | No change | No change |
| `global_cosine` | Cosine, `1e-4` → `1e-6` over 9,000 updates | No change | No change |
| `reset_cosine` | Cosine within each true task-mixture segment, floor `1e-6` | New cosine cycle peaks at `1e-4` | New cycle peaks at `5e-5` |
| `global_plateau` | Overall ID-dev CE; factor `2/3`, patience 2, absolute threshold `0.001`, floor `1e-6` | Retain LR and plateau history | No change |
| `reset_plateau` | Same ID-dev CE and plateau settings | LR restores to `1e-4`, plateau best/bad count restart using boundary CE | LR restores to `5e-5` |

- For Cosine, the first optimizer update in a segment uses its peak and the last uses its floor. Global Cosine uses the same formula over the whole 9,000-update horizon.
- **Plateau monitoring:** Global checks at global updates 100, 200, ...; Reset checks at 100-update intervals measured from the most recent *real* transition. A new Reset Plateau segment initializes its best CE from the validation result measured at the boundary. Patience **2** means reduction after the third subsequent non-improving monitor check (PyTorch behavior).
- The **same overall ID-dev cross-entropy** is used for plateau policies throughout, not a task-specific moving target. This monitor itself has limitations because untrained tasks affect the same aggregate loss; examine task-specific CE in the analysis.
- Do **not** reset AdamW optimizer moments or model weights at transitions.
- Reset Cosine generally applies **more LR-weighted training** than Global Cosine, especially on new-task blocks. Report `cumulative_lr_by_task`; do not misattribute this to scheduler-history clearing alone.
- For Fully Mixed, `reset_cosine` and `global_cosine` must produce the same LR trajectory. Likewise, both plateau policies must have identical LR trajectories on Fully Mixed: these are built-in negative controls.

## Run order

**Phase A:** Constant, Global Cosine and Reset Cosine across four curricula × five seeds = **60 runs**. Pilot one seed first to check integration, then finish the same protocol without adjusting hyperparameters using performance results.

**Phase B:** Global and Reset ReduceLROnPlateau on the same curricula × seeds = **40 additional runs**. This is a pre-specified second comparison, not an excuse to tune settings after seeing Phase A.

The notebook can append completed runs without replacing previous runs. It does not resume *midway through an interrupted training run*; such a run starts again from the same seed.

## Evaluation and interpretation

- Save `schedule_comparison.json`, complete per-evaluation summaries and LR values, phase manifests, and exact-ID prediction traces for ID-dev and OOD-dev.
- Analysis notebook exports final-by-seed tables and matched seed contrasts with two-sided 95% Student-*t* intervals, learning curves against updates and Pair presentations, LR and LR-weighted exposure plots, transition forgetting/recovery, held-out subgroup errors, and pairwise question disagreement.
- Report per-seed directions and interval widths; don't infer broad significance from a small five-seed experiment. No cherry-picked best seed, policy, or checkpoint.
- Track Shape/Conjunction accuracy and error types: high Conjunction accuracy can coexist with ignoring Shape in missing-Shape negatives.
- A positive finding for Reset LR would justify a second **mechanism ablation** distinguishing (i) LR boost alone, (ii) plateau history reset alone, and (iii) both. Do not claim this distinction in the current experiment.
- After results are complete, create `docs/lr_reset_findings.md` with actual observations; **do not prewrite it**.

## Files

- `src/medium_visual_world/lr_policies.py`: actual LR semantics and genuine-boundary merge
- `scripts/medium_world/audit_lr_reset.py`: schedule, exposure, and policy-contract audit
- `notebooks/medium_world/lr_reset.ipynb`: training and completed-run saving
- `notebooks/medium_world/lr_reset_analysis.ipynb`: analysis of completed runs only
- `outputs/medium_world/experiments/phase_aware_lr/`: output created on the user's machine during training

## Local execution

1. Create the branch from the latest committed Scaffolded Spiral branch:
   ```powershell
   git switch experiment/staged-rehearsal
   git pull --ff-only origin experiment/staged-rehearsal
   git switch -c experiment/phase-aware-lr
   git push -u origin experiment/phase-aware-lr
   ```
2. Extract the new files at the repository root, keeping the existing `src/medium_visual_world/scaffolded_spiral.py` unchanged.
3. From repo root, run `python scripts/medium_world/audit_lr_reset.py`.
4. Launch Jupyter with its working directory at `notebooks/medium_world/`. Both notebooks use `PROJECT_ROOT = Path().cwd().parent.parent`.
5. Open `notebooks/medium_world/lr_reset.ipynb`. Pilot with `RUN_SEEDS = [42]` and `RUN_SCHEDULES = ("prereq_mixed_pair_focus", "mixed")`, while retaining `RUN_POLICIES = POLICIES[:3]`.
6. Set `RUN_SEEDS = SEEDS` and `RUN_SCHEDULES = SCHEDULES` and run all Phase A conditions. Completed runs are skipped on subsequent executions.
7. Run `notebooks/medium_world/lr_reset_analysis.ipynb` only after all **60** Phase A conditions have completed.
8. For Phase B, set `RUN_POLICIES = POLICIES[3:]`, finish the additional 40 runs, and include the two plateau policies in `ANALYZE_POLICIES` before rerunning the analysis notebook.

For a setup-only commit, use `Add phase-aware learning rate experiment and evaluation protocol`. Commit actual findings separately after the full results have been analyzed.
