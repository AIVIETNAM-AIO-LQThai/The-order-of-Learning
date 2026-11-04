# Medium-world plateau curriculum protocol

Goal: test whether learning order affects binding learning efficiency and final generalization, with the medium-world dataset and architecture unchanged.

**Freeze before the final seed comparison**

- Tasks: `color`, `shape`, `conjunction`, `pair` (2,500 examples each).
- Schedules: `mixed`; `staged` (attributes → conjunction → binding); `reverse` (binding → conjunction → attributes). All end with the same `common_mixed` consolidation.
- Main: 12 repetitions per example. Consolidation: 3 repetitions per example. Total: 15 × 10,000 = 150,000 presentations; batch 50 = **3,000 updates per model**. Main: 2,400 updates; common: 600 updates. Staged phase sizes: 1,200 / 600 / 600 / 600 updates. Reverse: 600 / 600 / 1,200 / 600.
- Same seed initializes the same weights across schedules. Same dataset, augmentation, model, optimizer, fixed dataset hashes, total budget and per-example exposure. The order differs by design. All scheduler decisions use the same **overall ID-dev cross-entropy** every 100 global updates (not stage-specific dev loss). `ReduceLROnPlateau` remains continuous across phases; stage boundaries **do not** restart the LR.
- AdamW, initial LR 1e-4, factor 0.5, patience 3 evaluations, absolute CE threshold 0.001, min LR 1e-6, weight decay 0, gradient clip 1.0, existing pixel-preserving flips/transpose augmentation.
- Primary outcomes: **fixed-final ID-dev `pair` accuracy**, and **ID-dev binding accuracy AULC from step 0 through step 3,000**. Secondary: all four task accuracies, final OOD-dev binding accuracy. Report paired seed differences and variability. OOD-dev should not drive LR scheduling or checkpoint selection. Test splits stay unused until protocol freeze.
- The `best_dev` checkpoint is selected by **overall ID-dev CE** for diagnostics; primary outcomes use **final** checkpoints. Avoid picking best pair accuracy retrospectively.
- The smoke run exercises a smaller model, and is only a pipeline check. The full seed-42 run is a protocol pilot, not a hypothesis test. The final comparison uses paired seeds 42–46.

## Execute from repository root (PowerShell)

```powershell
python valid_tests/test_plateau_curriculum.py
python -m pytest valid_tests -q
python -m experiments.medium_world.compare_learning_schedules --device cuda --smoke --seeds 42 --out outputs/medium_world/experiments/plateau_smoke
python -m experiments.medium_world.compare_learning_schedules --device cuda --seeds 42 --save-models --out outputs/medium_world/experiments/plateau_seed42
python -m experiments.medium_world.compare_learning_schedules --device cuda --seeds 42 43 44 45 46 --save-models --out outputs/medium_world/experiments/plateau_five_seeds
python -m experiments.medium_world.summarize_curriculum --input outputs/medium_world/experiments/plateau_five_seeds/schedule_comparison.json
```

This protocol intentionally **does not early-stop individual schedules**. Plateau detection reduces LR, while the update budget remains fixed across schedules. Revisit adaptive stage transitions only as a distinct follow-up experiment, after the fair fixed-budget comparison.
