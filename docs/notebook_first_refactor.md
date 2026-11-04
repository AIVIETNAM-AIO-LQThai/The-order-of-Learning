# Notebook-first repository refactor

This refactor is applied directly to `experiment/staged-rehearsal`; no new Git branch is required.
The existing `outputs/` directory is **not changed, deleted, regenerated, or relocated**.

## Structure

- `src/small_visual_world/`, `src/medium_visual_world/`: scene models, generators, audits, rendering, oracle, models, reusable scheduling/sampling/diagnostics.
- `scripts/generate_dataset.py`: the **only dataset-generation CLI**; refuses to overwrite an existing dataset without `--force`.
- `notebooks/small_world/`, `notebooks/medium_world/`: all research notebooks, including training/evaluation.
- `docs/`, `outputs/`, `valid_tests/`: research evidence, experiment data, tests.
- `experiments/` and `src/small_visual_world/baseline_training.py`: removed after reusable helpers and imports migrate.

## Dataset provenance

Do **not** regenerate the frozen datasets in `outputs/{world}_world/dataset/` during refactoring.
To check reproducibility without modifying those files:

```powershell
python -m scripts.generate_dataset --world small --verify-existing
python -m scripts.generate_dataset --world medium --verify-existing
```

For new datasets, select a new output directory instead of overwriting the frozen split:

```powershell
python -m scripts.generate_dataset --world medium --out outputs/medium_world/dataset_new
```

## Running notebooks

The migration injects a first cell that discovers the project root and restores the old `notebooks/` working directory. Existing notebook references to `Path.cwd().parent` therefore continue to work. It also adds the project root to `sys.path`.

- `notebooks/medium_world/long_budget.ipynb`: 9,000-step matched-exposure comparison, configurable `plan` / `smoke` / `train`, cosine LR without stage resets. The curriculum implementation is imported from `src.medium_visual_world.curriculum`.
- All previously tracked notebooks are moved into their corresponding world folders, using short names.
- Historical experiment-specific `.py` runners remain available in prior Git commits on `experiment/staged-rehearsal`.

Do not use the test splits for exploratory tuning. All results already committed under `outputs/` remain unchanged.
