# The Order of Learning

Research whether ordering simple-to-compositional visual questions improves learning efficiency, knowledge retention and generalization under matched data exposure.

## Workflow

**Generate → Audit → EDA → Train → Evaluate → Analyze**

- **Reusable code:** `src/small_visual_world/` and `src/medium_visual_world/` contain world definitions, generators, oracle, renderer, dataset/model interfaces, curriculum plans, reusable sampling and diagnostics.
- **Generate datasets:** `scripts/generate_dataset.py`; the only normal command-line entry point. Frozen data and previous results remain under `outputs/`.
- **Experiments:** notebooks under `notebooks/small_world/` and `notebooks/medium_world/`; training, evaluation, figures, and analysis stay in the notebooks.
- **Scientific context:** `docs/`; tests in `valid_tests/`.

## Environment

```powershell
python -m pip install -e ".[dev,train]"
python -m pip install jupyter pandas nbformat
python -m pytest valid_tests -q
```

Open notebooks from within the repository using VS Code or Jupyter. Each migrated notebook resolves the repository root automatically.

## Dataset generation and integrity

```powershell
# Safe: reproduce manifests into temporary directory and compare with frozen datasets.
python -m scripts.generate_dataset --world small --verify-existing
python -m scripts.generate_dataset --world medium --verify-existing

# New dataset in a distinct directory (no overwrite of frozen splits):
python -m scripts.generate_dataset --world medium --out outputs/medium_world/dataset_new
```

Configs: `configs/small_world/scene_generation.json` and `configs/medium_world/scene_generation.json`.
Existing `outputs/{small,medium}_world/dataset/` must remain unchanged for comparisons with recorded training runs.

## Notebooks

- `notebooks/small_world/`: EDA, nine-seed learning-order comparison, research figures.
- `notebooks/medium_world/`: EDA, diagnostics, five-seed plateau, rehearsal comparisons, 9,000-step training and analysis.

See `docs/notebook_first_refactor.md` for migration details.
Historical experiment scripts can be recovered from earlier commits of the research branch; numerical results are retained under `outputs/`.
