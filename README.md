# The Order of Learning

Study whether the order of learning visual tasks affects a Vision Transformer’s performance, retention, and generalization. Results from the original synthetic world are exploratory, not evidence of OOD generalization.

## Research worlds

- **small_world**: Original synthetic scenes and experiments, including the nine-seed comparison.
- **medium_world**: Planned 5,000-scene synthetic environment with new colors, shapes, and tasks.
- **large_world**: Future public-dataset experiments.

Code lives under `src/<world>_visual_world/` and `experiments/<world>/`. Each world’s datasets, experimental results, checkpoints, and figures live under `outputs/<world>/`. The notebooks stay directly in `notebooks/` and run from that directory. The reference PDFs in `thesis/` retain their original filenames.

## Install and test

Run Python modules from the repository root; run notebooks from `notebooks/`.

```powershell
python -m pip install -e ".[dev,train]"
python -m pytest -q
```

## Original synthetic world

Existing audited data: `outputs/small_world/dataset/`. Results and checkpoints: `outputs/small_world/experiments/`.

```powershell
# Inspect existing data
python -m src.small_visual_world.audit --root outputs/small_world/dataset

# Only if you intend to REGENERATE the existing dataset
python -m experiments.small_world.prepare_dataset

# Model and individual experiments
python -m experiments.small_world.train_multimodal_baseline smoke --device cpu
python -m experiments.small_world.debug_visual_grounding --device cpu --phase template --template color
python -m experiments.small_world.train_color_grounding --device cpu --epochs 40
python -m experiments.small_world.train_attribute_binding --device cpu --task pair --epochs 60
python -m experiments.small_world.compare_generalization_controls --device cpu --task pair --variants baseline augment decay --seeds 42 --epochs 40
python -m experiments.small_world.compare_learning_schedules --device cpu --seeds 42 --main-epochs 20 --common-epochs 4
```

The training commands may take substantial compute. They use development splits for exploratory evaluation. Do not reinterpret those measurements as held-out test results.

## Notebooks

- `analyze_seeds_original.ipynb`: nine-seed learning-order comparison and task-interference analysis.
- `generate_research_figures.ipynb`: regenerate small-world figures from recorded experiments and training-scene data without retraining.

Run both from `notebooks/` in VS Code/Jupyter. Generated outputs remain in `outputs/small_world/`; the figure notebook also exports report images to `report/Figures/` when run.
