# Visual Learning Project — one consolidated repository

**Research question:** does the order in which a Vision Transformer learns color, shape and attribute-binding questions affect its performance and later transfer? The current results are exploratory and do not establish OOD compositional generalization.

The uploaded archive contained a dataset-only project duplicated alongside an extended project. This folder is the **single canonical project**: one `src/visual_world` package, one test suite, one audited dataset, and functional experiment names. Older alternate rendered images are preserved under `archive/earlier_renderings/`; Python caches and duplicate manifests were not carried over.

## Structure

```text
visual_world/
├── src/visual_world/           # synthetic scene generation, symbolic oracle, rendering, data audit, ViT utilities
├── small_experiments/          # individually runnable experiments (descriptive names)
├── tests/                      # original combined tests
├── configs/                    # data-generation configuration
├── outputs/
│   ├── dataset/                # fixed manifests, metadata, audit and scene gallery
│   └── experiments/
│       ├── transformer_baseline/
│       ├── visual_grounding_debug/
│       ├── color_grounding/
│       ├── attribute_binding/
│       ├── generalization_controls/
│       └── learning_order/
├── figures/                    # earlier reference schematics; notebook regenerates new ones under outputs/
├── thesis/                     # research papers; PDF names are unchanged
├── docs/                       # historical research protocols and notes
├── archive/earlier_renderings/ # distinct older gallery versions
├── prepare_dataset.py
├── train_multimodal_baseline.py
├── notebooks/generate_research_figures.ipynb  # regenerates PNG figures from saved data
├── report/                      # Vietnamese LaTeX report, assets copied from notebook
├── FILE_INDEX.csv              # purpose of EVERY file except Markdown documents
└── FILE_MOVES.csv              # mapping from uploaded paths to consolidated paths
```

## Install and test

From **this project root** (not from inside `small_experiments`):

```powershell
python -m pip install -e ".[dev,train]"
python -m pytest -q
```

For Windows GPU usage, install a compatible CUDA PyTorch build via the [PyTorch installer](https://pytorch.org/get-started/locally/) before running CUDA experiments. All included experiments default to CUDA; use `--device cpu` when testing without a GPU.

## Dataset

The original audited manifests are already in `outputs/dataset/` (1,800 train examples across 3 levels). The reserved `id_test`, `ood_test` and `adapt_test` manifests remain present but should NOT be read when tuning on the development data. To regenerate the dataset **(this will overwrite the existing data)**:

```powershell
python prepare_dataset.py
```

To inspect an existing dataset without regenerating it:

```powershell
python -m src.visual_world.audit --root outputs/dataset
```

## Run experiments (from project root)

```powershell
# Original multimodal ViT and diagnostics
python train_multimodal_baseline.py smoke --device cpu
python -m small_experiments.debug_visual_grounding --device cpu --phase template --template color

# Revised spatial grounding model
python -m small_experiments.train_color_grounding --device cpu --epochs 40

# Single-task attribute binding / shape recognition
python -m small_experiments.train_attribute_binding --device cpu --task shape --epochs 50
python -m small_experiments.train_attribute_binding --device cpu --task pair --epochs 60

# Train/evaluate three generalization controls
python -m small_experiments.compare_generalization_controls --device cpu --task pair --variants baseline augment decay --seeds 42 --epochs 40

# Matched-exposure curriculum comparison
python -m small_experiments.compare_learning_schedules --device cpu --seeds 42 --main-epochs 20 --common-epochs 4
```

**These commands train models and may take substantial compute.** The accompanying JSON files already contain the historical run outputs and `.pt` files preserve available trained checkpoints. The learning-order experiment uses the original model and data, but renamed paths; numerical reproduction may vary with hardware and PyTorch versions.

## File documentation

`FILE_INDEX.csv` describes the purpose of each non-Markdown file, including the figure notebook, experimental figures, and unchanged reference PDFs. Markdown files are intentionally excluded from that inventory, as requested. The legacy rename map was removed so current documentation describes the actual project rather than obsolete filenames.

Do not interpret engineering development scores as final test or out-of-distribution findings. Current work includes only a single-seed exploratory schedule comparison.

## Regenerate figures (no retraining)

Open `notebooks/generate_research_figures.ipynb` in VS Code/Jupyter and choose **Run All**.
The notebook reads the recorded JSON metrics and scene data, then exports charts and
explanatory diagrams into `outputs/experiments/<experiment>/` and `outputs/dataset/`.
Figures needed by the Vietnamese report are copied to `report/Figures/`.

The notebook uses Matplotlib, NumPy and Pillow; install the optional `train` dependencies
from `pyproject.toml` or run `python -m pip install matplotlib numpy pillow notebook`.
It never retrains a neural network or overwrites thesis PDFs. Architectural diagrams
are schematic reconstructions, not captured network attention maps.
