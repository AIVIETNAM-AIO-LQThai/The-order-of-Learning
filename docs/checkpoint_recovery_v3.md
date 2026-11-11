# Five-seed diagnostics — checkpoint recovery and root-path fix (v3)

This revision fixes the regression seen in the executed `representation_checkpoints(2).ipynb`:

- The notebook now **archives incomplete checkpoint sets** in `incomplete_checkpoint_backups/<schedule>_seed_<seed>[/version]` and retrains only that incomplete schedule/seed from step zero. It never deletes the partial checkpoints.
- Complete three-checkpoint sets are loaded only after validating their seed, schedule, step, dataset hashes, source hashes, model configuration, training configuration, and finite weights.
- Checkpoint serialization stores environment string subclasses as ordinary `str`; older `TorchVersion` metadata is supported with `safe_globals` under `weights_only=True`.
- All three notebooks now begin with exactly the requested root setup: `PROJECT_ROOT = Path.cwd().resolve().parent.parent` and guarded `sys.path.insert`. Work from `notebooks/medium_world`.
- All prior saved error outputs were cleared so the next run starts clean.
- Full experiment remains five seeds, two schedules, three checkpoints (30), no pilot.

A partially trained model must restart from step zero: only its model weights were saved, not optimizer or RNG states. Existing complete/verified schedule+seed groups are reused.
