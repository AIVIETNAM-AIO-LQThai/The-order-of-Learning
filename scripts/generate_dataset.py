"""One entry point for deterministic dataset generation, audit, and reproduction.

Run from repository root:
  python -m scripts.generate_dataset --world small --verify-existing
  python -m scripts.generate_dataset --world medium --verify-existing
  python -m scripts.generate_dataset --world medium --out outputs/medium_world/new_dataset

Existing datasets are protected unless --force is explicitly supplied.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def defaults(world):
    return ROOT / "configs" / f"{world}_world" / "scene_generation.json"


def read_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Expected a JSON object in dataset configuration")
    return config


def validate_hashes(folder: Path):
    metadata = json.loads((folder / "dataset_metadata.json").read_text(encoding="utf-8"))
    expected = metadata["manifests_sha256"]
    mismatched = []
    for name, digest in expected.items():
        path = folder / name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(f"Missing or modified dataset manifests: {mismatched}")
    return expected


def generate_world(world: str, folder: Path, config: dict):
    if world == "small":
        from src.small_visual_world.generate import generate
        from src.small_visual_world.audit import audit
        from src.small_visual_world.gallery import make_gallery
        metadata = generate(folder, **config)
        make_gallery(folder)
        result = audit(folder)
        if not result["pass"]:
            raise RuntimeError("Small-world dataset audit failed")
    else:
        from src.medium_visual_world.generate import generate
        from src.medium_visual_world.audit import audit
        metadata = generate(folder, **config)
        result = audit(folder)
        if result["status"] != "passed":
            raise RuntimeError("Medium-world dataset audit failed")
    (folder / "audit_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_hashes(folder)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", choices=("small", "medium"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing dataset")
    parser.add_argument("--verify-existing", action="store_true", help="Regenerate in a temporary directory and compare hashes; never alter existing dataset")
    args = parser.parse_args()
    config_path = args.config or defaults(args.world)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    folder = args.out or ROOT / "outputs" / f"{args.world}_world" / "dataset"
    if not folder.is_absolute():
        folder = ROOT / folder
    folder = folder.resolve()
    config = read_config(config_path)
    if args.verify_existing:
        expected = validate_hashes(folder)
        with TemporaryDirectory(prefix=f"{args.world}_regen_") as scratch:
            scratch = Path(scratch) / "dataset"
            trial_config = config.copy()
            trial_config["materialize_all" if args.world == "small" else "materialize"] = False
            generated = generate_world(args.world, scratch, trial_config)
            if generated["manifests_sha256"] != expected:
                raise RuntimeError("Reproduction hashes differ: generator/config may have changed")
        print(f"PASS: {args.world} dataset reproduced; original untouched: {folder}")
        return
    if folder.exists() and any(folder.iterdir()) and not args.force:
        parser.error(f"Refusing to overwrite existing dataset: {folder} (use --verify-existing or choose --out)")
    folder.mkdir(parents=True, exist_ok=True)
    metadata = generate_world(args.world, folder, config)
    print(json.dumps({"world": args.world, "output": str(folder),
                      "manifests_sha256": metadata["manifests_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
