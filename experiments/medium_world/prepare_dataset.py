from pathlib import Path
import argparse
import json
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.medium_visual_world.generate import generate
from src.medium_visual_world.audit import audit

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "medium_world" / "dataset"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-scenes", type=int, default=5000)
    parser.add_argument("--id-dev-scenes", type=int, default=512)
    parser.add_argument("--id-test-scenes", type=int, default=512)
    parser.add_argument("--ood-dev-scenes", type=int, default=400)
    parser.add_argument("--ood-test-scenes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    metadata = generate(
        OUTPUT_DIR, seed=args.seed,
        train_scenes=args.train_scenes,
        id_dev_scenes=args.id_dev_scenes,
        id_test_scenes=args.id_test_scenes,
        ood_dev_scenes=args.ood_dev_scenes,
        ood_test_scenes=args.ood_test_scenes,
        materialize=not args.no_images
    )
    report = audit(OUTPUT_DIR)
    print(json.dumps({"dataset": metadata["splits"],
                      "audit": report["status"],
                      "total_unique_scenes": report["unique_scenes"]}, indent=2))


if __name__ == "__main__":
    main()
