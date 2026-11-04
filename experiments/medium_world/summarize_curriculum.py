"""Print fixed-final and AULC comparisons from schedule_comparison.json."""
from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from collections import defaultdict
from statistics import mean, stdev

SCHEDULES = ("mixed", "staged", "reverse")


def summarize(report: dict):
    runs = report["runs"]
    by_schedule = defaultdict(dict)
    for run in runs:
        if run["status"] != "completed":
            continue
        seed = run["seed"]
        schedule = run["schedule"]
        if seed in by_schedule[schedule]:
            raise ValueError(f"Duplicate {schedule}, seed {seed}")
        by_schedule[schedule][seed] = run
    common = set.intersection(*(set(by_schedule[s]) for s in SCHEDULES))
    if not common:
        raise ValueError("No seed has all three completed schedules")
    seeds = sorted(common)
    print("Paired seeds:", seeds)
    for schedule in SCHEDULES:
        samples = [by_schedule[schedule][seed] for seed in seeds]
        updates = {r["steps_run"] for r in samples}
        if len(updates) != 1:
            raise ValueError("Inconsistent update counts")
        def values(key):
            if key == "aulc":
                return [r["id_pair_aulc"] * 100 for r in samples]
            if key == "id_pair":
                return [r["final"]["id_dev"]["by_template"]["pair"] * 100 for r in samples]
            return [r["final"]["ood_dev"]["by_template"]["pair"] * 100 for r in samples]
        def formatted(nums):
            return f"{mean(nums):6.2f} +/- {stdev(nums):5.2f}" if len(nums) > 1 else f"{mean(nums):6.2f}"
        print(f"{schedule:8} | AULC {formatted(values('aulc'))} | "
              f"ID-final {formatted(values('id_pair'))} | "
              f"OOD-final {formatted(values('ood_pair'))}")
    for other in ("mixed", "reverse"):
        print(f"\nPaired staged - {other} differences (percentage points):")
        for key in ("id_pair_aulc", "id_pair", "ood_pair"):
            if key == "id_pair_aulc":
                get = lambda r: r["id_pair_aulc"]
            elif key == "id_pair":
                get = lambda r: r["final"]["id_dev"]["by_template"]["pair"]
            else:
                get = lambda r: r["final"]["ood_dev"]["by_template"]["pair"]
            diff = [(get(by_schedule["staged"][s]) - get(by_schedule[other][s])) * 100
                    for s in seeds]
            print(f"{key:14}: {mean(diff):+.2f} mean | "
                  f"per-seed={[round(d, 2) for d in diff]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    summarize(report)


if __name__ == "__main__":
    main()
