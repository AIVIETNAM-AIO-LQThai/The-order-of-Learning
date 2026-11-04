"""Reusable stratified train/development split for small-world adaptation."""
import random


def stratified_adaptation_indices(dataset, seed: int = 1919,
                                  dev_per_class: int = 16) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    groups = {0: [], 1: []}
    for index, label in enumerate(dataset.labels):
        groups[int(label)].append(index)
    if any(len(indices) <= dev_per_class for indices in groups.values()):
        raise ValueError("Too few adaptation rows for stratified split")
    dev, train = [], []
    for cls in (0, 1):
        rng.shuffle(groups[cls])
        dev.extend(groups[cls][:dev_per_class])
        train.extend(groups[cls][dev_per_class:])
    return sorted(train), sorted(dev)
