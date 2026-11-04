"""Reusable balanced subsets and subset evaluation for medium-world tasks."""
import random


def balanced_indices(dataset, task: str, n: int, seed: int) -> list[int]:
    if n < 2 or n % 2:
        raise ValueError("Subset must be an even number of at least 2")
    groups = {0: [], 1: []}
    for index, row in enumerate(dataset.rows):
        if row["template"] == task:
            groups[int(row["label"])].append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
        if len(indices) < n // 2:
            raise ValueError(f"Not enough balanced {task} records for n={n}")
    result = groups[0][:n // 2] + groups[1][:n // 2]
    rng.shuffle(result)
    return result


def evaluate_indices(model, dataset, indices: list[int], device, batch_size: int = 64) -> dict:
    """Evaluate a selected subset without modifying the dataset or model."""
    import torch
    from torch import nn

    if not indices:
        raise ValueError("Cannot evaluate an empty subset")
    model.eval()
    correct = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            images, questions, labels = dataset.batch(batch_indices, device)
            logits = model(images, questions)
            correct += (logits.argmax(1) == labels).sum().item()
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
    return {"n": len(indices), "accuracy": correct / len(indices),
            "loss": loss_sum / len(indices)}
