from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import random
import torch
from torch import nn
from src.medium_visual_world.learning import (
    MediumDataset, AttributeBindingViT, augment_images, evaluate, set_seed
)


def balanced_indices(dataset: MediumDataset, task: str, n: int, seed: int) -> list[int]:
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


def evaluate_indices(model, dataset: MediumDataset, indices: list[int],
                     device: torch.device, batch_size: int = 64) -> dict:
    model.eval()
    correct = 0
    loss_sum = 0.0
    n = len(indices)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            images, questions, labels = dataset.batch(batch_indices, device)
            logits = model(images, questions)
            correct += (logits.argmax(1) == labels).sum().item()
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
    return {"n": n, "accuracy": correct / n, "loss": loss_sum / n}


def run(task: str, subset: int, steps: int, batch_size: int, lr: float,
        seed: int, augment: bool, data: Path, out: Path, device: torch.device,
        eval_every: int, save_model: bool = False) -> dict:
    set_seed(seed)
    train = MediumDataset(data / "train.jsonl")
    dev = MediumDataset(data / "id_dev.jsonl")
    train_indices = balanced_indices(train, task, subset, seed)
    dev_indices = [i for i, row in enumerate(dev.rows) if row["template"] == task]
    model = AttributeBindingViT(dim=48, patch_size=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(seed + 1337)
    history = []
    order = []
    for step in range(1, steps + 1):
        if len(order) < batch_size:
            order = train_indices.copy()
            rng.shuffle(order)
        indices, order = order[:batch_size], order[batch_size:]
        model.train()
        x, q, y = train.batch(indices, device)
        if augment:
            x = augment_images(x)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x, q), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            record = {"step": step,
                      "train": evaluate_indices(model, train, train_indices, device),
                      "id_dev": evaluate_indices(model, dev, dev_indices, device)}
            history.append(record)
            print(f'{task} n={subset} step={step}: '
                  f'train={record["train"]["accuracy"]:.3f}, '
                  f'ID-dev={record["id_dev"]["accuracy"]:.3f}', flush=True)
    result = {"task": task, "subset": subset, "seed": seed, "steps": steps,
              "batch_size": batch_size, "lr": lr, "augmentation": augment,
              "history": history, "final": history[-1]}
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{task}_n{subset}_seed{seed}"
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if save_model:
        torch.save({"model": model.state_dict(), "dim": 48, "patch_size": 4,
                    "task": task, "subset": subset, "seed": seed, "steps": steps},
                   out / f"{stem}.pt")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("shape", "pair", "color", "conjunction"), required=True)
    parser.add_argument("--subset", type=int, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "outputs/medium_world/dataset")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/medium_world/experiments/diagnostics/single_task")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.steps < 1 or args.eval_every < 1 or args.batch_size < 1 or args.batch_size > args.subset:
        parser.error("Invalid steps, eval interval, or batch size")
    run(args.task, args.subset, args.steps, args.batch_size, args.lr, args.seed,
        args.augment, args.data, args.out, torch.device(args.device),
        args.eval_every, args.save_model)


if __name__ == "__main__":
    main()
