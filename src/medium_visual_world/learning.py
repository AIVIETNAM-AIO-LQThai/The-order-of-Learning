from pathlib import Path
import sys
PROJECT_ROOT = Path().cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
import numpy as np
import torch
from torch import nn
from PIL import Image

from src.medium_visual_world.generate import read_jsonl
from src.medium_visual_world.world import COLORS, SHAPES, from_dict
from src.medium_visual_world.render import render

TASKS = ("color", "shape", "conjunction", "pair")
OPERATORS = ("EXISTS_COLOR", "EXISTS_SHAPE", "HAS_COLOR_AND_SHAPE", "EXISTS_PAIR")
VOCAB = ("[PAD]", "[CLS]", *OPERATORS, *COLORS, *SHAPES)
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_query(words: list[str]) -> list[int]:
    if not 2 <= len(words) <= 3:
        raise ValueError("Expected one operator and one or two attributes")
    return [TOKEN_TO_ID["[CLS]"]] + [TOKEN_TO_ID[word] for word in words] + [0] * (3 - len(words))


class MediumDataset:
    def __init__(self, path: Path, limit: int | None = None):
        self.path = Path(path)
        self.rows = list(read_jsonl(self.path))
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"No examples in {path}")
        scene_ids = {}
        images = []
        image_indices = []
        queries = []
        labels = []
        for row in self.rows:
            scene_id = row["scene_id"]
            if scene_id not in scene_ids:
                image_path = self.path.parent / row["image"]
                if image_path.is_file():
                    with Image.open(image_path) as image:
                        pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
                else:
                    pixels = np.array(render(from_dict(row["scene"])), dtype=np.uint8, copy=True)
                scene_ids[scene_id] = len(images)
                images.append(np.transpose(pixels, (2, 0, 1)).copy())
            image_indices.append(scene_ids[scene_id])
            queries.append(encode_query(row["query_tokens"]))
            labels.append(int(row["label"]))
        self.images = torch.from_numpy(np.stack(images))
        self.image_indices = torch.tensor(image_indices, dtype=torch.long)
        self.queries = torch.tensor(queries, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tasks = [row["template"] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def batch(self, indices, device: torch.device, constant_image: bool = False):
        image_ids = self.image_indices[indices]
        x = self.images[image_ids].to(device=device, dtype=torch.float32).div_(255.0)
        if constant_image:
            background = torch.tensor((243, 246, 244), device=device, dtype=x.dtype) / 255.0
            x = background.view(1, 3, 1, 1).expand_as(x)
        q = self.queries[indices].to(device)
        y = self.labels[indices].to(device)
        return x, q, y


class AttributeBindingViT(nn.Module):
    def __init__(self, dim: int = 48, patch_size: int = 4, depth: int = 1):
        super().__init__()
        if dim % 4:
            raise ValueError("dim must be divisible by four")
        if 64 % patch_size:
            raise ValueError("patch_size must divide 64")
        patches = (64 // patch_size) ** 2
        self.patch = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, patches, dim))
        layer = nn.TransformerEncoderLayer(
            dim, 4, dim * 2, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.pre_norm = nn.LayerNorm(dim)
        self.post_norm = nn.LayerNorm(dim)
        self.emb = nn.Embedding(len(VOCAB), dim, padding_idx=0)
        self.query = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.LayerNorm(dim))
        self.score_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x, q):
        if x.shape[1:] != (3, 64, 64):
            raise ValueError("Expected 64x64 RGB images")
        e = self.emb(q[:, 1:4])
        query = self.query(e.flatten(1))
        patches = self.patch(x).flatten(2).transpose(1, 2)
        v = self.pre_norm(patches) + 0.5 * self.post_norm(self.encoder(patches + self.pos))
        v = self.score_norm(v)
        scale = v.shape[-1] ** 0.5
        score1 = torch.einsum("bnd,bd->bn", v, e[:, 1]) / scale
        score2 = torch.einsum("bnd,bd->bn", v, e[:, 2]) / scale
        score_joint = torch.einsum("bnd,bd->bn", v, query) / scale
        interaction = torch.tanh(score1) * torch.tanh(score2)
        features = torch.stack((
            score1.amax(1), score1.mean(1),
            score2.amax(1), score2.mean(1),
            score_joint.amax(1), score_joint.mean(1),
            interaction.amax(1), interaction.mean(1)), dim=-1)
        return self.head(features)


def augment_images(images: torch.Tensor) -> torch.Tensor:
    for dimension in (-1, -2):
        selector = torch.rand((len(images), 1, 1, 1), device=images.device) < 0.5
        images = torch.where(selector, images.flip(dimension), images)
    selector = torch.rand((len(images), 1, 1, 1), device=images.device) < 0.5
    return torch.where(selector, images.transpose(-1, -2), images)


@torch.inference_mode()
def evaluate(model, dataset: MediumDataset, device: torch.device,
             batch_size: int = 64, limit: int | None = None,
             constant_image: bool = False) -> dict:
    model.eval()
    total = min(len(dataset), limit) if limit is not None else len(dataset)
    correct = 0
    loss_sum = 0.0
    counts = {task: 0 for task in TASKS}
    successes = {task: 0 for task in TASKS}
    for start in range(0, total, batch_size):
        indices = list(range(start, min(total, start + batch_size)))
        x, q, y = dataset.batch(indices, device, constant_image)
        scores = model(x, q)
        loss_sum += nn.functional.cross_entropy(scores, y, reduction="sum").item()
        outcomes = (scores.argmax(1) == y).cpu().tolist()
        for index, successful in zip(indices, outcomes):
            task = dataset.tasks[index]
            counts[task] += 1
            successes[task] += int(successful)
            correct += int(successful)
    return {"n": total, "accuracy": correct / total, "loss": loss_sum / total,
            "by_template": {task: successes[task] / count for task, count in counts.items() if count}}


def environment() -> dict:
    result = {"python": sys.version.split()[0], "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
    return result
