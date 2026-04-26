from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_class_channel(labels: torch.Tensor, height: int, width: int, num_classes: int) -> torch.Tensor:
    normalized = labels.float().view(-1, 1, 1, 1) / max(float(num_classes - 1), 1.0)
    return normalized.expand(-1, 1, height, width)


def denorm_to_unit(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def save_image_grid(images: torch.Tensor, path: Path, nrow: int) -> None:
    grid = make_grid(denorm_to_unit(images), nrow=nrow)
    save_image(grid, path)


def save_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cycle(loader: Iterable):
    while True:
        for batch in loader:
            yield batch


def now_seconds() -> float:
    return time.time()


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remain = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{remain:02d}"


def slice_indices(num_steps: int, count: int) -> list[int]:
    if count <= 1:
        return [num_steps]
    return sorted({min(num_steps, int(round(i * num_steps / (count - 1)))) for i in range(count)})


def metric_score_from_error(error: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return max(0.0, 100.0 * (1.0 - min(error / tolerance, 1.0)))


def mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def human_bool(value: bool) -> str:
    return "yes" if value else "no"
