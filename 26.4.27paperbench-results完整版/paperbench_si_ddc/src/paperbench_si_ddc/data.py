from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


def build_transform(image_size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    valid: Dataset
    num_classes: int


def _load_torchvision_cifar10(config: dict, image_size: int) -> DatasetBundle:
    dataset_cfg = config.get("dataset", {})
    transform = build_transform(image_size)
    cache_dir = dataset_cfg.get("cache_dir") or "./data"
    num_classes = int(dataset_cfg.get("num_classes", 10))

    root = Path(cache_dir) / "torchvision"
    root.mkdir(parents=True, exist_ok=True)

    train_ds = CIFAR10(root=str(root), train=True, transform=transform, download=True)
    valid_ds = CIFAR10(root=str(root), train=False, transform=transform, download=True)

    return DatasetBundle(
        train=train_ds,
        valid=valid_ds,
        num_classes=num_classes,
    )


def load_imagenet_bundle(config: dict, image_size: int) -> DatasetBundle:
    return _load_torchvision_cifar10(config, image_size=image_size)


def build_loader(
    dataset,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True,
    image_size: int | None = None,
    split: str = "train",
) -> DataLoader:
    if isinstance(dataset, dict):
        bundle = load_imagenet_bundle(dataset, image_size=int(image_size or 32))
        dataset = bundle.train if split == "train" else bundle.valid

    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
        persistent_workers=persistent_workers,
    )


def subset_dataset(dataset: Dataset, limit: int) -> Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))
