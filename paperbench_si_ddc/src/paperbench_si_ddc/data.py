from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


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


class HuggingFaceImageDataset(Dataset):
    def __init__(self, dataset, image_key: str, label_key: str, transform: Callable):
        self.dataset = dataset
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        image = item[self.image_key]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.transform(image), int(item[self.label_key])


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    valid: Dataset
    num_classes: int


def _load_huggingface_cifar10(config: dict, image_size: int) -> DatasetBundle:
    from datasets import load_dataset

    dataset_cfg = config.get("dataset", {})
    transform = build_transform(image_size)
    cache_dir = dataset_cfg.get("cache_dir") or None
    name = dataset_cfg.get("name", "cifar10")
    train_split = dataset_cfg.get("train_split", "train")
    valid_split = dataset_cfg.get("valid_split", "test")
    image_key = dataset_cfg.get("image_key", "img")
    label_key = dataset_cfg.get("label_key", "label")
    num_classes = int(dataset_cfg.get("num_classes", 10))

    train_ds = load_dataset(name, split=train_split, cache_dir=cache_dir)
    valid_ds = load_dataset(name, split=valid_split, cache_dir=cache_dir)

    return DatasetBundle(
        train=HuggingFaceImageDataset(train_ds, image_key=image_key, label_key=label_key, transform=transform),
        valid=HuggingFaceImageDataset(valid_ds, image_key=image_key, label_key=label_key, transform=transform),
        num_classes=num_classes,
    )


def load_imagenet_bundle(config: dict, image_size: int) -> DatasetBundle:
    # Keep the historical function name so the existing pipeline API stays stable.
    return _load_huggingface_cifar10(config, image_size=image_size)


def build_loader(
    dataset,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True,
    image_size: int | None = None,
    split: str = "train",
) -> DataLoader:
    # Older callers sometimes pass the config instead of a Dataset; accept both.
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
