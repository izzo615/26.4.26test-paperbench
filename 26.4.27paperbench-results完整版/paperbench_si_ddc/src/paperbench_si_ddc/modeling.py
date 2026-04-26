from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -scale)
        angles = t[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


def _group_count(channels: int, requested_groups: int) -> int:
    for groups in range(min(requested_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int):
        super().__init__()
        group_count = _group_count(out_channels, groups)
        self.norm1 = nn.GroupNorm(_group_count(in_channels, groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(group_count, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.conv1(self.act(self.norm1(x)))
        hidden = hidden + self.time_proj(time_embedding).view(time_embedding.shape[0], -1, 1, 1)
        hidden = self.conv2(self.act(self.norm2(hidden)))
        return hidden + residual


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SimpleUNet(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        channels: int,
        out_dim: int,
        dim_mults: tuple[int, ...],
        resnet_block_groups: int,
    ):
        super().__init__()
        dims = [dim, *[dim * mult for mult in dim_mults]]
        time_dim = dim * 4

        self.init_conv = nn.Conv2d(channels, dim, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.skip_channels: list[int] = []

        current_channels = dim
        for level, next_channels in enumerate(dims[1:]):
            block1 = ResBlock(current_channels, next_channels, time_dim, resnet_block_groups)
            block2 = ResBlock(next_channels, next_channels, time_dim, resnet_block_groups)
            downsample = Downsample(next_channels) if level < len(dims[1:]) - 1 else nn.Identity()
            self.down_blocks.append(nn.ModuleList([block1, block2, downsample]))
            self.skip_channels.append(next_channels)
            current_channels = next_channels

        self.mid_block1 = ResBlock(current_channels, current_channels, time_dim, resnet_block_groups)
        self.mid_block2 = ResBlock(current_channels, current_channels, time_dim, resnet_block_groups)

        for level, skip_channels in reversed(list(enumerate(self.skip_channels))):
            block1 = ResBlock(current_channels + skip_channels, skip_channels, time_dim, resnet_block_groups)
            block2 = ResBlock(skip_channels, skip_channels, time_dim, resnet_block_groups)
            upsample = Upsample(skip_channels) if level > 0 else nn.Identity()
            self.up_blocks.append(nn.ModuleList([block1, block2, upsample]))
            current_channels = skip_channels

        self.final_block = ResBlock(current_channels + dim, dim, time_dim, resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 0:
            time = time.expand(x.shape[0])
        if time.ndim != 1:
            time = time.view(x.shape[0])

        time_embedding = self.time_mlp(time.float())
        initial = self.init_conv(x)
        hidden = initial
        skips = []

        for block1, block2, downsample in self.down_blocks:
            hidden = block1(hidden, time_embedding)
            hidden = block2(hidden, time_embedding)
            skips.append(hidden)
            hidden = downsample(hidden)

        hidden = self.mid_block1(hidden, time_embedding)
        hidden = self.mid_block2(hidden, time_embedding)

        for block1, block2, upsample in self.up_blocks:
            skip = skips.pop()
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = torch.nn.functional.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = torch.cat([hidden, skip], dim=1)
            hidden = block1(hidden, time_embedding)
            hidden = block2(hidden, time_embedding)
            hidden = upsample(hidden)

        if hidden.shape[-2:] != initial.shape[-2:]:
            hidden = torch.nn.functional.interpolate(hidden, size=initial.shape[-2:], mode="nearest")
        hidden = torch.cat([hidden, initial], dim=1)
        hidden = self.final_block(hidden, time_embedding)
        return self.final_conv(hidden)


def build_unet(input_channels: int, output_channels: int, model_config: dict, resnet_block_groups: int) -> nn.Module:
    # Replace the external dependency with a local module that preserves the same call contract.
    return SimpleUNet(
        dim=int(model_config["base_dim"]),
        channels=input_channels,
        out_dim=output_channels,
        dim_mults=tuple(int(value) for value in model_config["dim_mults"]),
        resnet_block_groups=int(resnet_block_groups),
    )


def checkpoint_path(root: Path, checkpoint_name: str) -> Path:
    return root / checkpoint_name


def save_training_checkpoint(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "metadata": metadata,
    }
    torch.save(payload, path)


def maybe_load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    if not path.exists():
        return 0
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))
