from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_linear_interpolant(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t_image = t.view(-1, 1, 1, 1)
    xt = t_image * x0 + (1.0 - t_image) * x1
    xdot = x1 - x0
    return xt, xdot


def sample_tile_mask(batch_size: int, height: int, width: int, tile_grid: int, keep_probability: float, device: torch.device) -> torch.Tensor:
    coarse = torch.bernoulli(
        torch.full((batch_size, 1, tile_grid, tile_grid), keep_probability, device=device)
    )
    return F.interpolate(coarse, size=(height, width), mode="nearest")


def make_inpainting_base(x1: torch.Tensor, observed_mask: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x1)
    return observed_mask * x1 + (1.0 - observed_mask) * noise


def down_up_sample(x: torch.Tensor, low_resolution: int, high_resolution: int) -> torch.Tensor:
    low = F.interpolate(x, size=(low_resolution, low_resolution), mode="bicubic", align_corners=False)
    high = F.interpolate(low, size=(high_resolution, high_resolution), mode="bicubic", align_corners=False)
    return high


def make_superres_base(x1: torch.Tensor, low_resolution: int, high_resolution: int, noise_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    upsampled = down_up_sample(x1, low_resolution=low_resolution, high_resolution=high_resolution)
    x0 = upsampled + noise_scale * torch.randn_like(x1)
    return x0, upsampled


def project_observed_region(sample: torch.Tensor, observed_pixels: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    return observed_mask * observed_pixels + (1.0 - observed_mask) * sample
