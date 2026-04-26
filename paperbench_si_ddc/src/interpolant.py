from __future__ import annotations

from typing import Optional

import torch


def gamma(t: torch.Tensor, sigma: float) -> torch.Tensor:
    return sigma * torch.sqrt(torch.clamp(t * (1.0 - t), min=1e-8))


def gamma_dot(t: torch.Tensor, sigma: float) -> torch.Tensor:
    denom = 2.0 * torch.sqrt(torch.clamp(t * (1.0 - t), min=1e-8))
    return sigma * (1.0 - 2.0 * t) / denom


def sample_stochastic_interpolant(
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    t: torch.Tensor,
    sigma: float,
    noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x0)
    t_column = t.unsqueeze(-1)
    xt = (1.0 - t_column) * x0 + t_column * x1 + gamma(t, sigma).unsqueeze(-1) * noise
    xt_dot = x1 - x0 + gamma_dot(t, sigma).unsqueeze(-1) * noise
    return xt, xt_dot


@torch.no_grad()
def integrate_probability_flow(
    model,
    x0: torch.Tensor,
    *,
    labels: Optional[torch.Tensor],
    steps: int,
) -> torch.Tensor:
    model.eval()
    traj = [x0.detach().cpu()]
    x = x0.clone()
    dt = 1.0 / steps
    for idx in range(steps):
        t = torch.full((x.shape[0],), idx / steps, device=x.device, dtype=x.dtype)
        k1 = model(x, t, labels)
        x_mid = x + 0.5 * dt * k1
        t_mid = torch.full((x.shape[0],), min((idx + 0.5) / steps, 1.0), device=x.device, dtype=x.dtype)
        k2 = model(x_mid, t_mid, labels)
        x = x + dt * k2
        traj.append(x.detach().cpu())
    return torch.stack(traj, dim=0)
