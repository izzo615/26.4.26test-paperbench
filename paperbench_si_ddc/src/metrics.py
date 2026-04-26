from __future__ import annotations

from typing import Dict

import torch


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y) ** 2


def rbf_mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        merged = torch.cat([x, y], dim=0)
        distances = torch.pdist(merged)
        bandwidth = torch.median(distances).item()
        if bandwidth <= 0.0:
            bandwidth = 1.0
        gamma = 1.0 / (2.0 * bandwidth * bandwidth)
        k_xx = torch.exp(-gamma * _pairwise_sq_dists(x, x))
        k_yy = torch.exp(-gamma * _pairwise_sq_dists(y, y))
        k_xy = torch.exp(-gamma * _pairwise_sq_dists(x, y))
        n = x.shape[0]
        m = y.shape[0]
        mmd2 = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1))
        mmd2 += (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1))
        mmd2 -= 2.0 * k_xy.mean()
        return float(mmd2.item())


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, num_projections: int = 128) -> float:
    with torch.no_grad():
        device = x.device
        projections = torch.randn((num_projections, x.shape[1]), device=device)
        projections = projections / projections.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x_proj = torch.sort(x @ projections.T, dim=0).values
        y_proj = torch.sort(y @ projections.T, dim=0).values
        return float(torch.sqrt(((x_proj - y_proj) ** 2).mean()).item())


def path_efficiency(traj: torch.Tensor) -> float:
    with torch.no_grad():
        steps = traj[1:] - traj[:-1]
        path_length = steps.norm(dim=-1).sum(dim=0)
        displacement = (traj[-1] - traj[0]).norm(dim=-1).clamp_min(1e-8)
        return float((path_length / displacement).mean().item())


def mode_balance_error(samples: torch.Tensor, means: torch.Tensor) -> float:
    with torch.no_grad():
        labels = torch.cdist(samples, means).argmin(dim=1)
        counts = torch.bincount(labels, minlength=means.shape[0]).float()
        probs = counts / counts.sum().clamp_min(1.0)
        target = torch.full_like(probs, 1.0 / means.shape[0])
        return float(torch.abs(probs - target).sum().item())


def summarize_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    traj: torch.Tensor,
    target_means: torch.Tensor,
) -> Dict[str, float]:
    return {
        "mmd2": rbf_mmd(generated, target),
        "sliced_w2": sliced_wasserstein(generated, target),
        "path_efficiency": path_efficiency(traj),
        "mode_balance_l1": mode_balance_error(generated, target_means),
    }
