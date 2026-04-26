from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class SyntheticProblemConfig:
    target_means: list
    base_means: list
    target_cov: list
    affine: list
    dependent_noise_cov: list


class GMMTransportProblem:
    """Three-mode Gaussian-mixture transport problem with paper-style couplings."""

    def __init__(self, config: Dict):
        cfg = SyntheticProblemConfig(**config)
        self.target_means = torch.tensor(cfg.target_means, dtype=torch.float32)
        self.base_means = torch.tensor(cfg.base_means, dtype=torch.float32)
        self.target_cov = torch.tensor(cfg.target_cov, dtype=torch.float32)
        self.affine = torch.tensor(cfg.affine, dtype=torch.float32)
        self.dependent_noise_cov = torch.tensor(cfg.dependent_noise_cov, dtype=torch.float32)
        self.num_modes = int(self.target_means.shape[0])
        self.dim = int(self.target_means.shape[1])

        self.target_scale = torch.linalg.cholesky(self.target_cov)
        self.dependent_noise_scale = torch.linalg.cholesky(self.dependent_noise_cov)
        self.base_cov = self.affine @ self.target_cov @ self.affine.T + self.dependent_noise_cov
        self.base_scale = torch.linalg.cholesky(self.base_cov)

    def sample_labels(
        self,
        n: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.randint(0, self.num_modes, (n,), generator=generator, device=device)

    def _expand(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        return tensor.to(device=device)

    def _sample_gaussian_from_labels(
        self,
        means: torch.Tensor,
        scale: torch.Tensor,
        labels: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = labels.device
        noise = torch.randn((labels.shape[0], self.dim), generator=generator, device=device)
        loc = self._expand(means, device)[labels]
        return loc + noise @ self._expand(scale, device).T

    def sample_target(
        self,
        n: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: torch.device,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if labels is None:
            labels = self.sample_labels(n, generator=generator, device=device)
        x1 = self._sample_gaussian_from_labels(
            self.target_means,
            self.target_scale,
            labels,
            generator=generator,
        )
        return x1, labels

    def sample_initial_by_labels(
        self,
        labels: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return self._sample_gaussian_from_labels(
            self.base_means,
            self.base_scale,
            labels,
            generator=generator,
        )

    def sample_initial_marginal(
        self,
        n: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = self.sample_labels(n, generator=generator, device=device)
        x0 = self.sample_initial_by_labels(labels, generator=generator)
        return x0, labels

    def sample_dependent_pairs(
        self,
        n: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1, labels = self.sample_target(n, generator=generator, device=device)
        centered = x1 - self._expand(self.target_means, device)[labels]
        dependent_noise = torch.randn((n, self.dim), generator=generator, device=device)
        x0 = (
            self._expand(self.base_means, device)[labels]
            + centered @ self._expand(self.affine, device).T
            + dependent_noise @ self._expand(self.dependent_noise_scale, device).T
        )
        return x0, x1, labels

    def sample_conditioned_pairs(
        self,
        n: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = self.sample_labels(n, generator=generator, device=device)
        x1, _ = self.sample_target(n, generator=generator, device=device, labels=labels)
        x0 = self.sample_initial_by_labels(labels, generator=generator)
        return x0, x1, labels

    def sample_independent_pairs(
        self,
        n: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1, labels = self.sample_target(n, generator=generator, device=device)
        x0, _ = self.sample_initial_marginal(n, generator=generator, device=device)
        return x0, x1, labels

    def sample_pairs(
        self,
        variant: str,
        n: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if variant == "dependent":
            return self.sample_dependent_pairs(n, generator=generator, device=device)
        if variant == "conditioned":
            return self.sample_conditioned_pairs(n, generator=generator, device=device)
        if variant == "independent":
            return self.sample_independent_pairs(n, generator=generator, device=device)
        raise ValueError(f"Unknown variant: {variant}")
