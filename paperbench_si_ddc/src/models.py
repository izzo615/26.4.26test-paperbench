from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class TimeFeatures(nn.Module):
    def __init__(self, num_frequencies: int):
        super().__init__()
        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        return torch.cat([t.unsqueeze(-1), torch.sin(angles), torch.cos(angles)], dim=-1)


class VelocityField(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        depth: int,
        time_features: int,
        num_classes: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.time_encoder = TimeFeatures(time_features)
        input_dim = 2 + 1 + 2 * time_features + num_classes

        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = [x, self.time_encoder(t)]
        if self.num_classes > 0:
            if labels is None:
                raise ValueError("labels are required for class-conditioned models")
            features.append(torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float())
        return self.network(torch.cat(features, dim=-1))
