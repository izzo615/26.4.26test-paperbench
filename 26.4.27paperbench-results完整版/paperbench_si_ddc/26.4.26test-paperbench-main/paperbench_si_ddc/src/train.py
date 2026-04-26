from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.interpolant import sample_stochastic_interpolant
from src.models import VelocityField


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(config: Dict, *, conditioned: bool, num_classes: int) -> nn.Module:
    return VelocityField(
        hidden_dim=int(config["hidden_dim"]),
        depth=int(config["depth"]),
        time_features=int(config["time_features"]),
        num_classes=num_classes if conditioned else 0,
    )


def train_variant(
    problem,
    variant: str,
    config: Dict,
    output_checkpoint: str,
    device: torch.device,
) -> Tuple[nn.Module, pd.DataFrame]:
    conditioned = variant == "conditioned"
    model = build_model(config, conditioned=conditioned, num_classes=problem.num_modes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    generator = torch.Generator(device=device)
    variant_seed_offset = {
        "independent": 11,
        "conditioned": 23,
        "dependent": 37,
    }[variant]
    generator.manual_seed(int(config["seed"]) + variant_seed_offset)

    steps = int(config["train_steps"])
    batch_size = int(config["batch_size"])
    val_batch_size = int(config["val_batch_size"])
    eval_interval = int(config["eval_interval"])
    sigma = float(config["sigma"])
    t_eps = float(config["t_eps"])

    history: List[Dict[str, float]] = []
    best_state = None
    best_val = float("inf")

    for step in range(1, steps + 1):
        model.train()
        x0, x1, labels = problem.sample_pairs(variant, batch_size, generator=generator, device=device)
        t = torch.rand((batch_size,), generator=generator, device=device)
        t = t_eps + (1.0 - 2.0 * t_eps) * t
        xt, xt_dot = sample_stochastic_interpolant(x0, x1, t=t, sigma=sigma)
        pred = model(xt, t, labels if conditioned else None)
        loss = torch.mean((pred - xt_dot) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if step == 1 or step % eval_interval == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_x0, val_x1, val_labels = problem.sample_pairs(variant, val_batch_size, generator=generator, device=device)
                val_t = torch.rand((val_batch_size,), generator=generator, device=device)
                val_t = t_eps + (1.0 - 2.0 * t_eps) * val_t
                val_xt, val_xt_dot = sample_stochastic_interpolant(val_x0, val_x1, t=val_t, sigma=sigma)
                val_pred = model(val_xt, val_t, val_labels if conditioned else None)
                val_loss = torch.mean((val_pred - val_xt_dot) ** 2)

            train_value = float(loss.detach().cpu().item())
            val_value = float(val_loss.detach().cpu().item())
            history.append({"step": step, "train_loss": train_value, "val_loss": val_value})

            if val_value < best_val:
                best_val = val_value
                best_state = {
                    "model": model.state_dict(),
                    "history": history,
                    "variant": variant,
                    "conditioned": conditioned,
                    "config": config,
                }

    if best_state is None:
        raise RuntimeError("Training did not record any validation checkpoints.")

    torch.save(best_state, output_checkpoint)
    model.load_state_dict(best_state["model"])
    return model, pd.DataFrame(history)
