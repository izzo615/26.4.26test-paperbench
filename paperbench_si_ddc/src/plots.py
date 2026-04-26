from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY_NAMES = {
    "independent": "Independent coupling",
    "conditioned": "Class-conditioned baseline",
    "dependent": "Data-dependent coupling",
}

DISPLAY_COLORS = {
    "independent": "#d1495b",
    "conditioned": "#00798c",
    "dependent": "#edae49",
}


def _setup_axis(ax):
    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(-6.0, 6.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_loss_curves(histories: Dict[str, pd.DataFrame], output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    for key, history in histories.items():
        plt.plot(history["step"], history["train_loss"], label=DISPLAY_NAMES[key], color=DISPLAY_COLORS[key], linewidth=2)
        plt.plot(history["step"], history["val_loss"], color=DISPLAY_COLORS[key], linewidth=1, linestyle="--", alpha=0.6)
    plt.xlabel("Training step")
    plt.ylabel("Velocity regression loss")
    plt.title("Stochastic interpolant training curves")
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_transport_snapshots(
    snapshots: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(11, 11))
    time_keys = ["t0", "t_mid", "t1"]
    time_titles = ["t = 0.0", "t = 0.5", "t = 1.0"]

    for row, key in enumerate(["independent", "conditioned", "dependent"]):
        for col, (time_key, title) in enumerate(zip(time_keys, time_titles)):
            ax = axes[row, col]
            _setup_axis(ax)
            points = snapshots[key][time_key]
            ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.35, color=DISPLAY_COLORS[key], linewidths=0)
            if col == 0:
                ax.set_ylabel(DISPLAY_NAMES[key], fontsize=11)
            if row == 0:
                ax.set_title(title)
    fig.suptitle("Probability-flow snapshots for the three couplings", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_generated_samples(
    target: np.ndarray,
    generated: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    panels = [
        ("Reference target", target, "#2f2f2f"),
        (DISPLAY_NAMES["independent"], generated["independent"], DISPLAY_COLORS["independent"]),
        (DISPLAY_NAMES["conditioned"], generated["conditioned"], DISPLAY_COLORS["conditioned"]),
        (DISPLAY_NAMES["dependent"], generated["dependent"], DISPLAY_COLORS["dependent"]),
    ]
    for ax, (title, points, color) in zip(axes.flat, panels):
        _setup_axis(ax)
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.35, color=color, linewidths=0)
        ax.set_title(title)
    fig.suptitle("Generated samples at t = 1", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_trajectories(
    trajectories: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 4.5))
    for ax, key in zip(axes, ["independent", "conditioned", "dependent"]):
        _setup_axis(ax)
        traj = trajectories[key]
        for idx in range(traj.shape[1]):
            ax.plot(traj[:, idx, 0], traj[:, idx, 1], color=DISPLAY_COLORS[key], alpha=0.2, linewidth=1.0)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], s=10, color="#1f1f1f", alpha=0.8, linewidths=0)
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=10, color=DISPLAY_COLORS[key], alpha=0.8, linewidths=0)
        ax.set_title(DISPLAY_NAMES[key])
    fig.suptitle("Sample trajectories under the learned probability flows", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
