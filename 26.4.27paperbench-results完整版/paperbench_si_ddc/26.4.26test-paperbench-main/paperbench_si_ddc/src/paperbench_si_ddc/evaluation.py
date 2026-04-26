from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from paperbench_si_ddc.utils import denorm_to_unit, mean_or_zero, metric_score_from_error, save_json


def save_tensor_folder(images: torch.Tensor, folder: Path, start_index: int = 0) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    images = denorm_to_unit(images).mul(255.0).byte().cpu()
    for offset, image in enumerate(images):
        path = folder / f"{start_index + offset:06d}.png"
        array = image.permute(1, 2, 0).numpy()
        Image.fromarray(array).save(path)
    return start_index + images.shape[0]


def export_reference_set(dataset, folder: Path, limit: int) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    existing = len(list(folder.glob("*.png")))
    if existing >= limit:
        return existing
    end = min(limit, len(dataset))
    for index in range(existing, end):
        image, _ = dataset[index]
        array = denorm_to_unit(image.unsqueeze(0))[0].mul(255.0).byte().permute(1, 2, 0).numpy()
        Image.fromarray(array).save(folder / f"{index:06d}.png")
    return end


def compute_torch_fidelity_metrics(generated_root: Path, reference_root: Path, cuda: bool) -> dict:
    from torch_fidelity import calculate_metrics

    metrics = calculate_metrics(
        input1=str(generated_root),
        input2=str(reference_root),
        fid=True,
        isc=True,
        cuda=bool(cuda),
        batch_size=32,
        save_cpu_ram=True,
        samples_find_deep=True,
    )
    return {
        "fid50k": float(metrics["frechet_inception_distance"]),
        "is_mean": float(metrics["inception_score_mean"]),
        "is_std": float(metrics["inception_score_std"]),
    }


def build_metric_table(measurements: list[dict], reference_metrics: dict) -> pd.DataFrame:
    rows = []
    table2 = reference_metrics.get("table_2_inpainting_fid50k", {})
    table3 = reference_metrics.get("table_3_superres_64_to_256_fid50k", {})
    for measurement in measurements:
        experiment = measurement["experiment"]
        split = measurement.get("split", "")
        paper_fid = None
        if experiment in table2:
            paper_fid = table2[experiment]
        if experiment == "superres_dependent" and split:
            paper_fid = table3.get(f"superres_dependent_{split}")
        abs_error = None if paper_fid is None else abs(measurement["fid50k"] - paper_fid)
        rows.append(
            {
                "experiment": experiment,
                "split": split,
                "fid50k": measurement["fid50k"],
                "is_mean": measurement["is_mean"],
                "is_std": measurement["is_std"],
                "paper_fid50k": paper_fid,
                "fid_abs_error": abs_error,
            }
        )
    return pd.DataFrame(rows)


def summarize_quantitative_score(metrics_df: pd.DataFrame) -> float:
    errors = []
    for _, row in metrics_df.iterrows():
        if pd.isna(row["fid_abs_error"]):
            continue
        errors.append(metric_score_from_error(float(row["fid_abs_error"]), tolerance=1.0))
    return mean_or_zero(errors)


def write_metric_outputs(metrics_df: pd.DataFrame, folder: Path) -> dict:
    csv_path = folder / "reproduction_metrics.csv"
    md_path = folder / "reproduction_metrics.md"
    metrics_df.to_csv(csv_path, index=False)

    lines = [
        "| experiment | split | fid50k | is_mean | is_std | paper_fid50k | fid_abs_error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            "| {experiment} | {split} | {fid50k:.4f} | {is_mean:.4f} | {is_std:.4f} | {paper_fid} | {error} |".format(
                experiment=row["experiment"],
                split=row["split"] or "-",
                fid50k=float(row["fid50k"]),
                is_mean=float(row["is_mean"]),
                is_std=float(row["is_std"]),
                paper_fid="-" if pd.isna(row["paper_fid50k"]) else f"{float(row['paper_fid50k']):.4f}",
                error="-" if pd.isna(row["fid_abs_error"]) else f"{float(row['fid_abs_error']):.4f}",
            )
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "markdown": str(md_path)}
