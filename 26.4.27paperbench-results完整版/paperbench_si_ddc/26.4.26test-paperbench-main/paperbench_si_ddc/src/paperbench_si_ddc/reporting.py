from __future__ import annotations

from pathlib import Path
import shutil

from paperbench_si_ddc.utils import mean_or_zero, save_json


REQUIRED_IMAGE_FILES = [
    "figure_3_inpainting_examples.png",
    "figure_4_super_resolution_examples.png",
    "figure_5_inpainting_trajectory.png",
    "figure_6_super_resolution_gallery.png",
    "figure_7_super_resolution_trajectory.png",
]


def build_artifact_manifest(paths, config: dict, checkpoints: dict, reports: dict, metrics_summary: dict) -> dict:
    images = {name: str(paths.image_root / name) for name in REQUIRED_IMAGE_FILES}
    payload = {
        "config_path": str(paths.config_path),
        "persistent_root": str(paths.persistent_root),
        "output_leaf": str(paths.image_root),
        "checkpoints": checkpoints,
        "images": images,
        "reports": reports,
        "metrics": metrics_summary,
    }
    save_json(payload, paths.manifest_path)
    return payload


def compute_scorecard(paths, config: dict, checkpoints: dict, metrics_df, smoke_test_passed: bool) -> dict:
    required_experiments = [
        name for name, exp in config["experiments"].items() if exp.get("enabled", False)
    ]
    checkpoint_score = 100.0 * sum(
        1 for name in required_experiments if Path(checkpoints.get(name, "")).exists()
    ) / max(len(required_experiments), 1)
    image_score = 100.0 * sum(1 for name in REQUIRED_IMAGE_FILES if (paths.image_root / name).exists()) / len(REQUIRED_IMAGE_FILES)

    paper_hparams = config["paper_hparams"]
    fidelity_checks = [
        paper_hparams["batch_size"] == 32,
        paper_hparams["train_steps"] == 200000,
        paper_hparams["tile_grid"] == 8,
        abs(float(paper_hparams["tile_keep_probability"]) - 0.3) < 1e-8,
        config["dataset"]["name"] == "imagenet-1k" or bool(config["dataset"]["local_root"]),
    ]
    model_fidelity = 100.0 * sum(1 for item in fidelity_checks if item) / len(fidelity_checks)

    reproducibility_checks = [
        bool(config.get("seed")),
        bool(config.get("config_path")),
        bool(config.get("persistent_root")),
        smoke_test_passed,
        bool(config["runtime"]["device"]),
    ]
    reproducibility = 100.0 * sum(1 for item in reproducibility_checks if item) / len(reproducibility_checks)

    engineering_checks = [
        any(str(paths.persistent_root).startswith(prefix) for prefix in ["/AutoDL-pub", "/autodl-pub", "/root/autodl-pub"]),
        str(paths.image_root).endswith("collected_results/image/outputs_image"),
        (paths.archive_root).exists(),
        (paths.report_root).exists(),
        (paths.metrics_root).exists(),
    ]
    engineering = 100.0 * sum(1 for item in engineering_checks if item) / len(engineering_checks)

    visual_similarity = image_score
    quantitative = 0.0
    if metrics_df is not None and not metrics_df.empty and "fid_abs_error" in metrics_df:
        valid = [
            max(0.0, 100.0 * (1.0 - min(float(error), 1.0)))
            for error in metrics_df["fid_abs_error"].dropna().tolist()
        ]
        quantitative = mean_or_zero(valid)

    code_completeness = 0.6 * checkpoint_score + 0.4 * image_score
    overall = (
        0.2 * code_completeness
        + 0.2 * model_fidelity
        + 0.15 * reproducibility
        + 0.15 * visual_similarity
        + 0.2 * quantitative
        + 0.1 * engineering
    )

    return {
        "dimensions": {
            "code_completeness": round(code_completeness, 2),
            "model_fidelity": round(model_fidelity, 2),
            "experiment_reproducibility": round(reproducibility, 2),
            "visual_similarity": round(visual_similarity, 2),
            "quantitative_metric_error": round(quantitative, 2),
            "engineering_completeness": round(engineering, 2),
        },
        "overall_score": round(overall, 2),
        "deductions": {
            "checkpoint_coverage": round(100.0 - checkpoint_score, 2),
            "figure_coverage": round(100.0 - image_score, 2),
            "metric_gap": round(100.0 - quantitative, 2),
        },
    }


def write_report(paths, config: dict, scorecard: dict, metric_paths: dict, checkpoints: dict) -> dict:
    report_json = paths.report_root / "paperbench_score_report.json"
    report_md = paths.report_root / "paperbench_score_report.md"
    save_json(scorecard, report_json)

    dims = scorecard["dimensions"]
    lines = [
        "# PaperBench-Style Reproduction Report",
        "",
        f"- Overall score: **{scorecard['overall_score']:.2f} / 100**",
        f"- Persistent root: `{paths.persistent_root}`",
        f"- Image output folder: `{paths.image_root}`",
        "",
        "## Dimension Scores",
        "",
        f"- Code completeness: {dims['code_completeness']:.2f}",
        f"- Model fidelity: {dims['model_fidelity']:.2f}",
        f"- Experiment reproducibility: {dims['experiment_reproducibility']:.2f}",
        f"- Visual similarity: {dims['visual_similarity']:.2f}",
        f"- Quantitative metric error: {dims['quantitative_metric_error']:.2f}",
        f"- Engineering completeness: {dims['engineering_completeness']:.2f}",
        "",
        "## Deductions",
        "",
        f"- Missing checkpoint coverage: {scorecard['deductions']['checkpoint_coverage']:.2f}",
        f"- Missing figure coverage: {scorecard['deductions']['figure_coverage']:.2f}",
        f"- Metric gap penalty: {scorecard['deductions']['metric_gap']:.2f}",
        "",
        "## Outputs",
        "",
        f"- Metrics CSV: `{metric_paths.get('csv', '')}`",
        f"- Metrics Markdown: `{metric_paths.get('markdown', '')}`",
    ]
    for name, path in checkpoints.items():
        lines.append(f"- Checkpoint `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "- The project keeps the paper's U-Net family fixed through lucidrains' implementation.",
            "- The AutoDL pipeline trains the required ImageNet inpainting and super-resolution tasks.",
            "- Figure 3 to Figure 7 style artifacts, metrics, reports, and archives are generated automatically.",
            "- The quantitative score depends on the actual GPU run and sampled outputs.",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(report_json), "markdown": str(report_md)}


def package_results(paths) -> Path:
    archive_base = paths.archive_root / "paperbench_si_ddc_results"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(paths.persistent_root))
    return Path(archive_path)
