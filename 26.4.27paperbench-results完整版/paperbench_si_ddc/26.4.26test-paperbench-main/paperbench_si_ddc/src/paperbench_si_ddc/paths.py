from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _resolve_persistent_root(raw_path: str) -> Path:
    path = Path(raw_path)
    if str(path).startswith(("C:\\", "C:/")):
        return path
    if path.exists():
        return path
    variants = []
    path_text = str(path)
    if path_text.startswith("/AutoDL-pub"):
        suffix = path_text[len("/AutoDL-pub") :].lstrip("/")
        variants.extend(
            [
                Path("/AutoDL-pub") / suffix,
                Path("/autodl-pub") / suffix,
                Path("/root/autodl-pub") / suffix,
            ]
        )
    for candidate in variants:
        if candidate.parent.exists() or candidate.exists():
            return candidate
    return path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    config_path: Path
    persistent_root: Path
    image_root: Path
    checkpoint_root: Path
    log_root: Path
    metrics_root: Path
    report_root: Path
    archive_root: Path
    cache_root: Path
    reference_root: Path
    manifest_path: Path

    @classmethod
    def from_config(cls, project_root: Path, config_path: Path, config: dict) -> "ProjectPaths":
        persistent_root = _resolve_persistent_root(config["persistent_root"])
        image_root = persistent_root / config["output_leaf"]
        checkpoint_root = persistent_root / "checkpoints"
        log_root = persistent_root / "logs"
        metrics_root = persistent_root / "metrics"
        report_root = persistent_root / "reports"
        archive_root = persistent_root / "archives"
        cache_root = persistent_root / "cache"
        reference_root = persistent_root / "reference_sets"
        manifest_path = report_root / "artifact_manifest.json"

        for directory in [
            persistent_root,
            image_root,
            checkpoint_root,
            log_root,
            metrics_root,
            report_root,
            archive_root,
            cache_root,
            reference_root,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        return cls(
            project_root=project_root,
            config_path=config_path,
            persistent_root=persistent_root,
            image_root=image_root,
            checkpoint_root=checkpoint_root,
            log_root=log_root,
            metrics_root=metrics_root,
            report_root=report_root,
            archive_root=archive_root,
            cache_root=cache_root,
            reference_root=reference_root,
            manifest_path=manifest_path,
        )
