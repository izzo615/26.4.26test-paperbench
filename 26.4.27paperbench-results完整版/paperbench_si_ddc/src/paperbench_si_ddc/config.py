from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["config_path"] = str(path)
    reference_path = path.parent / "reference_metrics.json"
    if reference_path.exists():
        config["reference_metrics"] = json.loads(reference_path.read_text(encoding="utf-8"))
    else:
        config["reference_metrics"] = {}
    return config


def normalize_stage_argument(stage_argument: str) -> list[str]:
    tokens = [token.strip().lower() for token in stage_argument.split(",") if token.strip()]
    if not tokens:
        return ["all"]
    if "all" in tokens:
        return ["train", "sample", "evaluate", "report", "package"]
    return tokens
