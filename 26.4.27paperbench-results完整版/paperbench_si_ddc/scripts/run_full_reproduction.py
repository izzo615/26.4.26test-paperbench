from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paperbench_si_ddc.config import load_config
from paperbench_si_ddc.pipeline import ReproductionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full SI-DDC PaperBench-style reproduction pipeline.")
    parser.add_argument("--config", default="configs/autodl_full.json", help="Path to the JSON config file.")
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages. Supported: all, train, sample, evaluate, report, package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    pipeline = ReproductionPipeline(project_root=ROOT, config_path=config_path, config=config)
    pipeline.run(stage_argument=args.stages)


if __name__ == "__main__":
    main()
