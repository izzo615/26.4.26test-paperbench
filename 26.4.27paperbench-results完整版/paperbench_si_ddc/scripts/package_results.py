from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paperbench_si_ddc.config import load_config
from paperbench_si_ddc.paths import ProjectPaths
from paperbench_si_ddc.reporting import package_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack the persistent reproduction outputs into a zip archive.")
    parser.add_argument("--config", default="configs/autodl_full.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    paths = ProjectPaths.from_config(ROOT, config_path, config)
    archive_path = package_results(paths)
    print(archive_path)


if __name__ == "__main__":
    main()
