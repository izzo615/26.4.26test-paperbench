from __future__ import annotations

import compileall
import json
from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "autodl_full.json"
    json.loads(config_path.read_text(encoding="utf-8"))

    success = compileall.compile_dir(str(root / "src"), quiet=1)
    success = compileall.compile_dir(str(root / "scripts"), quiet=1) and success
    if not success:
        raise SystemExit("compileall failed")

    sys.stdout.write("local_smoke_test: ok\n")


if __name__ == "__main__":
    main()
