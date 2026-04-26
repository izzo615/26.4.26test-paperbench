from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    script = Path(__file__).resolve().with_name("run_full_reproduction.py")
    command = [sys.executable, str(script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
