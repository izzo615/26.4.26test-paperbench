#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/autodl_full.json"
STAGES="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --stages)
      STAGES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

python -m pip install -r "${ROOT_DIR}/requirements.txt"
python "${ROOT_DIR}/scripts/run_full_reproduction.py" --config "${CONFIG_PATH}" --stages "${STAGES}"
