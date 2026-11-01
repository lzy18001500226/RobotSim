#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  printf 'MuJoCo 3.3.6 Python environment not found: %s\n' "$PYTHON" >&2
  exit 2
fi

exec env MUJOCO_GL=glfw "$PYTHON" "$SCRIPT_DIR/viewer.py" "$@"
