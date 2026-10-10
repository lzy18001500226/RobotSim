#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROBOTSIM_MUJOCO_PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'MuJoCo Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 2
fi

cd "$REPO_ROOT"
exec env MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON_BIN" -u \
  simulation/mujoco/run_g1_dfq_physical_grasp.py "$@"
