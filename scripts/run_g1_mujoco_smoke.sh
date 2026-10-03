#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/robotsim-g1-mujoco.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

python3 - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"G1 MuJoCo smoke requires Python 3.12.x for its reproducible environment; "
        f"found {sys.version.split()[0]}"
    )
print(f"Python {sys.version.split()[0]}")
PY

MODEL_COMMIT="$(awk -F '|' '
  function trim(value) {
    sub(/^[[:space:]]+/, "", value)
    sub(/[[:space:]]+$/, "", value)
    return value
  }
  trim($2) == "unitree_mujoco" {
    print trim($4)
    count++
  }
  END { if (count != 1) exit 1 }
' "$ROOT/third_party/LOCK.md")"
if [[ ! "$MODEL_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Could not resolve the pinned unitree_mujoco commit from third_party/LOCK.md" >&2
  exit 1
fi

MODEL_CHECKOUT="$WORK_DIR/third_party/unitree_mujoco"
mkdir -p "$MODEL_CHECKOUT"
git -C "$MODEL_CHECKOUT" init --quiet
git -C "$MODEL_CHECKOUT" remote add origin \
  https://github.com/unitreerobotics/unitree_mujoco.git
git -C "$MODEL_CHECKOUT" fetch --quiet --no-tags --depth=1 origin "$MODEL_COMMIT"
git -C "$MODEL_CHECKOUT" checkout --quiet --detach FETCH_HEAD
ACTUAL_MODEL_COMMIT="$(git -C "$MODEL_CHECKOUT" rev-parse HEAD)"
if [[ "$ACTUAL_MODEL_COMMIT" != "$MODEL_COMMIT" ]]; then
  echo "Fetched unitree_mujoco $ACTUAL_MODEL_COMMIT instead of $MODEL_COMMIT" >&2
  exit 1
fi

python3 -m venv "$WORK_DIR/venv"
"$WORK_DIR/venv/bin/python" -m pip --disable-pip-version-check --no-cache-dir \
  install -r "$ROOT/simulation/mujoco/requirements-smoke.txt"

cd "$ROOT"
ROBOTSIM_UNITREE_MUJOCO_DIR="$WORK_DIR/third_party/unitree_mujoco" \
  "$WORK_DIR/venv/bin/python" -m unittest discover \
    -s simulation/mujoco/tests -v
