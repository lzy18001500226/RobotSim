#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROBOTSIM_M0_RUN_DIR:-/tmp/robotsim-issue43-m0}"
CANDIDATE_DIR="${ROBOTSIM_M0_CANDIDATE_DIR:-$RUN_DIR/checkouts/humanoid_vla}"
UNITREE_DIR="${ROBOTSIM_M0_UNITREE_DIR:-$RUN_DIR/checkouts/unitree_mujoco}"
OUTPUT_DIR="${ROBOTSIM_M0_OUTPUT_DIR:-$RUN_DIR/output}"
UPSTREAM_COMMIT="3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12"
UNITREE_COMMIT="1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"

ensure_checkout() {
    local label="$1"
    local url="$2"
    local revision="$3"
    local path="$4"

    if [[ -e "$path" && ! -d "$path/.git" ]]; then
        printf 'Refusing to use non-Git %s path: %s\n' "$label" "$path" >&2
        return 1
    fi
    if [[ ! -d "$path/.git" ]]; then
        mkdir -p "$(dirname "$path")"
        git clone --no-checkout "$url" "$path"
        git -C "$path" fetch origin "$revision"
        git -C "$path" checkout --detach "$revision"
    fi

    local actual
    actual="$(git -C "$path" rev-parse HEAD)"
    if [[ "$actual" != "$revision" ]]; then
        printf '%s checkout must be %s; found %s at %s\n' \
            "$label" "$revision" "$actual" "$path" >&2
        return 1
    fi
    if [[ -n "$(git -C "$path" status --porcelain)" ]]; then
        printf 'Refusing dirty %s checkout: %s\n' "$label" "$path" >&2
        return 1
    fi
}

command -v git >/dev/null || { echo 'git is required' >&2; exit 2; }
command -v uv >/dev/null || { echo 'uv is required to create the pinned Python 3.10 environment' >&2; exit 2; }

mkdir -p "$RUN_DIR/checkouts" "$OUTPUT_DIR"
ensure_checkout \
    "Humanoid VLA" \
    "https://github.com/ozkannceylan/humanoid_vla.git" \
    "$UPSTREAM_COMMIT" \
    "$CANDIDATE_DIR"
ensure_checkout \
    "Unitree MuJoCo meshes" \
    "https://github.com/unitreerobotics/unitree_mujoco.git" \
    "$UNITREE_COMMIT" \
    "$UNITREE_DIR"

MESH_DIR="${ROBOTSIM_M0_MESH_DIR:-$UNITREE_DIR/unitree_robots/g1/meshes}"
[[ -d "$MESH_DIR" ]] || { printf 'G1 mesh directory missing: %s\n' "$MESH_DIR" >&2; exit 2; }

VENV_DIR="$RUN_DIR/venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    uv venv --python 3.10 "$VENV_DIR"
fi
uv pip install --python "$VENV_DIR/bin/python" \
    -r "$ROOT_DIR/simulation/mujoco/requirements-m0.txt"

LOG_PATH="$OUTPUT_DIR/run.log"
RESULT_PATH="$OUTPUT_DIR/m0_result.json"
VIDEO_PATH="$OUTPUT_DIR/m0_pick_place.mp4"
SCREENSHOT_PATH="$OUTPUT_DIR/m0_final.png"

cd "$ROOT_DIR"
set +e
MUJOCO_GL=egl PYTHONPATH="$CANDIDATE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV_DIR/bin/python" -m simulation.mujoco.m0_pick_place \
        --candidate-root "$CANDIDATE_DIR" \
        --mesh-dir "$MESH_DIR" \
        --output-dir "$OUTPUT_DIR/model" \
        --output-json "$RESULT_PATH" \
        --video "$VIDEO_PATH" \
        --screenshot "$SCREENSHOT_PATH" \
    2>&1 | tee "$LOG_PATH"
status=${PIPESTATUS[0]}
set -e
exit "$status"
