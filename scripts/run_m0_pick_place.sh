#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROBOTSIM_M0_RUN_DIR:-/tmp/robotsim-issue43-m0}"
CANDIDATE_DIR="${ROBOTSIM_M0_CANDIDATE_DIR:-$RUN_DIR/checkouts/humanoid_vla}"
UNITREE_DIR="${ROBOTSIM_M0_UNITREE_DIR:-$RUN_DIR/checkouts/unitree_mujoco}"
OUTPUT_DIR="${ROBOTSIM_M0_OUTPUT_DIR:-$RUN_DIR/output}"
UPSTREAM_COMMIT="3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12"
UNITREE_COMMIT="1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
RUN_ID="${ROBOTSIM_M0_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RESULT_PATH="$OUTPUT_DIR/m0_result.json"
LOG_PATH="$OUTPUT_DIR/run.log"
VIDEO_PATH="$OUTPUT_DIR/m0_pick_place.mp4"
SCREENSHOT_PATH="$OUTPUT_DIR/m0_final.png"

json_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    printf '%s' "$value"
}

robotsim_sha="unknown"
robotsim_dirty="unknown"

write_preflight_result() {
    local state="$1"
    local passed="$2"
    local error="${3:-}"
    mkdir -p "$OUTPUT_DIR"
    cat >"$RESULT_PATH" <<EOF
{
  "issue": 43,
  "demo": "M0 G1 bottle pick-and-place",
  "run_id": "$(json_escape "$RUN_ID")",
  "state": "$(json_escape "$state")",
  "passed": $passed,
  "robotsim_sha": "$(json_escape "$robotsim_sha")",
  "robotsim_dirty": "$robotsim_dirty",
  "seed": 42,
  "upstream_shas": {
    "humanoid_vla": "$UPSTREAM_COMMIT",
    "unitree_mujoco_meshes": "$UNITREE_COMMIT"
  },
  "output_directory": "$(json_escape "$OUTPUT_DIR")",
  "error": "$(json_escape "$error")"
}
EOF
}

fail_preflight() {
    local status="$1"
    shift
    local message="$*"
    printf 'M0 preflight failed: %s\n' "$message" >&2
    write_preflight_result "PREFLIGHT_FAILED" false "$message"
    exit "$status"
}

preflight_error() {
    local status="$?"
    local command="${BASH_COMMAND:-unknown}"
    trap - ERR
    write_preflight_result "PREFLIGHT_FAILED" false "command failed (exit $status): $command"
    exit "$status"
}

trap preflight_error ERR

command -v git >/dev/null || fail_preflight 2 "git is required"
robotsim_sha="$(git -C "$ROOT_DIR" rev-parse HEAD)"
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain)" ]]; then
    robotsim_dirty=true
else
    robotsim_dirty=false
fi
write_preflight_result "PREFLIGHT" false ""

command -v uv >/dev/null || fail_preflight 2 "uv is required to create the pinned Python 3.10 environment"

ensure_checkout() {
    local label="$1"
    local url="$2"
    local revision="$3"
    local path="$4"

    if [[ -e "$path" && ! -d "$path/.git" ]]; then
        fail_preflight 2 "refusing non-Git $label path: $path"
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
        fail_preflight 2 "$label checkout must be $revision; found $actual at $path"
    fi
    if [[ -n "$(git -C "$path" status --porcelain)" ]]; then
        fail_preflight 2 "refusing dirty $label checkout: $path"
    fi
}

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

DEFAULT_MESH_DIR="$UNITREE_DIR/unitree_robots/g1/meshes"
MESH_DIR="${ROBOTSIM_M0_MESH_DIR:-$DEFAULT_MESH_DIR}"
[[ -d "$DEFAULT_MESH_DIR" ]] || fail_preflight 2 "verified G1 mesh directory missing: $DEFAULT_MESH_DIR"
[[ -d "$MESH_DIR" ]] || fail_preflight 2 "G1 mesh directory missing: $MESH_DIR"
default_mesh_resolved="$(cd "$DEFAULT_MESH_DIR" && pwd -P)"
mesh_resolved="$(cd "$MESH_DIR" && pwd -P)"
if [[ "$mesh_resolved" != "$default_mesh_resolved" ]]; then
    fail_preflight 2 "ROBOTSIM_M0_MESH_DIR must resolve to the verified pinned Unitree mesh directory"
fi
MESH_DIR="$mesh_resolved"
MESH_PROVENANCE="unitree_mujoco@$UNITREE_COMMIT:unitree_robots/g1/meshes"

VENV_DIR="$RUN_DIR/venv"
if [[ -x "$VENV_DIR/bin/python" ]]; then
    if ! "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)'; then
        fail_preflight 2 "existing M0 venv is not Python 3.10.x: $VENV_DIR"
    fi
else
    uv venv --python 3.10 "$VENV_DIR"
fi
uv pip install --python "$VENV_DIR/bin/python" \
    -r "$ROOT_DIR/simulation/mujoco/requirements-m0.txt"

trap - ERR
cd "$ROOT_DIR"
set +e
ROBOTSIM_M0_RUN_ID="$RUN_ID" \
ROBOTSIM_M0_MESH_PROVENANCE="$MESH_PROVENANCE" \
MUJOCO_GL=egl \
PYTHONPATH="$CANDIDATE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}" \
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
