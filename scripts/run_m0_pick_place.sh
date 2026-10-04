#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${ROBOTSIM_M0_RUN_DIR:-/tmp/robotsim-issue43-m0}"
CANDIDATE_DIR="${ROBOTSIM_M0_CANDIDATE_DIR:-$RUN_ROOT/checkouts/humanoid_vla}"
UNITREE_DIR="${ROBOTSIM_M0_UNITREE_DIR:-$RUN_ROOT/checkouts/unitree_mujoco}"
OUTPUT_ROOT="${ROBOTSIM_M0_OUTPUT_DIR:-/tmp/robotsim-issue43-m0/output}"
UPSTREAM_COMMIT="3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12"
UNITREE_COMMIT="1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
GRASP_REFERENCE_COMMIT="ace298393ec6cadc1f4a66e70a3311e1d2c4d7ff"
SEED="${ROBOTSIM_M0_SEED:-42}"
if [[ "$SEED" =~ ^[0-9]+$ ]]; then SEED_JSON="$SEED"; else SEED_JSON=null; fi

json_quote() {
    local value=${1-}
    local out=""
    local ch byte escaped i
    local LC_ALL=C

    # Encode byte-by-byte so every JSON control byte U+0001..U+001F is
    # emitted as \\u00XX. Bash variables/POSIX paths cannot contain NUL.
    for ((i = 0; i < ${#value}; i++)); do
        ch=${value:i:1}
        case "$ch" in
            "\\")
                out+='\\\\'
                ;;
            "\"")
                out+='\\"'
                ;;
            *)
                printf -v byte '%d' "'$ch"
                if (( byte < 32 )); then
                    printf -v escaped '\\u%04x' "$byte"
                    out+="$escaped"
                else
                    out+="$ch"
                fi
                ;;
        esac
    done
    printf '"%s"' "$out"
}

write_result() {
    local state="$1"
    local error="$2"
    {
        printf '{\n'
        printf '  "issue": 43,\n'
        printf '  "demo": "M0 G1 bottle pick-and-place",\n'
        printf '  "run_id": %s,\n' "$(json_quote "$RUN_ID")"
        printf '  "state": %s,\n' "$(json_quote "$state")"
        printf '  "stage": %s,\n' "$(json_quote "$CURRENT_STAGE")"
        printf '  "passed": false,\n'
        printf '  "complete": false,\n'
        printf '  "error": %s,\n' "$(json_quote "$error")"
        printf '  "robotsim_sha": %s,\n' "$(json_quote "$ROBOTSIM_SHA")"
        printf '  "robotsim_dirty": %s,\n' "$ROBOTSIM_DIRTY_JSON"
        printf '  "python_version": %s,\n' "$(json_quote "$PYTHON_VERSION")"
        printf '  "mujoco_version": %s,\n' "$(json_quote "$MUJOCO_VERSION")"
        printf '  "numpy_version": %s,\n' "$(json_quote "$NUMPY_VERSION")"
        printf '  "h5py_version": %s,\n' "$(json_quote "$H5PY_VERSION")"
        printf '  "opencv_version": %s,\n' "$(json_quote "$OPENCV_VERSION")"
        printf '  "upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s, "grasp_reference": %s},\n' \
            "$(json_quote "$CANDIDATE_SHA")" "$(json_quote "$UNITREE_SHA")" "$(json_quote "$GRASP_REFERENCE_COMMIT")"
        printf '  "upstream_dirty": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$CANDIDATE_DIRTY_JSON" "$UNITREE_DIRTY_JSON"
        printf '  "expected_upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$(json_quote "$UPSTREAM_COMMIT")" "$(json_quote "$UNITREE_COMMIT")"
        printf '  "mesh_provenance": {"path_class": %s, "path_relative_to_unitree": %s},\n' \
            "$(json_quote "$MESH_PATH_CLASS")" "$(json_quote "$MESH_RELATIVE_PATH")"
        printf '  "seed": %s,\n' "$SEED_JSON"
        printf '  "acceptance_thresholds": {"target_table_margin_m": 0.03, "bilateral_grasp_force_n_per_palm": 2.0, "bilateral_grasp_frames": 5, "minimum_lift_height_m": 0.05, "contact_free_release_frames": 5, "stable_duration_s": 1.0, "linear_speed_m_s": 0.03, "angular_speed_rad_s": 0.20, "stable_position_radius_m": 0.02, "control_step_translation_m": 0.20, "physics_step_translation_m": 0.005, "physics_step_angular_jump_rad": 0.025, "weld_event_translation_snap_m": 0.002, "weld_event_angular_snap_rad": 0.017453292519943295, "maximum_penetration_m": 0.025, "drop_height_m": 0.50},\n'
        printf '  "output_directory": %s\n' "$(json_quote "$OUTPUT_DIR")"
        printf '}\n'
    } > "$RESULT_PATH"
}

CURRENT_STAGE="INITIALIZE"
ROBOTSIM_SHA="not_checked"
ROBOTSIM_DIRTY_JSON=null
PYTHON_VERSION="not_run"
MUJOCO_VERSION="not_run"
NUMPY_VERSION="not_run"
H5PY_VERSION="not_run"
OPENCV_VERSION="not_run"
CANDIDATE_SHA="not_checked"
UNITREE_SHA="not_checked"
CANDIDATE_DIRTY_JSON=null
UNITREE_DIRTY_JSON=null
MESH_PATH_CLASS="not_checked"
MESH_RELATIVE_PATH="not_checked"

RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"

bootstrap_failure() {
    local message="$1"
    OUTPUT_DIR="/tmp/robotsim-issue43-m0/bootstrap-failures/$RUN_ID"
    if ! mkdir -p "$OUTPUT_DIR"; then
        printf '%s; could not persist a failure record under the fallback directory\n' "$message" >&2
        return 1
    fi
    RESULT_PATH="$OUTPUT_DIR/m0_result.json"
    write_result "PREFLIGHT" ""
    CURRENT_STAGE="OUTPUT_DIRECTORY"
    write_result "FAILED" "$message"
    printf '%s; failure result: %s\n' "$message" "$RESULT_PATH" >&2
}

if ! mkdir -p "$OUTPUT_ROOT" 2>/dev/null; then
    bootstrap_failure "Could not create configured M0 output root"
    exit 2
fi
while [[ -e "$OUTPUT_DIR" ]]; do
    RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
    OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
done
if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
    bootstrap_failure "Could not create unique M0 run directory"
    exit 2
fi
RESULT_PATH="$OUTPUT_DIR/m0_result.json"
write_result "PREFLIGHT" ""

on_exit() {
    local status=$?
    if (( status != 0 )); then
        local result_run_id=""
        local result_state=""
        if [[ -f "$RESULT_PATH" ]]; then
            result_run_id="$(sed -n 's/^[[:space:]]*"run_id":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
            result_state="$(sed -n 's/^[[:space:]]*"state":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
        fi
        if [[ "$result_run_id" != "$RUN_ID" || "$result_state" != "FAILED" ]]; then
            write_result "FAILED" "Preflight or launcher failed at ${CURRENT_STAGE} (exit ${status})"
        fi
        printf 'M0 invocation %s failed at %s; failure result: %s\n' \
            "$RUN_ID" "$CURRENT_STAGE" "$RESULT_PATH" >&2
    fi
}
trap on_exit EXIT

CURRENT_STAGE="TOOL_CHECK"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    printf 'ROBOTSIM_M0_SEED must be a non-negative integer\n' >&2
    exit 2
fi
command -v git >/dev/null || { printf 'git is required\n' >&2; exit 2; }
command -v uv >/dev/null || { printf 'uv is required to create the pinned Python 3.10 environment\n' >&2; exit 2; }
command -v realpath >/dev/null || { printf 'realpath is required for mesh provenance validation\n' >&2; exit 2; }

CURRENT_STAGE="ROBOTSIM_IDENTITY"
ROBOTSIM_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD)"
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=all)" ]]; then
    ROBOTSIM_DIRTY_JSON=true
else
    ROBOTSIM_DIRTY_JSON=false
fi
write_result "PREFLIGHT" ""

ensure_checkout() {
    local label="$1"
    local url="$2"
    local revision="$3"
    local path="$4"
    local sha_var="$5"
    local dirty_var="${sha_var%_SHA}_DIRTY_JSON"

    if [[ -e "$path" && ! -d "$path/.git" ]]; then
        printf 'Refusing to use non-Git %s path\n' "$label" >&2
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
    printf -v "$sha_var" '%s' "$actual"
    write_result "PREFLIGHT" ""
    if [[ "$actual" != "$revision" ]]; then
        printf '%s checkout has the wrong pinned revision\n' "$label" >&2
        return 1
    fi

    local status_output
    status_output="$(git -C "$path" status --porcelain --untracked-files=all)"
    if [[ -n "$status_output" ]]; then
        printf -v "$dirty_var" '%s' true
        write_result "PREFLIGHT" ""
        printf 'Refusing dirty %s checkout\n' "$label" >&2
        return 1
    fi
    printf -v "$dirty_var" '%s' false
    write_result "PREFLIGHT" ""
}

CURRENT_STAGE="HUMANOID_VLA_CHECKOUT"
ensure_checkout \
    "Humanoid VLA" \
    "https://github.com/ozkannceylan/humanoid_vla.git" \
    "$UPSTREAM_COMMIT" \
    "$CANDIDATE_DIR" \
    CANDIDATE_SHA

CURRENT_STAGE="UNITREE_CHECKOUT"
ensure_checkout \
    "Unitree MuJoCo meshes" \
    "https://github.com/unitreerobotics/unitree_mujoco.git" \
    "$UNITREE_COMMIT" \
    "$UNITREE_DIR" \
    UNITREE_SHA

CURRENT_STAGE="MESH_PROVENANCE"
UNITREE_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR")"
G1_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR_RESOLVED/unitree_robots/g1")"
case "$G1_DIR_RESOLVED" in
    "$UNITREE_DIR_RESOLVED"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="g1_model_directory_resolves_outside_unitree_checkout"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 asset directory escapes the pinned Unitree checkout\n' >&2
        exit 2
        ;;
esac
VERIFIED_MESH_ROOT="$(realpath -e "$G1_DIR_RESOLVED/meshes")"
case "$VERIFIED_MESH_ROOT" in
    "$G1_DIR_RESOLVED/meshes"|"$G1_DIR_RESOLVED/meshes"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="resolved_g1_mesh_root_escapes_verified_tree"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 mesh root escapes the verified G1 asset directory\n' >&2
        exit 2
        ;;
esac
MESH_REQUESTED="${ROBOTSIM_M0_MESH_DIR:-$VERIFIED_MESH_ROOT}"
if ! MESH_DIR="$(realpath -e "$MESH_REQUESTED")" || [[ ! -d "$MESH_DIR" ]]; then
    MESH_PATH_CLASS="missing_or_unresolvable"
    MESH_RELATIVE_PATH="not_available"
    write_result "PREFLIGHT" ""
    printf 'G1 mesh directory is missing or cannot be resolved\n' >&2
    exit 2
fi
case "$MESH_DIR" in
    "$VERIFIED_MESH_ROOT"|"$VERIFIED_MESH_ROOT"/*)
        MESH_PATH_CLASS="pinned_unitree_g1_mesh_tree"
        MESH_RELATIVE_PATH="$(realpath --relative-to="$UNITREE_DIR_RESOLVED" "$MESH_DIR")"
        ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="outside_verified_unitree_g1_mesh_tree"
        write_result "PREFLIGHT" ""
        printf 'ROBOTSIM_M0_MESH_DIR must resolve inside the pinned Unitree G1 mesh tree\n' >&2
        exit 2
        ;;
esac
write_result "PREFLIGHT" ""

CURRENT_STAGE="PYTHON_ENVIRONMENT"
VENV_DIR="$RUN_ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
if [[ -e "$VENV_PYTHON" ]]; then
    if [[ ! -x "$VENV_PYTHON" ]]; then
        printf 'Existing M0 venv Python is not executable\n' >&2
        exit 2
    fi
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'Existing M0 venv must use Python 3.10.x; found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
else
    uv venv --python 3.10 "$VENV_DIR"
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'uv created an incompatible Python venv; expected 3.10.x, found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
fi

CURRENT_STAGE="DEPENDENCY_INSTALL"
uv pip install --python "$VENV_PYTHON" -r "$ROOT_DIR/simulation/mujoco/requirements-m0.txt"
read -r MUJOCO_VERSION NUMPY_VERSION H5PY_VERSION OPENCV_VERSION < <(
    "$VENV_PYTHON" -c 'import cv2,h5py,mujoco,numpy; print(mujoco.__version__, numpy.__version__, h5py.__version__, cv2.__version__)'
)
write_result "PREFLIGHT" ""

LOG_PATH="$OUTPUT_DIR/run.log"
VIDEO_PATH="$OUTPUT_DIR/m0_pick_place.mp4"
SCREENSHOT_PATH="$OUTPUT_DIR/m0_final.png"
CURRENT_STAGE="RUNTIME"
cd "$ROOT_DIR"
set +e
MUJOCO_GL=egl PYTHONPATH="$CANDIDATE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV_PYTHON" -m simulation.mujoco.m0_pick_place \
        --candidate-root "$CANDIDATE_DIR" \
        --unitree-root "$UNITREE_DIR" \
        --mesh-dir "$MESH_DIR" \
        --output-dir "$OUTPUT_DIR/model" \
        --output-json "$RESULT_PATH" \
        --video "$VIDEO_PATH" \
        --screenshot "$SCREENSHOT_PATH" \
        --run-id "$RUN_ID" \
        --seed "$SEED" \
    2>&1 | tee "$LOG_PATH"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if (( status == 0 && pipeline_status[1] != 0 )); then
    status=${pipeline_status[1]}
fi
set -e
exit "$status"
\b'/\\b}
    value=${value//
write_result() {
    local state="$1"
    local error="$2"
    {
        printf '{\n'
        printf '  "issue": 43,\n'
        printf '  "demo": "M0 G1 bottle pick-and-place",\n'
        printf '  "run_id": %s,\n' "$(json_quote "$RUN_ID")"
        printf '  "state": %s,\n' "$(json_quote "$state")"
        printf '  "stage": %s,\n' "$(json_quote "$CURRENT_STAGE")"
        printf '  "passed": false,\n'
        printf '  "complete": false,\n'
        printf '  "error": %s,\n' "$(json_quote "$error")"
        printf '  "robotsim_sha": %s,\n' "$(json_quote "$ROBOTSIM_SHA")"
        printf '  "robotsim_dirty": %s,\n' "$ROBOTSIM_DIRTY_JSON"
        printf '  "python_version": %s,\n' "$(json_quote "$PYTHON_VERSION")"
        printf '  "mujoco_version": %s,\n' "$(json_quote "$MUJOCO_VERSION")"
        printf '  "numpy_version": %s,\n' "$(json_quote "$NUMPY_VERSION")"
        printf '  "h5py_version": %s,\n' "$(json_quote "$H5PY_VERSION")"
        printf '  "opencv_version": %s,\n' "$(json_quote "$OPENCV_VERSION")"
        printf '  "upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s, "grasp_reference": %s},\n' \
            "$(json_quote "$CANDIDATE_SHA")" "$(json_quote "$UNITREE_SHA")" "$(json_quote "$GRASP_REFERENCE_COMMIT")"
        printf '  "upstream_dirty": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$CANDIDATE_DIRTY_JSON" "$UNITREE_DIRTY_JSON"
        printf '  "expected_upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$(json_quote "$UPSTREAM_COMMIT")" "$(json_quote "$UNITREE_COMMIT")"
        printf '  "mesh_provenance": {"path_class": %s, "path_relative_to_unitree": %s},\n' \
            "$(json_quote "$MESH_PATH_CLASS")" "$(json_quote "$MESH_RELATIVE_PATH")"
        printf '  "seed": %s,\n' "$SEED_JSON"
        printf '  "acceptance_thresholds": {"target_table_margin_m": 0.03, "bilateral_grasp_force_n_per_palm": 2.0, "bilateral_grasp_frames": 5, "minimum_lift_height_m": 0.05, "contact_free_release_frames": 5, "stable_duration_s": 1.0, "linear_speed_m_s": 0.03, "angular_speed_rad_s": 0.20, "stable_position_radius_m": 0.02, "control_step_translation_m": 0.20, "physics_step_translation_m": 0.005, "physics_step_angular_jump_rad": 0.025, "weld_event_translation_snap_m": 0.002, "weld_event_angular_snap_rad": 0.017453292519943295, "maximum_penetration_m": 0.025, "drop_height_m": 0.50},\n'
        printf '  "output_directory": %s\n' "$(json_quote "$OUTPUT_DIR")"
        printf '}\n'
    } > "$RESULT_PATH"
}

CURRENT_STAGE="INITIALIZE"
ROBOTSIM_SHA="not_checked"
ROBOTSIM_DIRTY_JSON=null
PYTHON_VERSION="not_run"
MUJOCO_VERSION="not_run"
NUMPY_VERSION="not_run"
H5PY_VERSION="not_run"
OPENCV_VERSION="not_run"
CANDIDATE_SHA="not_checked"
UNITREE_SHA="not_checked"
CANDIDATE_DIRTY_JSON=null
UNITREE_DIRTY_JSON=null
MESH_PATH_CLASS="not_checked"
MESH_RELATIVE_PATH="not_checked"

RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"

bootstrap_failure() {
    local message="$1"
    OUTPUT_DIR="/tmp/robotsim-issue43-m0/bootstrap-failures/$RUN_ID"
    if ! mkdir -p "$OUTPUT_DIR"; then
        printf '%s; could not persist a failure record under the fallback directory\n' "$message" >&2
        return 1
    fi
    RESULT_PATH="$OUTPUT_DIR/m0_result.json"
    write_result "PREFLIGHT" ""
    CURRENT_STAGE="OUTPUT_DIRECTORY"
    write_result "FAILED" "$message"
    printf '%s; failure result: %s\n' "$message" "$RESULT_PATH" >&2
}

if ! mkdir -p "$OUTPUT_ROOT" 2>/dev/null; then
    bootstrap_failure "Could not create configured M0 output root"
    exit 2
fi
while [[ -e "$OUTPUT_DIR" ]]; do
    RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
    OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
done
if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
    bootstrap_failure "Could not create unique M0 run directory"
    exit 2
fi
RESULT_PATH="$OUTPUT_DIR/m0_result.json"
write_result "PREFLIGHT" ""

on_exit() {
    local status=$?
    if (( status != 0 )); then
        local result_run_id=""
        local result_state=""
        if [[ -f "$RESULT_PATH" ]]; then
            result_run_id="$(sed -n 's/^[[:space:]]*"run_id":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
            result_state="$(sed -n 's/^[[:space:]]*"state":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
        fi
        if [[ "$result_run_id" != "$RUN_ID" || "$result_state" != "FAILED" ]]; then
            write_result "FAILED" "Preflight or launcher failed at ${CURRENT_STAGE} (exit ${status})"
        fi
        printf 'M0 invocation %s failed at %s; failure result: %s\n' \
            "$RUN_ID" "$CURRENT_STAGE" "$RESULT_PATH" >&2
    fi
}
trap on_exit EXIT

CURRENT_STAGE="TOOL_CHECK"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    printf 'ROBOTSIM_M0_SEED must be a non-negative integer\n' >&2
    exit 2
fi
command -v git >/dev/null || { printf 'git is required\n' >&2; exit 2; }
command -v uv >/dev/null || { printf 'uv is required to create the pinned Python 3.10 environment\n' >&2; exit 2; }
command -v realpath >/dev/null || { printf 'realpath is required for mesh provenance validation\n' >&2; exit 2; }

CURRENT_STAGE="ROBOTSIM_IDENTITY"
ROBOTSIM_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD)"
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=all)" ]]; then
    ROBOTSIM_DIRTY_JSON=true
else
    ROBOTSIM_DIRTY_JSON=false
fi
write_result "PREFLIGHT" ""

ensure_checkout() {
    local label="$1"
    local url="$2"
    local revision="$3"
    local path="$4"
    local sha_var="$5"
    local dirty_var="${sha_var%_SHA}_DIRTY_JSON"

    if [[ -e "$path" && ! -d "$path/.git" ]]; then
        printf 'Refusing to use non-Git %s path\n' "$label" >&2
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
    printf -v "$sha_var" '%s' "$actual"
    write_result "PREFLIGHT" ""
    if [[ "$actual" != "$revision" ]]; then
        printf '%s checkout has the wrong pinned revision\n' "$label" >&2
        return 1
    fi

    local status_output
    status_output="$(git -C "$path" status --porcelain --untracked-files=all)"
    if [[ -n "$status_output" ]]; then
        printf -v "$dirty_var" '%s' true
        write_result "PREFLIGHT" ""
        printf 'Refusing dirty %s checkout\n' "$label" >&2
        return 1
    fi
    printf -v "$dirty_var" '%s' false
    write_result "PREFLIGHT" ""
}

CURRENT_STAGE="HUMANOID_VLA_CHECKOUT"
ensure_checkout \
    "Humanoid VLA" \
    "https://github.com/ozkannceylan/humanoid_vla.git" \
    "$UPSTREAM_COMMIT" \
    "$CANDIDATE_DIR" \
    CANDIDATE_SHA

CURRENT_STAGE="UNITREE_CHECKOUT"
ensure_checkout \
    "Unitree MuJoCo meshes" \
    "https://github.com/unitreerobotics/unitree_mujoco.git" \
    "$UNITREE_COMMIT" \
    "$UNITREE_DIR" \
    UNITREE_SHA

CURRENT_STAGE="MESH_PROVENANCE"
UNITREE_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR")"
G1_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR_RESOLVED/unitree_robots/g1")"
case "$G1_DIR_RESOLVED" in
    "$UNITREE_DIR_RESOLVED"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="g1_model_directory_resolves_outside_unitree_checkout"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 asset directory escapes the pinned Unitree checkout\n' >&2
        exit 2
        ;;
esac
VERIFIED_MESH_ROOT="$(realpath -e "$G1_DIR_RESOLVED/meshes")"
case "$VERIFIED_MESH_ROOT" in
    "$G1_DIR_RESOLVED/meshes"|"$G1_DIR_RESOLVED/meshes"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="resolved_g1_mesh_root_escapes_verified_tree"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 mesh root escapes the verified G1 asset directory\n' >&2
        exit 2
        ;;
esac
MESH_REQUESTED="${ROBOTSIM_M0_MESH_DIR:-$VERIFIED_MESH_ROOT}"
if ! MESH_DIR="$(realpath -e "$MESH_REQUESTED")" || [[ ! -d "$MESH_DIR" ]]; then
    MESH_PATH_CLASS="missing_or_unresolvable"
    MESH_RELATIVE_PATH="not_available"
    write_result "PREFLIGHT" ""
    printf 'G1 mesh directory is missing or cannot be resolved\n' >&2
    exit 2
fi
case "$MESH_DIR" in
    "$VERIFIED_MESH_ROOT"|"$VERIFIED_MESH_ROOT"/*)
        MESH_PATH_CLASS="pinned_unitree_g1_mesh_tree"
        MESH_RELATIVE_PATH="$(realpath --relative-to="$UNITREE_DIR_RESOLVED" "$MESH_DIR")"
        ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="outside_verified_unitree_g1_mesh_tree"
        write_result "PREFLIGHT" ""
        printf 'ROBOTSIM_M0_MESH_DIR must resolve inside the pinned Unitree G1 mesh tree\n' >&2
        exit 2
        ;;
esac
write_result "PREFLIGHT" ""

CURRENT_STAGE="PYTHON_ENVIRONMENT"
VENV_DIR="$RUN_ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
if [[ -e "$VENV_PYTHON" ]]; then
    if [[ ! -x "$VENV_PYTHON" ]]; then
        printf 'Existing M0 venv Python is not executable\n' >&2
        exit 2
    fi
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'Existing M0 venv must use Python 3.10.x; found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
else
    uv venv --python 3.10 "$VENV_DIR"
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'uv created an incompatible Python venv; expected 3.10.x, found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
fi

CURRENT_STAGE="DEPENDENCY_INSTALL"
uv pip install --python "$VENV_PYTHON" -r "$ROOT_DIR/simulation/mujoco/requirements-m0.txt"
read -r MUJOCO_VERSION NUMPY_VERSION H5PY_VERSION OPENCV_VERSION < <(
    "$VENV_PYTHON" -c 'import cv2,h5py,mujoco,numpy; print(mujoco.__version__, numpy.__version__, h5py.__version__, cv2.__version__)'
)
write_result "PREFLIGHT" ""

LOG_PATH="$OUTPUT_DIR/run.log"
VIDEO_PATH="$OUTPUT_DIR/m0_pick_place.mp4"
SCREENSHOT_PATH="$OUTPUT_DIR/m0_final.png"
CURRENT_STAGE="RUNTIME"
cd "$ROOT_DIR"
set +e
MUJOCO_GL=egl PYTHONPATH="$CANDIDATE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV_PYTHON" -m simulation.mujoco.m0_pick_place \
        --candidate-root "$CANDIDATE_DIR" \
        --unitree-root "$UNITREE_DIR" \
        --mesh-dir "$MESH_DIR" \
        --output-dir "$OUTPUT_DIR/model" \
        --output-json "$RESULT_PATH" \
        --video "$VIDEO_PATH" \
        --screenshot "$SCREENSHOT_PATH" \
        --run-id "$RUN_ID" \
        --seed "$SEED" \
    2>&1 | tee "$LOG_PATH"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if (( status == 0 && pipeline_status[1] != 0 )); then
    status=${pipeline_status[1]}
fi
set -e
exit "$status"
\f'/\\f}
    value=${value//
write_result() {
    local state="$1"
    local error="$2"
    {
        printf '{\n'
        printf '  "issue": 43,\n'
        printf '  "demo": "M0 G1 bottle pick-and-place",\n'
        printf '  "run_id": %s,\n' "$(json_quote "$RUN_ID")"
        printf '  "state": %s,\n' "$(json_quote "$state")"
        printf '  "stage": %s,\n' "$(json_quote "$CURRENT_STAGE")"
        printf '  "passed": false,\n'
        printf '  "complete": false,\n'
        printf '  "error": %s,\n' "$(json_quote "$error")"
        printf '  "robotsim_sha": %s,\n' "$(json_quote "$ROBOTSIM_SHA")"
        printf '  "robotsim_dirty": %s,\n' "$ROBOTSIM_DIRTY_JSON"
        printf '  "python_version": %s,\n' "$(json_quote "$PYTHON_VERSION")"
        printf '  "mujoco_version": %s,\n' "$(json_quote "$MUJOCO_VERSION")"
        printf '  "numpy_version": %s,\n' "$(json_quote "$NUMPY_VERSION")"
        printf '  "h5py_version": %s,\n' "$(json_quote "$H5PY_VERSION")"
        printf '  "opencv_version": %s,\n' "$(json_quote "$OPENCV_VERSION")"
        printf '  "upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s, "grasp_reference": %s},\n' \
            "$(json_quote "$CANDIDATE_SHA")" "$(json_quote "$UNITREE_SHA")" "$(json_quote "$GRASP_REFERENCE_COMMIT")"
        printf '  "upstream_dirty": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$CANDIDATE_DIRTY_JSON" "$UNITREE_DIRTY_JSON"
        printf '  "expected_upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$(json_quote "$UPSTREAM_COMMIT")" "$(json_quote "$UNITREE_COMMIT")"
        printf '  "mesh_provenance": {"path_class": %s, "path_relative_to_unitree": %s},\n' \
            "$(json_quote "$MESH_PATH_CLASS")" "$(json_quote "$MESH_RELATIVE_PATH")"
        printf '  "seed": %s,\n' "$SEED_JSON"
        printf '  "acceptance_thresholds": {"target_table_margin_m": 0.03, "bilateral_grasp_force_n_per_palm": 2.0, "bilateral_grasp_frames": 5, "minimum_lift_height_m": 0.05, "contact_free_release_frames": 5, "stable_duration_s": 1.0, "linear_speed_m_s": 0.03, "angular_speed_rad_s": 0.20, "stable_position_radius_m": 0.02, "control_step_translation_m": 0.20, "physics_step_translation_m": 0.005, "physics_step_angular_jump_rad": 0.025, "weld_event_translation_snap_m": 0.002, "weld_event_angular_snap_rad": 0.017453292519943295, "maximum_penetration_m": 0.025, "drop_height_m": 0.50},\n'
        printf '  "output_directory": %s\n' "$(json_quote "$OUTPUT_DIR")"
        printf '}\n'
    } > "$RESULT_PATH"
}

CURRENT_STAGE="INITIALIZE"
ROBOTSIM_SHA="not_checked"
ROBOTSIM_DIRTY_JSON=null
PYTHON_VERSION="not_run"
MUJOCO_VERSION="not_run"
NUMPY_VERSION="not_run"
H5PY_VERSION="not_run"
OPENCV_VERSION="not_run"
CANDIDATE_SHA="not_checked"
UNITREE_SHA="not_checked"
CANDIDATE_DIRTY_JSON=null
UNITREE_DIRTY_JSON=null
MESH_PATH_CLASS="not_checked"
MESH_RELATIVE_PATH="not_checked"

RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"

bootstrap_failure() {
    local message="$1"
    OUTPUT_DIR="/tmp/robotsim-issue43-m0/bootstrap-failures/$RUN_ID"
    if ! mkdir -p "$OUTPUT_DIR"; then
        printf '%s; could not persist a failure record under the fallback directory\n' "$message" >&2
        return 1
    fi
    RESULT_PATH="$OUTPUT_DIR/m0_result.json"
    write_result "PREFLIGHT" ""
    CURRENT_STAGE="OUTPUT_DIRECTORY"
    write_result "FAILED" "$message"
    printf '%s; failure result: %s\n' "$message" "$RESULT_PATH" >&2
}

if ! mkdir -p "$OUTPUT_ROOT" 2>/dev/null; then
    bootstrap_failure "Could not create configured M0 output root"
    exit 2
fi
while [[ -e "$OUTPUT_DIR" ]]; do
    RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
    OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
done
if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
    bootstrap_failure "Could not create unique M0 run directory"
    exit 2
fi
RESULT_PATH="$OUTPUT_DIR/m0_result.json"
write_result "PREFLIGHT" ""

on_exit() {
    local status=$?
    if (( status != 0 )); then
        local result_run_id=""
        local result_state=""
        if [[ -f "$RESULT_PATH" ]]; then
            result_run_id="$(sed -n 's/^[[:space:]]*"run_id":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
            result_state="$(sed -n 's/^[[:space:]]*"state":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
        fi
        if [[ "$result_run_id" != "$RUN_ID" || "$result_state" != "FAILED" ]]; then
            write_result "FAILED" "Preflight or launcher failed at ${CURRENT_STAGE} (exit ${status})"
        fi
        printf 'M0 invocation %s failed at %s; failure result: %s\n' \
            "$RUN_ID" "$CURRENT_STAGE" "$RESULT_PATH" >&2
    fi
}
trap on_exit EXIT

CURRENT_STAGE="TOOL_CHECK"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    printf 'ROBOTSIM_M0_SEED must be a non-negative integer\n' >&2
    exit 2
fi
command -v git >/dev/null || { printf 'git is required\n' >&2; exit 2; }
command -v uv >/dev/null || { printf 'uv is required to create the pinned Python 3.10 environment\n' >&2; exit 2; }
command -v realpath >/dev/null || { printf 'realpath is required for mesh provenance validation\n' >&2; exit 2; }

CURRENT_STAGE="ROBOTSIM_IDENTITY"
ROBOTSIM_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD)"
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=all)" ]]; then
    ROBOTSIM_DIRTY_JSON=true
else
    ROBOTSIM_DIRTY_JSON=false
fi
write_result "PREFLIGHT" ""

ensure_checkout() {
    local label="$1"
    local url="$2"
    local revision="$3"
    local path="$4"
    local sha_var="$5"
    local dirty_var="${sha_var%_SHA}_DIRTY_JSON"

    if [[ -e "$path" && ! -d "$path/.git" ]]; then
        printf 'Refusing to use non-Git %s path\n' "$label" >&2
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
    printf -v "$sha_var" '%s' "$actual"
    write_result "PREFLIGHT" ""
    if [[ "$actual" != "$revision" ]]; then
        printf '%s checkout has the wrong pinned revision\n' "$label" >&2
        return 1
    fi

    local status_output
    status_output="$(git -C "$path" status --porcelain --untracked-files=all)"
    if [[ -n "$status_output" ]]; then
        printf -v "$dirty_var" '%s' true
        write_result "PREFLIGHT" ""
        printf 'Refusing dirty %s checkout\n' "$label" >&2
        return 1
    fi
    printf -v "$dirty_var" '%s' false
    write_result "PREFLIGHT" ""
}

CURRENT_STAGE="HUMANOID_VLA_CHECKOUT"
ensure_checkout \
    "Humanoid VLA" \
    "https://github.com/ozkannceylan/humanoid_vla.git" \
    "$UPSTREAM_COMMIT" \
    "$CANDIDATE_DIR" \
    CANDIDATE_SHA

CURRENT_STAGE="UNITREE_CHECKOUT"
ensure_checkout \
    "Unitree MuJoCo meshes" \
    "https://github.com/unitreerobotics/unitree_mujoco.git" \
    "$UNITREE_COMMIT" \
    "$UNITREE_DIR" \
    UNITREE_SHA

CURRENT_STAGE="MESH_PROVENANCE"
UNITREE_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR")"
G1_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR_RESOLVED/unitree_robots/g1")"
case "$G1_DIR_RESOLVED" in
    "$UNITREE_DIR_RESOLVED"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="g1_model_directory_resolves_outside_unitree_checkout"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 asset directory escapes the pinned Unitree checkout\n' >&2
        exit 2
        ;;
esac
VERIFIED_MESH_ROOT="$(realpath -e "$G1_DIR_RESOLVED/meshes")"
case "$VERIFIED_MESH_ROOT" in
    "$G1_DIR_RESOLVED/meshes"|"$G1_DIR_RESOLVED/meshes"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="resolved_g1_mesh_root_escapes_verified_tree"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 mesh root escapes the verified G1 asset directory\n' >&2
        exit 2
        ;;
esac
MESH_REQUESTED="${ROBOTSIM_M0_MESH_DIR:-$VERIFIED_MESH_ROOT}"
if ! MESH_DIR="$(realpath -e "$MESH_REQUESTED")" || [[ ! -d "$MESH_DIR" ]]; then
    MESH_PATH_CLASS="missing_or_unresolvable"
    MESH_RELATIVE_PATH="not_available"
    write_result "PREFLIGHT" ""
    printf 'G1 mesh directory is missing or cannot be resolved\n' >&2
    exit 2
fi
case "$MESH_DIR" in
    "$VERIFIED_MESH_ROOT"|"$VERIFIED_MESH_ROOT"/*)
        MESH_PATH_CLASS="pinned_unitree_g1_mesh_tree"
        MESH_RELATIVE_PATH="$(realpath --relative-to="$UNITREE_DIR_RESOLVED" "$MESH_DIR")"
        ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="outside_verified_unitree_g1_mesh_tree"
        write_result "PREFLIGHT" ""
        printf 'ROBOTSIM_M0_MESH_DIR must resolve inside the pinned Unitree G1 mesh tree\n' >&2
        exit 2
        ;;
esac
write_result "PREFLIGHT" ""

CURRENT_STAGE="PYTHON_ENVIRONMENT"
VENV_DIR="$RUN_ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
if [[ -e "$VENV_PYTHON" ]]; then
    if [[ ! -x "$VENV_PYTHON" ]]; then
        printf 'Existing M0 venv Python is not executable\n' >&2
        exit 2
    fi
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'Existing M0 venv must use Python 3.10.x; found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
else
    uv venv --python 3.10 "$VENV_DIR"
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'uv created an incompatible Python venv; expected 3.10.x, found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
fi

CURRENT_STAGE="DEPENDENCY_INSTALL"
uv pip install --python "$VENV_PYTHON" -r "$ROOT_DIR/simulation/mujoco/requirements-m0.txt"
read -r MUJOCO_VERSION NUMPY_VERSION H5PY_VERSION OPENCV_VERSION < <(
    "$VENV_PYTHON" -c 'import cv2,h5py,mujoco,numpy; print(mujoco.__version__, numpy.__version__, h5py.__version__, cv2.__version__)'
)
write_result "PREFLIGHT" ""

LOG_PATH="$OUTPUT_DIR/run.log"
VIDEO_PATH="$OUTPUT_DIR/m0_pick_place.mp4"
SCREENSHOT_PATH="$OUTPUT_DIR/m0_final.png"
CURRENT_STAGE="RUNTIME"
cd "$ROOT_DIR"
set +e
MUJOCO_GL=egl PYTHONPATH="$CANDIDATE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV_PYTHON" -m simulation.mujoco.m0_pick_place \
        --candidate-root "$CANDIDATE_DIR" \
        --unitree-root "$UNITREE_DIR" \
        --mesh-dir "$MESH_DIR" \
        --output-dir "$OUTPUT_DIR/model" \
        --output-json "$RESULT_PATH" \
        --video "$VIDEO_PATH" \
        --screenshot "$SCREENSHOT_PATH" \
        --run-id "$RUN_ID" \
        --seed "$SEED" \
    2>&1 | tee "$LOG_PATH"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if (( status == 0 && pipeline_status[1] != 0 )); then
    status=${pipeline_status[1]}
fi
set -e
exit "$status"
\n'/\\n}
    value=${value//
write_result() {
    local state="$1"
    local error="$2"
    {
        printf '{\n'
        printf '  "issue": 43,\n'
        printf '  "demo": "M0 G1 bottle pick-and-place",\n'
        printf '  "run_id": %s,\n' "$(json_quote "$RUN_ID")"
        printf '  "state": %s,\n' "$(json_quote "$state")"
        printf '  "stage": %s,\n' "$(json_quote "$CURRENT_STAGE")"
        printf '  "passed": false,\n'
        printf '  "complete": false,\n'
        printf '  "error": %s,\n' "$(json_quote "$error")"
        printf '  "robotsim_sha": %s,\n' "$(json_quote "$ROBOTSIM_SHA")"
        printf '  "robotsim_dirty": %s,\n' "$ROBOTSIM_DIRTY_JSON"
        printf '  "python_version": %s,\n' "$(json_quote "$PYTHON_VERSION")"
        printf '  "mujoco_version": %s,\n' "$(json_quote "$MUJOCO_VERSION")"
        printf '  "numpy_version": %s,\n' "$(json_quote "$NUMPY_VERSION")"
        printf '  "h5py_version": %s,\n' "$(json_quote "$H5PY_VERSION")"
        printf '  "opencv_version": %s,\n' "$(json_quote "$OPENCV_VERSION")"
        printf '  "upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s, "grasp_reference": %s},\n' \
            "$(json_quote "$CANDIDATE_SHA")" "$(json_quote "$UNITREE_SHA")" "$(json_quote "$GRASP_REFERENCE_COMMIT")"
        printf '  "upstream_dirty": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$CANDIDATE_DIRTY_JSON" "$UNITREE_DIRTY_JSON"
        printf '  "expected_upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$(json_quote "$UPSTREAM_COMMIT")" "$(json_quote "$UNITREE_COMMIT")"
        printf '  "mesh_provenance": {"path_class": %s, "path_relative_to_unitree": %s},\n' \
            "$(json_quote "$MESH_PATH_CLASS")" "$(json_quote "$MESH_RELATIVE_PATH")"
        printf '  "seed": %s,\n' "$SEED_JSON"
        printf '  "acceptance_thresholds": {"target_table_margin_m": 0.03, "bilateral_grasp_force_n_per_palm": 2.0, "bilateral_grasp_frames": 5, "minimum_lift_height_m": 0.05, "contact_free_release_frames": 5, "stable_duration_s": 1.0, "linear_speed_m_s": 0.03, "angular_speed_rad_s": 0.20, "stable_position_radius_m": 0.02, "control_step_translation_m": 0.20, "physics_step_translation_m": 0.005, "physics_step_angular_jump_rad": 0.025, "weld_event_translation_snap_m": 0.002, "weld_event_angular_snap_rad": 0.017453292519943295, "maximum_penetration_m": 0.025, "drop_height_m": 0.50},\n'
        printf '  "output_directory": %s\n' "$(json_quote "$OUTPUT_DIR")"
        printf '}\n'
    } > "$RESULT_PATH"
}

CURRENT_STAGE="INITIALIZE"
ROBOTSIM_SHA="not_checked"
ROBOTSIM_DIRTY_JSON=null
PYTHON_VERSION="not_run"
MUJOCO_VERSION="not_run"
NUMPY_VERSION="not_run"
H5PY_VERSION="not_run"
OPENCV_VERSION="not_run"
CANDIDATE_SHA="not_checked"
UNITREE_SHA="not_checked"
CANDIDATE_DIRTY_JSON=null
UNITREE_DIRTY_JSON=null
MESH_PATH_CLASS="not_checked"
MESH_RELATIVE_PATH="not_checked"

RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"

bootstrap_failure() {
    local message="$1"
    OUTPUT_DIR="/tmp/robotsim-issue43-m0/bootstrap-failures/$RUN_ID"
    if ! mkdir -p "$OUTPUT_DIR"; then
        printf '%s; could not persist a failure record under the fallback directory\n' "$message" >&2
        return 1
    fi
    RESULT_PATH="$OUTPUT_DIR/m0_result.json"
    write_result "PREFLIGHT" ""
    CURRENT_STAGE="OUTPUT_DIRECTORY"
    write_result "FAILED" "$message"
    printf '%s; failure result: %s\n' "$message" "$RESULT_PATH" >&2
}

if ! mkdir -p "$OUTPUT_ROOT" 2>/dev/null; then
    bootstrap_failure "Could not create configured M0 output root"
    exit 2
fi
while [[ -e "$OUTPUT_DIR" ]]; do
    RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
    OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
done
if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
    bootstrap_failure "Could not create unique M0 run directory"
    exit 2
fi
RESULT_PATH="$OUTPUT_DIR/m0_result.json"
write_result "PREFLIGHT" ""

on_exit() {
    local status=$?
    if (( status != 0 )); then
        local result_run_id=""
        local result_state=""
        if [[ -f "$RESULT_PATH" ]]; then
            result_run_id="$(sed -n 's/^[[:space:]]*"run_id":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
            result_state="$(sed -n 's/^[[:space:]]*"state":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
        fi
        if [[ "$result_run_id" != "$RUN_ID" || "$result_state" != "FAILED" ]]; then
            write_result "FAILED" "Preflight or launcher failed at ${CURRENT_STAGE} (exit ${status})"
        fi
        printf 'M0 invocation %s failed at %s; failure result: %s\n' \
            "$RUN_ID" "$CURRENT_STAGE" "$RESULT_PATH" >&2
    fi
}
trap on_exit EXIT

CURRENT_STAGE="TOOL_CHECK"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    printf 'ROBOTSIM_M0_SEED must be a non-negative integer\n' >&2
    exit 2
fi
command -v git >/dev/null || { printf 'git is required\n' >&2; exit 2; }
command -v uv >/dev/null || { printf 'uv is required to create the pinned Python 3.10 environment\n' >&2; exit 2; }
command -v realpath >/dev/null || { printf 'realpath is required for mesh provenance validation\n' >&2; exit 2; }

CURRENT_STAGE="ROBOTSIM_IDENTITY"
ROBOTSIM_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD)"
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=all)" ]]; then
    ROBOTSIM_DIRTY_JSON=true
else
    ROBOTSIM_DIRTY_JSON=false
fi
write_result "PREFLIGHT" ""

ensure_checkout() {
    local label="$1"
    local url="$2"
    local revision="$3"
    local path="$4"
    local sha_var="$5"
    local dirty_var="${sha_var%_SHA}_DIRTY_JSON"

    if [[ -e "$path" && ! -d "$path/.git" ]]; then
        printf 'Refusing to use non-Git %s path\n' "$label" >&2
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
    printf -v "$sha_var" '%s' "$actual"
    write_result "PREFLIGHT" ""
    if [[ "$actual" != "$revision" ]]; then
        printf '%s checkout has the wrong pinned revision\n' "$label" >&2
        return 1
    fi

    local status_output
    status_output="$(git -C "$path" status --porcelain --untracked-files=all)"
    if [[ -n "$status_output" ]]; then
        printf -v "$dirty_var" '%s' true
        write_result "PREFLIGHT" ""
        printf 'Refusing dirty %s checkout\n' "$label" >&2
        return 1
    fi
    printf -v "$dirty_var" '%s' false
    write_result "PREFLIGHT" ""
}

CURRENT_STAGE="HUMANOID_VLA_CHECKOUT"
ensure_checkout \
    "Humanoid VLA" \
    "https://github.com/ozkannceylan/humanoid_vla.git" \
    "$UPSTREAM_COMMIT" \
    "$CANDIDATE_DIR" \
    CANDIDATE_SHA

CURRENT_STAGE="UNITREE_CHECKOUT"
ensure_checkout \
    "Unitree MuJoCo meshes" \
    "https://github.com/unitreerobotics/unitree_mujoco.git" \
    "$UNITREE_COMMIT" \
    "$UNITREE_DIR" \
    UNITREE_SHA

CURRENT_STAGE="MESH_PROVENANCE"
UNITREE_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR")"
G1_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR_RESOLVED/unitree_robots/g1")"
case "$G1_DIR_RESOLVED" in
    "$UNITREE_DIR_RESOLVED"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="g1_model_directory_resolves_outside_unitree_checkout"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 asset directory escapes the pinned Unitree checkout\n' >&2
        exit 2
        ;;
esac
VERIFIED_MESH_ROOT="$(realpath -e "$G1_DIR_RESOLVED/meshes")"
case "$VERIFIED_MESH_ROOT" in
    "$G1_DIR_RESOLVED/meshes"|"$G1_DIR_RESOLVED/meshes"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="resolved_g1_mesh_root_escapes_verified_tree"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 mesh root escapes the verified G1 asset directory\n' >&2
        exit 2
        ;;
esac
MESH_REQUESTED="${ROBOTSIM_M0_MESH_DIR:-$VERIFIED_MESH_ROOT}"
if ! MESH_DIR="$(realpath -e "$MESH_REQUESTED")" || [[ ! -d "$MESH_DIR" ]]; then
    MESH_PATH_CLASS="missing_or_unresolvable"
    MESH_RELATIVE_PATH="not_available"
    write_result "PREFLIGHT" ""
    printf 'G1 mesh directory is missing or cannot be resolved\n' >&2
    exit 2
fi
case "$MESH_DIR" in
    "$VERIFIED_MESH_ROOT"|"$VERIFIED_MESH_ROOT"/*)
        MESH_PATH_CLASS="pinned_unitree_g1_mesh_tree"
        MESH_RELATIVE_PATH="$(realpath --relative-to="$UNITREE_DIR_RESOLVED" "$MESH_DIR")"
        ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="outside_verified_unitree_g1_mesh_tree"
        write_result "PREFLIGHT" ""
        printf 'ROBOTSIM_M0_MESH_DIR must resolve inside the pinned Unitree G1 mesh tree\n' >&2
        exit 2
        ;;
esac
write_result "PREFLIGHT" ""

CURRENT_STAGE="PYTHON_ENVIRONMENT"
VENV_DIR="$RUN_ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
if [[ -e "$VENV_PYTHON" ]]; then
    if [[ ! -x "$VENV_PYTHON" ]]; then
        printf 'Existing M0 venv Python is not executable\n' >&2
        exit 2
    fi
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'Existing M0 venv must use Python 3.10.x; found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
else
    uv venv --python 3.10 "$VENV_DIR"
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'uv created an incompatible Python venv; expected 3.10.x, found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
fi

CURRENT_STAGE="DEPENDENCY_INSTALL"
uv pip install --python "$VENV_PYTHON" -r "$ROOT_DIR/simulation/mujoco/requirements-m0.txt"
read -r MUJOCO_VERSION NUMPY_VERSION H5PY_VERSION OPENCV_VERSION < <(
    "$VENV_PYTHON" -c 'import cv2,h5py,mujoco,numpy; print(mujoco.__version__, numpy.__version__, h5py.__version__, cv2.__version__)'
)
write_result "PREFLIGHT" ""

LOG_PATH="$OUTPUT_DIR/run.log"
VIDEO_PATH="$OUTPUT_DIR/m0_pick_place.mp4"
SCREENSHOT_PATH="$OUTPUT_DIR/m0_final.png"
CURRENT_STAGE="RUNTIME"
cd "$ROOT_DIR"
set +e
MUJOCO_GL=egl PYTHONPATH="$CANDIDATE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV_PYTHON" -m simulation.mujoco.m0_pick_place \
        --candidate-root "$CANDIDATE_DIR" \
        --unitree-root "$UNITREE_DIR" \
        --mesh-dir "$MESH_DIR" \
        --output-dir "$OUTPUT_DIR/model" \
        --output-json "$RESULT_PATH" \
        --video "$VIDEO_PATH" \
        --screenshot "$SCREENSHOT_PATH" \
        --run-id "$RUN_ID" \
        --seed "$SEED" \
    2>&1 | tee "$LOG_PATH"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if (( status == 0 && pipeline_status[1] != 0 )); then
    status=${pipeline_status[1]}
fi
set -e
exit "$status"
\r'/\\r}
    value=${value//
write_result() {
    local state="$1"
    local error="$2"
    {
        printf '{\n'
        printf '  "issue": 43,\n'
        printf '  "demo": "M0 G1 bottle pick-and-place",\n'
        printf '  "run_id": %s,\n' "$(json_quote "$RUN_ID")"
        printf '  "state": %s,\n' "$(json_quote "$state")"
        printf '  "stage": %s,\n' "$(json_quote "$CURRENT_STAGE")"
        printf '  "passed": false,\n'
        printf '  "complete": false,\n'
        printf '  "error": %s,\n' "$(json_quote "$error")"
        printf '  "robotsim_sha": %s,\n' "$(json_quote "$ROBOTSIM_SHA")"
        printf '  "robotsim_dirty": %s,\n' "$ROBOTSIM_DIRTY_JSON"
        printf '  "python_version": %s,\n' "$(json_quote "$PYTHON_VERSION")"
        printf '  "mujoco_version": %s,\n' "$(json_quote "$MUJOCO_VERSION")"
        printf '  "numpy_version": %s,\n' "$(json_quote "$NUMPY_VERSION")"
        printf '  "h5py_version": %s,\n' "$(json_quote "$H5PY_VERSION")"
        printf '  "opencv_version": %s,\n' "$(json_quote "$OPENCV_VERSION")"
        printf '  "upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s, "grasp_reference": %s},\n' \
            "$(json_quote "$CANDIDATE_SHA")" "$(json_quote "$UNITREE_SHA")" "$(json_quote "$GRASP_REFERENCE_COMMIT")"
        printf '  "upstream_dirty": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$CANDIDATE_DIRTY_JSON" "$UNITREE_DIRTY_JSON"
        printf '  "expected_upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$(json_quote "$UPSTREAM_COMMIT")" "$(json_quote "$UNITREE_COMMIT")"
        printf '  "mesh_provenance": {"path_class": %s, "path_relative_to_unitree": %s},\n' \
            "$(json_quote "$MESH_PATH_CLASS")" "$(json_quote "$MESH_RELATIVE_PATH")"
        printf '  "seed": %s,\n' "$SEED_JSON"
        printf '  "acceptance_thresholds": {"target_table_margin_m": 0.03, "bilateral_grasp_force_n_per_palm": 2.0, "bilateral_grasp_frames": 5, "minimum_lift_height_m": 0.05, "contact_free_release_frames": 5, "stable_duration_s": 1.0, "linear_speed_m_s": 0.03, "angular_speed_rad_s": 0.20, "stable_position_radius_m": 0.02, "control_step_translation_m": 0.20, "physics_step_translation_m": 0.005, "physics_step_angular_jump_rad": 0.025, "weld_event_translation_snap_m": 0.002, "weld_event_angular_snap_rad": 0.017453292519943295, "maximum_penetration_m": 0.025, "drop_height_m": 0.50},\n'
        printf '  "output_directory": %s\n' "$(json_quote "$OUTPUT_DIR")"
        printf '}\n'
    } > "$RESULT_PATH"
}

CURRENT_STAGE="INITIALIZE"
ROBOTSIM_SHA="not_checked"
ROBOTSIM_DIRTY_JSON=null
PYTHON_VERSION="not_run"
MUJOCO_VERSION="not_run"
NUMPY_VERSION="not_run"
H5PY_VERSION="not_run"
OPENCV_VERSION="not_run"
CANDIDATE_SHA="not_checked"
UNITREE_SHA="not_checked"
CANDIDATE_DIRTY_JSON=null
UNITREE_DIRTY_JSON=null
MESH_PATH_CLASS="not_checked"
MESH_RELATIVE_PATH="not_checked"

RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"

bootstrap_failure() {
    local message="$1"
    OUTPUT_DIR="/tmp/robotsim-issue43-m0/bootstrap-failures/$RUN_ID"
    if ! mkdir -p "$OUTPUT_DIR"; then
        printf '%s; could not persist a failure record under the fallback directory\n' "$message" >&2
        return 1
    fi
    RESULT_PATH="$OUTPUT_DIR/m0_result.json"
    write_result "PREFLIGHT" ""
    CURRENT_STAGE="OUTPUT_DIRECTORY"
    write_result "FAILED" "$message"
    printf '%s; failure result: %s\n' "$message" "$RESULT_PATH" >&2
}

if ! mkdir -p "$OUTPUT_ROOT" 2>/dev/null; then
    bootstrap_failure "Could not create configured M0 output root"
    exit 2
fi
while [[ -e "$OUTPUT_DIR" ]]; do
    RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
    OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
done
if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
    bootstrap_failure "Could not create unique M0 run directory"
    exit 2
fi
RESULT_PATH="$OUTPUT_DIR/m0_result.json"
write_result "PREFLIGHT" ""

on_exit() {
    local status=$?
    if (( status != 0 )); then
        local result_run_id=""
        local result_state=""
        if [[ -f "$RESULT_PATH" ]]; then
            result_run_id="$(sed -n 's/^[[:space:]]*"run_id":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
            result_state="$(sed -n 's/^[[:space:]]*"state":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
        fi
        if [[ "$result_run_id" != "$RUN_ID" || "$result_state" != "FAILED" ]]; then
            write_result "FAILED" "Preflight or launcher failed at ${CURRENT_STAGE} (exit ${status})"
        fi
        printf 'M0 invocation %s failed at %s; failure result: %s\n' \
            "$RUN_ID" "$CURRENT_STAGE" "$RESULT_PATH" >&2
    fi
}
trap on_exit EXIT

CURRENT_STAGE="TOOL_CHECK"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    printf 'ROBOTSIM_M0_SEED must be a non-negative integer\n' >&2
    exit 2
fi
command -v git >/dev/null || { printf 'git is required\n' >&2; exit 2; }
command -v uv >/dev/null || { printf 'uv is required to create the pinned Python 3.10 environment\n' >&2; exit 2; }
command -v realpath >/dev/null || { printf 'realpath is required for mesh provenance validation\n' >&2; exit 2; }

CURRENT_STAGE="ROBOTSIM_IDENTITY"
ROBOTSIM_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD)"
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=all)" ]]; then
    ROBOTSIM_DIRTY_JSON=true
else
    ROBOTSIM_DIRTY_JSON=false
fi
write_result "PREFLIGHT" ""

ensure_checkout() {
    local label="$1"
    local url="$2"
    local revision="$3"
    local path="$4"
    local sha_var="$5"
    local dirty_var="${sha_var%_SHA}_DIRTY_JSON"

    if [[ -e "$path" && ! -d "$path/.git" ]]; then
        printf 'Refusing to use non-Git %s path\n' "$label" >&2
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
    printf -v "$sha_var" '%s' "$actual"
    write_result "PREFLIGHT" ""
    if [[ "$actual" != "$revision" ]]; then
        printf '%s checkout has the wrong pinned revision\n' "$label" >&2
        return 1
    fi

    local status_output
    status_output="$(git -C "$path" status --porcelain --untracked-files=all)"
    if [[ -n "$status_output" ]]; then
        printf -v "$dirty_var" '%s' true
        write_result "PREFLIGHT" ""
        printf 'Refusing dirty %s checkout\n' "$label" >&2
        return 1
    fi
    printf -v "$dirty_var" '%s' false
    write_result "PREFLIGHT" ""
}

CURRENT_STAGE="HUMANOID_VLA_CHECKOUT"
ensure_checkout \
    "Humanoid VLA" \
    "https://github.com/ozkannceylan/humanoid_vla.git" \
    "$UPSTREAM_COMMIT" \
    "$CANDIDATE_DIR" \
    CANDIDATE_SHA

CURRENT_STAGE="UNITREE_CHECKOUT"
ensure_checkout \
    "Unitree MuJoCo meshes" \
    "https://github.com/unitreerobotics/unitree_mujoco.git" \
    "$UNITREE_COMMIT" \
    "$UNITREE_DIR" \
    UNITREE_SHA

CURRENT_STAGE="MESH_PROVENANCE"
UNITREE_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR")"
G1_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR_RESOLVED/unitree_robots/g1")"
case "$G1_DIR_RESOLVED" in
    "$UNITREE_DIR_RESOLVED"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="g1_model_directory_resolves_outside_unitree_checkout"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 asset directory escapes the pinned Unitree checkout\n' >&2
        exit 2
        ;;
esac
VERIFIED_MESH_ROOT="$(realpath -e "$G1_DIR_RESOLVED/meshes")"
case "$VERIFIED_MESH_ROOT" in
    "$G1_DIR_RESOLVED/meshes"|"$G1_DIR_RESOLVED/meshes"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="resolved_g1_mesh_root_escapes_verified_tree"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 mesh root escapes the verified G1 asset directory\n' >&2
        exit 2
        ;;
esac
MESH_REQUESTED="${ROBOTSIM_M0_MESH_DIR:-$VERIFIED_MESH_ROOT}"
if ! MESH_DIR="$(realpath -e "$MESH_REQUESTED")" || [[ ! -d "$MESH_DIR" ]]; then
    MESH_PATH_CLASS="missing_or_unresolvable"
    MESH_RELATIVE_PATH="not_available"
    write_result "PREFLIGHT" ""
    printf 'G1 mesh directory is missing or cannot be resolved\n' >&2
    exit 2
fi
case "$MESH_DIR" in
    "$VERIFIED_MESH_ROOT"|"$VERIFIED_MESH_ROOT"/*)
        MESH_PATH_CLASS="pinned_unitree_g1_mesh_tree"
        MESH_RELATIVE_PATH="$(realpath --relative-to="$UNITREE_DIR_RESOLVED" "$MESH_DIR")"
        ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="outside_verified_unitree_g1_mesh_tree"
        write_result "PREFLIGHT" ""
        printf 'ROBOTSIM_M0_MESH_DIR must resolve inside the pinned Unitree G1 mesh tree\n' >&2
        exit 2
        ;;
esac
write_result "PREFLIGHT" ""

CURRENT_STAGE="PYTHON_ENVIRONMENT"
VENV_DIR="$RUN_ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
if [[ -e "$VENV_PYTHON" ]]; then
    if [[ ! -x "$VENV_PYTHON" ]]; then
        printf 'Existing M0 venv Python is not executable\n' >&2
        exit 2
    fi
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'Existing M0 venv must use Python 3.10.x; found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
else
    uv venv --python 3.10 "$VENV_DIR"
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'uv created an incompatible Python venv; expected 3.10.x, found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
fi

CURRENT_STAGE="DEPENDENCY_INSTALL"
uv pip install --python "$VENV_PYTHON" -r "$ROOT_DIR/simulation/mujoco/requirements-m0.txt"
read -r MUJOCO_VERSION NUMPY_VERSION H5PY_VERSION OPENCV_VERSION < <(
    "$VENV_PYTHON" -c 'import cv2,h5py,mujoco,numpy; print(mujoco.__version__, numpy.__version__, h5py.__version__, cv2.__version__)'
)
write_result "PREFLIGHT" ""

LOG_PATH="$OUTPUT_DIR/run.log"
VIDEO_PATH="$OUTPUT_DIR/m0_pick_place.mp4"
SCREENSHOT_PATH="$OUTPUT_DIR/m0_final.png"
CURRENT_STAGE="RUNTIME"
cd "$ROOT_DIR"
set +e
MUJOCO_GL=egl PYTHONPATH="$CANDIDATE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV_PYTHON" -m simulation.mujoco.m0_pick_place \
        --candidate-root "$CANDIDATE_DIR" \
        --unitree-root "$UNITREE_DIR" \
        --mesh-dir "$MESH_DIR" \
        --output-dir "$OUTPUT_DIR/model" \
        --output-json "$RESULT_PATH" \
        --video "$VIDEO_PATH" \
        --screenshot "$SCREENSHOT_PATH" \
        --run-id "$RUN_ID" \
        --seed "$SEED" \
    2>&1 | tee "$LOG_PATH"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if (( status == 0 && pipeline_status[1] != 0 )); then
    status=${pipeline_status[1]}
fi
set -e
exit "$status"
\t'/\\t}

    # Bash variables and POSIX paths cannot contain NUL. Escape every other
    # JSON control byte that may legally occur in an environment-provided path.
    for code in 1 2 3 4 5 6 7 11 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31; do
        printf -v octal '%03o' "$code"
        printf -v control '%b' "\\$octal"
        printf -v escaped '\\u%04x' "$code"
        value=${value//"$control"/$escaped}
    done
    printf '"%s"' "$value"
}

write_result() {
    local state="$1"
    local error="$2"
    {
        printf '{\n'
        printf '  "issue": 43,\n'
        printf '  "demo": "M0 G1 bottle pick-and-place",\n'
        printf '  "run_id": %s,\n' "$(json_quote "$RUN_ID")"
        printf '  "state": %s,\n' "$(json_quote "$state")"
        printf '  "stage": %s,\n' "$(json_quote "$CURRENT_STAGE")"
        printf '  "passed": false,\n'
        printf '  "complete": false,\n'
        printf '  "error": %s,\n' "$(json_quote "$error")"
        printf '  "robotsim_sha": %s,\n' "$(json_quote "$ROBOTSIM_SHA")"
        printf '  "robotsim_dirty": %s,\n' "$ROBOTSIM_DIRTY_JSON"
        printf '  "python_version": %s,\n' "$(json_quote "$PYTHON_VERSION")"
        printf '  "mujoco_version": %s,\n' "$(json_quote "$MUJOCO_VERSION")"
        printf '  "numpy_version": %s,\n' "$(json_quote "$NUMPY_VERSION")"
        printf '  "h5py_version": %s,\n' "$(json_quote "$H5PY_VERSION")"
        printf '  "opencv_version": %s,\n' "$(json_quote "$OPENCV_VERSION")"
        printf '  "upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s, "grasp_reference": %s},\n' \
            "$(json_quote "$CANDIDATE_SHA")" "$(json_quote "$UNITREE_SHA")" "$(json_quote "$GRASP_REFERENCE_COMMIT")"
        printf '  "upstream_dirty": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$CANDIDATE_DIRTY_JSON" "$UNITREE_DIRTY_JSON"
        printf '  "expected_upstream_shas": {"humanoid_vla": %s, "unitree_mujoco": %s},\n' \
            "$(json_quote "$UPSTREAM_COMMIT")" "$(json_quote "$UNITREE_COMMIT")"
        printf '  "mesh_provenance": {"path_class": %s, "path_relative_to_unitree": %s},\n' \
            "$(json_quote "$MESH_PATH_CLASS")" "$(json_quote "$MESH_RELATIVE_PATH")"
        printf '  "seed": %s,\n' "$SEED_JSON"
        printf '  "acceptance_thresholds": {"target_table_margin_m": 0.03, "bilateral_grasp_force_n_per_palm": 2.0, "bilateral_grasp_frames": 5, "minimum_lift_height_m": 0.05, "contact_free_release_frames": 5, "stable_duration_s": 1.0, "linear_speed_m_s": 0.03, "angular_speed_rad_s": 0.20, "stable_position_radius_m": 0.02, "control_step_translation_m": 0.20, "physics_step_translation_m": 0.005, "physics_step_angular_jump_rad": 0.025, "weld_event_translation_snap_m": 0.002, "weld_event_angular_snap_rad": 0.017453292519943295, "maximum_penetration_m": 0.025, "drop_height_m": 0.50},\n'
        printf '  "output_directory": %s\n' "$(json_quote "$OUTPUT_DIR")"
        printf '}\n'
    } > "$RESULT_PATH"
}

CURRENT_STAGE="INITIALIZE"
ROBOTSIM_SHA="not_checked"
ROBOTSIM_DIRTY_JSON=null
PYTHON_VERSION="not_run"
MUJOCO_VERSION="not_run"
NUMPY_VERSION="not_run"
H5PY_VERSION="not_run"
OPENCV_VERSION="not_run"
CANDIDATE_SHA="not_checked"
UNITREE_SHA="not_checked"
CANDIDATE_DIRTY_JSON=null
UNITREE_DIRTY_JSON=null
MESH_PATH_CLASS="not_checked"
MESH_RELATIVE_PATH="not_checked"

RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"

bootstrap_failure() {
    local message="$1"
    OUTPUT_DIR="/tmp/robotsim-issue43-m0/bootstrap-failures/$RUN_ID"
    if ! mkdir -p "$OUTPUT_DIR"; then
        printf '%s; could not persist a failure record under the fallback directory\n' "$message" >&2
        return 1
    fi
    RESULT_PATH="$OUTPUT_DIR/m0_result.json"
    write_result "PREFLIGHT" ""
    CURRENT_STAGE="OUTPUT_DIRECTORY"
    write_result "FAILED" "$message"
    printf '%s; failure result: %s\n' "$message" "$RESULT_PATH" >&2
}

if ! mkdir -p "$OUTPUT_ROOT" 2>/dev/null; then
    bootstrap_failure "Could not create configured M0 output root"
    exit 2
fi
while [[ -e "$OUTPUT_DIR" ]]; do
    RUN_ID="m0-${EPOCHSECONDS:-0}-$$-${RANDOM}"
    OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
done
if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
    bootstrap_failure "Could not create unique M0 run directory"
    exit 2
fi
RESULT_PATH="$OUTPUT_DIR/m0_result.json"
write_result "PREFLIGHT" ""

on_exit() {
    local status=$?
    if (( status != 0 )); then
        local result_run_id=""
        local result_state=""
        if [[ -f "$RESULT_PATH" ]]; then
            result_run_id="$(sed -n 's/^[[:space:]]*"run_id":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
            result_state="$(sed -n 's/^[[:space:]]*"state":[[:space:]]*"\([^"]*\)".*/\1/p' "$RESULT_PATH" | head -n 1 || true)"
        fi
        if [[ "$result_run_id" != "$RUN_ID" || "$result_state" != "FAILED" ]]; then
            write_result "FAILED" "Preflight or launcher failed at ${CURRENT_STAGE} (exit ${status})"
        fi
        printf 'M0 invocation %s failed at %s; failure result: %s\n' \
            "$RUN_ID" "$CURRENT_STAGE" "$RESULT_PATH" >&2
    fi
}
trap on_exit EXIT

CURRENT_STAGE="TOOL_CHECK"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    printf 'ROBOTSIM_M0_SEED must be a non-negative integer\n' >&2
    exit 2
fi
command -v git >/dev/null || { printf 'git is required\n' >&2; exit 2; }
command -v uv >/dev/null || { printf 'uv is required to create the pinned Python 3.10 environment\n' >&2; exit 2; }
command -v realpath >/dev/null || { printf 'realpath is required for mesh provenance validation\n' >&2; exit 2; }

CURRENT_STAGE="ROBOTSIM_IDENTITY"
ROBOTSIM_SHA="$(git -C "$ROOT_DIR" rev-parse HEAD)"
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=all)" ]]; then
    ROBOTSIM_DIRTY_JSON=true
else
    ROBOTSIM_DIRTY_JSON=false
fi
write_result "PREFLIGHT" ""

ensure_checkout() {
    local label="$1"
    local url="$2"
    local revision="$3"
    local path="$4"
    local sha_var="$5"
    local dirty_var="${sha_var%_SHA}_DIRTY_JSON"

    if [[ -e "$path" && ! -d "$path/.git" ]]; then
        printf 'Refusing to use non-Git %s path\n' "$label" >&2
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
    printf -v "$sha_var" '%s' "$actual"
    write_result "PREFLIGHT" ""
    if [[ "$actual" != "$revision" ]]; then
        printf '%s checkout has the wrong pinned revision\n' "$label" >&2
        return 1
    fi

    local status_output
    status_output="$(git -C "$path" status --porcelain --untracked-files=all)"
    if [[ -n "$status_output" ]]; then
        printf -v "$dirty_var" '%s' true
        write_result "PREFLIGHT" ""
        printf 'Refusing dirty %s checkout\n' "$label" >&2
        return 1
    fi
    printf -v "$dirty_var" '%s' false
    write_result "PREFLIGHT" ""
}

CURRENT_STAGE="HUMANOID_VLA_CHECKOUT"
ensure_checkout \
    "Humanoid VLA" \
    "https://github.com/ozkannceylan/humanoid_vla.git" \
    "$UPSTREAM_COMMIT" \
    "$CANDIDATE_DIR" \
    CANDIDATE_SHA

CURRENT_STAGE="UNITREE_CHECKOUT"
ensure_checkout \
    "Unitree MuJoCo meshes" \
    "https://github.com/unitreerobotics/unitree_mujoco.git" \
    "$UNITREE_COMMIT" \
    "$UNITREE_DIR" \
    UNITREE_SHA

CURRENT_STAGE="MESH_PROVENANCE"
UNITREE_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR")"
G1_DIR_RESOLVED="$(realpath -e "$UNITREE_DIR_RESOLVED/unitree_robots/g1")"
case "$G1_DIR_RESOLVED" in
    "$UNITREE_DIR_RESOLVED"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="g1_model_directory_resolves_outside_unitree_checkout"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 asset directory escapes the pinned Unitree checkout\n' >&2
        exit 2
        ;;
esac
VERIFIED_MESH_ROOT="$(realpath -e "$G1_DIR_RESOLVED/meshes")"
case "$VERIFIED_MESH_ROOT" in
    "$G1_DIR_RESOLVED/meshes"|"$G1_DIR_RESOLVED/meshes"/*) ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="resolved_g1_mesh_root_escapes_verified_tree"
        write_result "PREFLIGHT" ""
        printf 'Resolved G1 mesh root escapes the verified G1 asset directory\n' >&2
        exit 2
        ;;
esac
MESH_REQUESTED="${ROBOTSIM_M0_MESH_DIR:-$VERIFIED_MESH_ROOT}"
if ! MESH_DIR="$(realpath -e "$MESH_REQUESTED")" || [[ ! -d "$MESH_DIR" ]]; then
    MESH_PATH_CLASS="missing_or_unresolvable"
    MESH_RELATIVE_PATH="not_available"
    write_result "PREFLIGHT" ""
    printf 'G1 mesh directory is missing or cannot be resolved\n' >&2
    exit 2
fi
case "$MESH_DIR" in
    "$VERIFIED_MESH_ROOT"|"$VERIFIED_MESH_ROOT"/*)
        MESH_PATH_CLASS="pinned_unitree_g1_mesh_tree"
        MESH_RELATIVE_PATH="$(realpath --relative-to="$UNITREE_DIR_RESOLVED" "$MESH_DIR")"
        ;;
    *)
        MESH_PATH_CLASS="external_rejected"
        MESH_RELATIVE_PATH="outside_verified_unitree_g1_mesh_tree"
        write_result "PREFLIGHT" ""
        printf 'ROBOTSIM_M0_MESH_DIR must resolve inside the pinned Unitree G1 mesh tree\n' >&2
        exit 2
        ;;
esac
write_result "PREFLIGHT" ""

CURRENT_STAGE="PYTHON_ENVIRONMENT"
VENV_DIR="$RUN_ROOT/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
if [[ -e "$VENV_PYTHON" ]]; then
    if [[ ! -x "$VENV_PYTHON" ]]; then
        printf 'Existing M0 venv Python is not executable\n' >&2
        exit 2
    fi
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'Existing M0 venv must use Python 3.10.x; found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
else
    uv venv --python 3.10 "$VENV_DIR"
    PYTHON_VERSION="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"
    write_result "PREFLIGHT" ""
    if [[ "$PYTHON_VERSION" != 3.10.* ]]; then
        printf 'uv created an incompatible Python venv; expected 3.10.x, found %s\n' "$PYTHON_VERSION" >&2
        exit 2
    fi
fi

CURRENT_STAGE="DEPENDENCY_INSTALL"
uv pip install --python "$VENV_PYTHON" -r "$ROOT_DIR/simulation/mujoco/requirements-m0.txt"
read -r MUJOCO_VERSION NUMPY_VERSION H5PY_VERSION OPENCV_VERSION < <(
    "$VENV_PYTHON" -c 'import cv2,h5py,mujoco,numpy; print(mujoco.__version__, numpy.__version__, h5py.__version__, cv2.__version__)'
)
write_result "PREFLIGHT" ""

LOG_PATH="$OUTPUT_DIR/run.log"
VIDEO_PATH="$OUTPUT_DIR/m0_pick_place.mp4"
SCREENSHOT_PATH="$OUTPUT_DIR/m0_final.png"
CURRENT_STAGE="RUNTIME"
cd "$ROOT_DIR"
set +e
MUJOCO_GL=egl PYTHONPATH="$CANDIDATE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV_PYTHON" -m simulation.mujoco.m0_pick_place \
        --candidate-root "$CANDIDATE_DIR" \
        --unitree-root "$UNITREE_DIR" \
        --mesh-dir "$MESH_DIR" \
        --output-dir "$OUTPUT_DIR/model" \
        --output-json "$RESULT_PATH" \
        --video "$VIDEO_PATH" \
        --screenshot "$SCREENSHOT_PATH" \
        --run-id "$RUN_ID" \
        --seed "$SEED" \
    2>&1 | tee "$LOG_PATH"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if (( status == 0 && pipeline_status[1] != 0 )); then
    status=${pipeline_status[1]}
fi
set -e
exit "$status"
