#!/usr/bin/env bash
# Build only the evidence-local SIMULATION_ONLY LiDAR plugin.
set -euo pipefail

source_path="${1:?source .cc path required}"
output_path="${2:?output .so path required}"
runtime_root="${3:?RAICOM sim_mujoco root required}"
python_bin="${MUJOCO_PYTHON_BIN:-python3}"

if [[ -n "${MUJOCO_INCLUDE_ROOT:-}" ]]; then
  include_root="${MUJOCO_INCLUDE_ROOT}"
else
  include_root="$("${python_bin}" -c 'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).parent / "include")')"
fi
test -f "${include_root}/mujoco/mujoco.h"
grep -q '^#define mjVERSION_HEADER 337$' "${include_root}/mujoco/mujoco.h"
mkdir -p "$(dirname "${output_path}")"
g++ -std=c++17 -O2 -fPIC -shared \
  -I"${include_root}" \
  "${source_path}" \
  -L"${runtime_root}/lib" -Wl,-rpath,"${runtime_root}/lib" -lmujoco \
  -o "${output_path}"
sha256sum "${output_path}"
