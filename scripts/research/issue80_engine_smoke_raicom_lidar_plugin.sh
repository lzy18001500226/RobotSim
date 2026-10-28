#!/usr/bin/env bash
# Compile and run the engine-only plugin smoke against the specified MuJoCo tree.
set -euo pipefail

source_path="${1:?plugin source required}"
smoke_source="${2:?smoke source required}"
plugin_dir="${3:?plugin directory required}"
runtime_root="${4:?RAICOM sim root required}"
output_binary="${5:?smoke binary output required}"
python_bin="${MUJOCO_PYTHON_BIN:-python3}"
if [[ -n "${MUJOCO_INCLUDE_ROOT:-}" ]]; then
  include_root="${MUJOCO_INCLUDE_ROOT}"
else
  include_root="$("${python_bin}" -c 'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).parent / "include")')"
fi
test -f "${include_root}/mujoco/mujoco.h"
grep -q '^#define mjVERSION_HEADER 337$' "${include_root}/mujoco/mujoco.h"

g++ -std=c++17 -O2 -I"${include_root}" "${smoke_source}" \
  -L"${runtime_root}/lib" -Wl,-rpath,"${runtime_root}/lib" -lmujoco -o "${output_binary}"
"${output_binary}" "${plugin_dir}" "${output_binary}.json"
cat "${output_binary}.json"
