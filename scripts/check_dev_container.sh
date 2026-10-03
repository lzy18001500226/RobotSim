#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash

if [[ "${ROS_DISTRO:-}" != "humble" ]]; then
  echo "Expected ROS_DISTRO=humble, got ${ROS_DISTRO:-unset}" >&2
  exit 1
fi

ros2 --help >/dev/null 2>&1
printf 'ROS_DISTRO=%s\n' "$ROS_DISTRO"
python3 - <<'PY'
import rclpy
from std_msgs.msg import String

print(f"ROS imports: rclpy={rclpy.__file__}, std_msgs.String={String.__module__}")
PY

cmake --version | head -n 1
ninja --version
mujoco_version
python3 - <<'PY'
import ctypes
import os
from pathlib import Path

library = Path(os.environ["MUJOCO_HOME"]) / "lib" / "libmujoco.so.3.3.6"
if not library.is_file():
    raise SystemExit(f"Missing MuJoCo library: {library}")

mujoco = ctypes.CDLL(str(library))
mujoco.mj_versionString.restype = ctypes.c_char_p
version = mujoco.mj_versionString().decode("ascii")
print(f"Loaded native MuJoCo: {version}")
if not version.startswith("3.3.6"):
    raise SystemExit(f"Expected MuJoCo 3.3.6, got {version}")
PY

if [[ "${ROBOTSIM_REQUIRE_NVIDIA_GPU:-0}" == "1" ]]; then
  nvidia-smi --query-gpu=name,uuid --format=csv,noheader
else
  echo "GPU visibility check skipped; set ROBOTSIM_REQUIRE_NVIDIA_GPU=1 to require it."
fi
