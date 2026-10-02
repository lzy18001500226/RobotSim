#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash

SDK="$ROOT/third_party/unitree_sdk2"
SIM="$ROOT/third_party/unitree_mujoco"
PREFIX="$ROOT/third_party/install/unitree_robotics"

if [[ ! -d "$SDK" || ! -d "$SIM" ]]; then
  echo "Run scripts/fetch_third_party.sh first."
  exit 1
fi

mkdir -p "$PREFIX"

cmake -S "$SDK" -B "$SDK/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$SDK/build" -j"$(nproc)"
cmake --install "$SDK/build"

export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

ln -sfn "$MUJOCO_HOME" "$SIM/simulate/mujoco"

cmake -S "$SIM/simulate" -B "$SIM/simulate/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SIM/simulate/build" -j"$(nproc)"

echo
echo "G1 simulator build complete."
echo "Persistent SDK install prefix:"
echo "  $PREFIX"
echo
echo 'Next: set robot: "g1" in third_party/unitree_mujoco/simulate/config.yaml'
