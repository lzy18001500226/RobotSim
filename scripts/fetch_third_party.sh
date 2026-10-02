#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

clone_if_missing () {
  local url="$1"
  local dst="$2"
  if [[ ! -d "$dst/.git" ]]; then
    git clone "$url" "$dst"
  fi
}

clone_if_missing https://github.com/unitreerobotics/unitree_sdk2.git "$TP/unitree_sdk2"
clone_if_missing https://github.com/unitreerobotics/unitree_mujoco.git "$TP/unitree_mujoco"
clone_if_missing https://github.com/unitreerobotics/unitree_ros2.git "$TP/unitree_ros2"

cat <<'EOF'

Fetched upstream repositories.

IMPORTANT:
After the first confirmed green build, record exact commit SHAs in:
  third_party/LOCK.md

Do not rely on floating default branches for published reproducibility.
EOF
