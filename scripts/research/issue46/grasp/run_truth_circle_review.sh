#!/usr/bin/env bash
# Reuse the approved-circle adapter; no direct legacy replanner entry point.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo"
reviews=/home/lzy18001500226/projects/RobotSim/reviews/issue-46
py=/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python
export DISPLAY="${DISPLAY:-:0}"
exec "$py" scripts/research/issue46/grasp/stl_approved_circle_native_trial.py \
  --baseline "$reviews/raicom-stl-xvfb-native-reset-stand-20261105T230932Z" \
  --circle-review "$reviews/stl-variable-support-circle-review-20261105T080615Z/scene.json" \
  --candidate "$reviews/stl-standing-side-grasp-candidate-20261104T021318Z/result.json" \
  --native-physics-audit \
  --target-source truth-diagnostic \
  --execute-simulation "$@"
