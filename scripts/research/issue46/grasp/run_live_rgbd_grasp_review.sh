#!/usr/bin/env bash
# LIVE native mj_step + MC, H300 bottle, RGB-D target, table-standing episode.
# No walking, captured-pose playback, real hardware or alternate robot model.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo"
reviews=/home/lzy18001500226/projects/RobotSim/reviews/issue-46
export DISPLAY="${DISPLAY:-:0}"
export MUJOCO_GL=glfw
/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46/native_runtime/wslg_input_preflight.py
exec bash scripts/research/issue46/grasp/run_truth_circle_review.sh \
  --source-pad-coacd-comparison \
  --initial-hands-fully-open \
  --both-claws-zero-internal-friction \
  --right-claw-damping 1.27 \
  --elliptic-grip-solver-diagnostic \
  --native-render-adapter NVIDIA \
  --rgbd-render-adapter NVIDIA \
  --target-source fresh-rgbd-snapshot \
  --fixed-reviewed-checkpoint "$reviews/raicom-stl-xvfb-native-reset-stand-20261107T005238Z" \
  --grasp-height-above-center-m 0.10 \
  --physical-lift-test \
  --pause-for-human-review "$@"
