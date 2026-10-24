#!/bin/bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash
export LOG_PATH=/workspace/evidence/mc_logs
mkdir -p "$LOG_PATH"
cd /workspace/raicom2026/mc/bin
exec ./em_run_wsl2.sh
