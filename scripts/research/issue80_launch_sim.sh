#!/bin/bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash

current_dir=/workspace/raicom2026/sim_mujoco/bin
install_dir="$current_dir/.."
export LD_LIBRARY_PATH="$current_dir:$install_dir/bin:$install_dir/lib:$LD_LIBRARY_PATH"
export EM_APP_NAME=sim
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export AGIBOT_ENABLE_HDS_COMPONENT=false
export AGIBOT_ENABLE_EVENT_COMPONENT=false
export AGIBOT_ENABLE_MONITOR_COMPONENT=false
export AGIBOT_ENABLE_MONITOR=true
export AGIBOT_ENABLE_AUDIT_COMPONENT=false
export AIMRT_PLUGIN_SHM_DEFAULT_WAIT_TIME_US=300
export SIM_ROBOT_PATH=/workspace/raicom2026/sim_mujoco/configuration/robot/lx2501_3_t2d5
export SIM_RESOURCE_MODEL_PATH="$install_dir/resource/model"

mkdir -p /workspace/evidence/log /workspace/evidence/cfg/tmp
cd /workspace/evidence
simulator_config="${ROBOTSIM_SIMULATOR_CFG_OVERRIDE:-$SIM_ROBOT_PATH/simulator/default.yaml}"
exec "$current_dir/aima-sim-app" --cfg_file_path="$simulator_config"
