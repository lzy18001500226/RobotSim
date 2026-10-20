#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
image="robotsim:humble-raicom-runtime-20261019"

docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,exec,nosuid,size=2g \
  --env HOME=/tmp/home \
  --env ROS_DOMAIN_ID=194 \
  --env ROS_LOCALHOST_ONLY=1 \
  --env ROBOTSIM_ISOLATED_ROS_GRAPH=1 \
  --volume "${repo_root}/ros2_ws/src/robotsim_navigation:/input/robotsim_navigation:ro" \
  --entrypoint /bin/bash \
  "${image}" \
  -lc 'set -eo pipefail
    mkdir -p /tmp/home/.ros/log /tmp/ros2_ws/src
    source /opt/ros/humble/setup.bash
    cp -a /input/robotsim_navigation /tmp/ros2_ws/src/robotsim_navigation
    cd /tmp/ros2_ws
    colcon build --packages-select robotsim_navigation
    source install/setup.bash
    python3 -m pytest -q /tmp/ros2_ws/src/robotsim_navigation/test --junitxml=/tmp/ros2_ws/pytest-results.xml'
