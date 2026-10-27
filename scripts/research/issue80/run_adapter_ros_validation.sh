#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
image="robotsim:humble-raicom-runtime-20261019"
expected_image_id="sha256:fed10384dea88322debb4586ffb218f888c184ca0bbc31b576d9b3a3b0adcd31"
actual_image_id="$(docker image inspect --format '{{.Id}}' "${image}")"
if [[ "${actual_image_id}" != "${expected_image_id}" ]]; then
  printf 'Expected local image %s with ID %s; found %s\n' \
    "${image}" "${expected_image_id}" "${actual_image_id}" >&2
  exit 1
fi

docker run --rm \
  --pull=never \
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
