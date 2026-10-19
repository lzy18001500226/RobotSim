# Issue #80 offline point-goal controller prototype

This pure-math prototype is isolated from ROS, MuJoCo, MC, Docker, DDS, and GUI. It consumes measured planar Pose2D state and a Goal2D already expressed in the same frame. It returns bounded forward/angular commands and ALIGN, DRIVE, FINAL_ALIGN, SETTLING, ARRIVED, cancellation, timeout, stale-state, frame-mismatch, and invalid-input states.

The PoseStamped helper validates finite planar goal fields, quaternion, and frame equality. A future ROS adapter must perform tf2 transforms at the goal timestamp and separately validate stamp semantics. The controller uses monotonic local receive times for pose freshness and wall deadline checks. It never integrates commanded velocity to estimate position.

Reproduction (standard library only):
cd /home/lzy18001500226/robotsim-issue80-offline-pointgoal-20261020
python3 -B -m unittest discover -s scripts/research/issue80 -p 'test_*.py' -v

The synthetic unicycle trajectory test is mathematical only. It is not MuJoCo, locomotion, ROS integration, or physical navigation evidence. This prototype has no map, obstacle planner, ROS node, MC command bridge, or transform publisher. Keep it under scripts/research; it must not be imported by runtime packages.