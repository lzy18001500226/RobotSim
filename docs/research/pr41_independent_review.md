# Independent technical review: RobotSim PR #41

- PR: [#41 — Add minimal ROS 2 workspace and G1 adapter](https://github.com/lzy18001500226/RobotSim/pull/41)
- Expected and reviewed head: `2908a7da621f9e39c4f765619561d6aa18303119`
- Main observed during review: `11aebeb60a1d3c1c2f580c8f4d6bd6907581fd33`
- Review type: read-only source, contract, workflow, and public CI-status review
- Result: **REQUEST CHANGES**

## BLOCKING

### B1 — A rollback can be followed by an old-generation command

The SDK2 `LowState` callback updates `latest_low_state_` and `state_revision_` on the SDK callback thread. The simulation generation changes later, when the ROS state timer reads the lower `tick` and calls `StateTimeline::next()`. Until that timer runs, `timeline_.generation()` and the command buffer still name the prior generation. The independent 10 ms command timer reads the new `LowState` and can still publish the prior buffered torque command. An old-generation command callback can also be accepted during this interval because it validates against the not-yet-updated timeline.

Relevant code: [`g1_ros2_adapter.cpp` at PR head](https://github.com/lzy18001500226/RobotSim/blob/2908a7da621f9e39c4f765619561d6aa18303119/robots/unitree_g1/ros2/robotsim_g1_adapter/src/g1_ros2_adapter.cpp#L120-L165), especially the state update and deferred generation reset, and [the command timer](https://github.com/lzy18001500226/RobotSim/blob/2908a7da621f9e39c4f765619561d6aa18303119/robots/unitree_g1/ros2/robotsim_g1_adapter/src/g1_ros2_adapter.cpp#L225-L257). The command buffer correctly rejects mismatched generations once reset, but that reset happens too late for this interleaving: [`latest_command_buffer.cpp`](https://github.com/lzy18001500226/RobotSim/blob/2908a7da621f9e39c4f765619561d6aa18303119/ros2_ws/src/robotsim_core/src/latest_command_buffer.cpp#L20-L68).

This violates the simulation-state contract: rollback is a generation boundary and pending commands from the old generation must be discarded before post-rollback control can be applied. It can apply stale torque to the reset simulation. Before merge, fail closed across a pending rollback (for example, flag a rollback at LowState intake and inhibit command output until the new generation is established), and add a test that forces the state/command timer ordering around a rollback. The existing unit test checks the timeline in isolation, not this adapter interleaving.

## NON-BLOCKING

- **Lifecycle during pause:** the adapter always writes `LIFECYCLE_RUNNING` ([adapter line 175](https://github.com/lzy18001500226/RobotSim/blob/2908a7da621f9e39c4f765619561d6aa18303119/robots/unitree_g1/ros2/robotsim_g1_adapter/src/g1_ros2_adapter.cpp#L167-L181)). The pinned SDK2 bridge publishes `LowState` from a 1 kHz recurrent thread and derives `tick` from `mj_data->time`; it can therefore keep publishing repeated timestamps while physics is paused ([pinned bridge](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/simulate/src/unitree_sdk2_bridge.h#L168-L230)). The sequence behavior for equal timestamps is correct, and `/clock` exposes the stopped simulation time, but the lifecycle field is not truthful while paused. Add a pause signal or explicitly constrain/document this adapter's lifecycle scope.
- **RMW selection is documented, not enforced:** the workflow sets `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` for ROS processes and describes the observed SDK2 Cyclone DDS incompatibility with ROS Cyclone RMW. The executable checks domain 73, `ROS_LOCALHOST_ONLY=1`, and rejects `CYCLONEDDS_URI`, but does not check the selected RMW. A misconfigured environment may fail during SDK channel initialization rather than at the intended guard. The documented Fast DDS plus SDK2 Cyclone arrangement is appropriately limited to loopback/domain 73 and does not claim to settle production transport.
- **Pause/rollback integration coverage:** the live test verifies ordinary state, command, timeout, QoS, `/clock`, and static-TF behavior. The state-timeline unit test covers rollback and sequence reset; no live rollback scenario proves generation change and old-command invalidation together. The race in B1 is the required correction.
- **Evidence summary:** PR text reports L2 model smoke and L3 build/runtime results, including 3 packages, 8 colcon tests, 0 failures, and a live round trip. The workflow provides reproduction commands and pass conditions, but the PR body is a summary rather than attached raw logs/artifacts. This does not invalidate the source review; preserve raw logs with the next validation record if that evidence is expected to be durable.

## NIT

- The 29-joint list matches the pinned SDK2 G1 order (12 leg, 3 waist, 14 arm joints) in [`defines.h` at SDK2 commit `63096d0`](https://github.com/unitreerobotics/unitree_sdk2/blob/63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36/include/unitree/dds_wrapper/robots/g1/defines.h); name-based input reordering rejects unknown, duplicate, and missing joints. Vendor SDK types remain under the G1 adapter; shared interfaces/core do not depend on SDK2 types.
- Root pose and twist are explicitly marked unavailable and are not synthesized. The code emits no `world -> base_link` edge. The pinned [G1 29DoF MJCF](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/g1/g1_29dof.xml#L62-L67) places its `imu` site at the pelvis origin with no site rotation, so the emitted identity `base_link -> imu_link` static transform matches the declared waist/root `base_link` assumption.
- `/clock` and state stamps both use `LowState.tick` in milliseconds, sourced by the pinned simulator as rounded MuJoCo simulation time. A backward tick creates a new generation and resets state/command sequence to zero. State and command QoS are bounded (`KeepLast(1)`); commands are latest-value, one-source, finite/range checked, generation and strictly increasing sequence checked, and expire from local `steady_clock` receipt time.
- The adapter preserves live `mode_machine` from `LowState` and computes the SDK2 `LowCmd` CRC using the same `sizeof(LowCmd) / 4 - 1` word-count convention as the pinned SDK2 CRC implementation and examples. The current `rclcpp::spin` uses a single-threaded executor; the SDK callback protects shared `LowState` with a mutex. The timeline and command buffer are safe under this executor arrangement; preserve that assumption or add synchronization if executor concurrency changes.
- The PR records `Xvfb` viewer shutdown as X11 `BadDrawable` with a non-zero GUI shutdown status and separately states WSLg/GPU/hardware were not tested. This limitation is disclosed and does not overstate Unity, hardware, or Sim2Real validation.
- The G1 name-map test checks reordered values and invalid name sets. The live test checks presence of `base_link -> imu_link`; it could more directly assert the transform values and the absence of `world -> base_link`.

## Validation and status

- Reviewed the exact expected PR head and current main SHA above.
- Public GitHub checks page showed successful status for **Agent infrastructure checks** and **Headless G1 model and stepping test** on the reviewed head.
- The PR records L2 and L3 as PASS. The described L2/L3 commands and claims were read; they were not rerun as part of this read-only review.
- `git diff --check` for the PR diff against current main passed.
- No files on PR #41 were changed. No merge was performed.
- GitHub API commenting was unavailable in this environment (403 / invalid injected `GH_TOKEN`; public PR page was readable but comment permission unavailable). This report is therefore being published on the requested report-only branch.
