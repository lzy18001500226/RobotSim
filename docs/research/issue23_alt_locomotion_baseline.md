# Issue #23: G1 locomotion baseline scout

This report records the completed read-only comparison of released G1
locomotion baselines. No RobotSim runtime code or policy was changed or
trained. Repository and artifact details below are pinned to the listed
commits so the findings can be reproduced.

## Finding

No inspected 29-DoF candidate was verified to both hold position at zero
command and track forward, backward, yaw, and stop without an external support
path. The cleanest measured zero-command behavior came from Unitree's official
G1 MuJoCo example, but that checkpoint actuates only the legs on a 12-DoF
model. It is the best behavioral reference for the next local experiment, not
a plug-in replacement for RobotSim's pinned full-DoF G1.

## Top three

### 1. Unitree RL Gym — best measured zero-command reference

- **Source:** official [`unitreerobotics/unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym), commit
  [`276801e46c5d433564f24658bac64f254b7d2d4b`](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b), dated 2025-07-25.
- **License:** BSD-3-Clause repository license; the pretrained `motion.pt` is
  included in that repository and no separate weight license was found.
- **Robot and model:** G1 with 12 actuated leg joints; MuJoCo dimensions are
  `nq=19, nv=18, nu=12`. The torso and arms are fixed in this model. It is not
  RobotSim's full 29-actuator model.
- **Policy:** committed `deploy/pre_train/g1/motion.pt` (145,745 bytes),
  TorchScript. The observation is 47 float values: body angular velocity,
  projected gravity, three velocity commands, 12 relative joint positions,
  12 joint velocities, 12 previous actions, and two gait-phase values. The
  policy outputs 12 joint-position targets. The deployment applies a 0.25
  action scale and per-joint PD gains.
- **Timing and runtime:** MuJoCo 3.2.3, 2 ms physics step, 10-step
  decimation, 50 Hz policy. CPU-only PyTorch inference worked; no GPU is
  required. The stock script is
  `python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml`.
- **Stock command interface:** `cmd_init` in `deploy_mujoco/configs/g1.yaml`
  is `[0.5, 0, 0]`; the example holds this configured command and has no live
  keyboard/gamepad velocity input. The policy was trained with signed x/y and
  yaw command ranges. The headless check below changed the command vector
  between phases without changing the checkpoint.
- **Free-standing and command result:** headless playback used the committed
  scene and checkpoint, the deployment's observation/PD loop, and no band or
  other support. A zero command for 8 s kept the torso upright (height
  0.773–0.793 m, peak roll/pitch 4.0 degrees) with 0.18 m net planar drift.
  A 0.2 m/s forward command moved 0.74 m in 4 s; -0.2 m/s backward moved
  -0.59 m in 4 s; +0.4 rad/s yaw produced +1.42 rad in 4 s. Returning to zero
  for 8 s produced 0.19 m planar drift. States stayed finite. The zero and
  stop periods still had small gait motion (95th-percentile planar speed
  0.13 m/s), so this is stable upright standing with bounded drift, not a
  perfectly motionless hold.
- **Fall/recovery:** no active get-up or fall-recovery behavior is shown by
  this deployment.
- **Compatibility and effort:** lowest-cost behavior reference and easiest
  CPU reproduction. Porting to RobotSim needs a 12-to-29 joint mapping and a
  decision about the unactuated waist/arms. Model/contact differences also
  prevent treating this as direct validation of the pinned 29-DoF scene.

### 2. MuJoCo Playground — best full-DoF CPU policy, but fails neutral hold

- **Source:** Google DeepMind [`mujoco_playground`](https://github.com/google-deepmind/mujoco_playground), commit
  [`ef4fefc13033c0468af4ef651847f5348af0c7d7`](https://github.com/google-deepmind/mujoco_playground/tree/ef4fefc13033c0468af4ef651847f5348af0c7d7), dated 2026-09-28. Entry point:
  `mujoco_playground/experimental/sim2sim/play_g1_joystick.py`.
- **License:** Apache-2.0 code and committed ONNX policy. The G1 model comes
  from MuJoCo Menagerie commit
  [`1b86ece576591213e2b666ebf59508454200ca97`](https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97)
  and is BSD-3-Clause licensed.
- **Robot and model:** G1 with 29 actuators, `nq=36, nv=35, nu=29`; starts
  from the `knees_bent` keyframe at about 0.755 m. No elastic band is present.
  The model is a Playground/Menagerie MJCF, not RobotSim's pinned Unitree
  `g1_29dof.xml` geometry.
- **Policy:** committed `g1_policy.onnx` (903,543 bytes), CPU ONNX Runtime.
  Observation width 103 includes local pelvis velocity, gyro, projected
  gravity, velocity command, 29 relative joint angles, 29 joint velocities,
  29 previous actions, and four phase values. Output is 29 joint targets.
- **Timing and command interface:** MuJoCo 3.6.0 in this reproduction,
  2 ms physics and 20 ms policy tick (50 Hz). The Logitech F710 gamepad maps
  joystick axes to x/y/yaw; its configured scale is 1.5 m/s, 0.8 m/s and
  2π rad/s. Stock command:
  `python mujoco_playground/experimental/sim2sim/play_g1_joystick.py`.
- **Reproduction result:** a headless run loaded the exact source
  `OnnxController`, committed ONNX and pinned MJCF/mesh assets, with an
  injected command source instead of a physical gamepad. It stayed upright
  and finite through zero, forward, backward, yaw and stop phases. However,
  during 8 s at zero it walked/drifted 1.08 m; during the final 8 s stop it
  drifted 1.12 m. A -0.2 m/s backward command did not produce negative median
  forward velocity. The +0.4 rad/s yaw phase tracked at about 0.43 rad/s.
  This does not satisfy stable zero-command standing or reliable backward
  motion.
- **Dependencies and effort:** CPU inference needs MuJoCo, ONNX Runtime,
  `etils`, `hidapi`, and the pinned Menagerie G1 assets. The repository's
  normal install also brings in the JAX/MJX environment stack even though
  this native sim2sim inference path does not need a GPU. `get_assets()`
  expects the Menagerie checkout; a fresh checkout must fetch that dependency
  before starting the script. Medium setup, low inference cost.
- **Fall/recovery:** no stand-up recovery mode is shown in this entry point.

### 3. Teleop Walking Benchmark — strong walking evidence, poor laptop fit

- **Source:** community [`rhoyn/teleop-walking-benchmark`](https://github.com/rhoyn/teleop-walking-benchmark), commit
  [`6d331a85089975c572624a391d4cca895393aa5d`](https://github.com/rhoyn/teleop-walking-benchmark/tree/6d331a85089975c572624a391d4cca895393aa5d), dated 2026-09-15.
- **License:** MIT harness. The `gr00t_wbc` walk and balance ONNX weights are
  separately identified as NVIDIA Open Model licensed; other included
  checkpoints have their own terms and must not be assumed MIT.
- **Robot and model:** G1 29-DoF scene, but this controller commands only
  15 leg/waist joints. The other 14 arm joints are randomized by the benchmark
  and are not controlled by the locomotion policy.
- **Policy and evidence:** committed `model_walk.onnx` and
  `model_balance.onnx`; source declares 516 observations (86 values over six
  history frames), 15 outputs, 50 Hz control and 2 ms physics. The walk /
  balance selector uses a near-zero command threshold. The benchmark reports
  survival and recovery under an initial three-second crane hold followed by
  release, waypoint walking and perturbations up to 500 N. It does not report
  a dedicated stationary zero-command hold; its README says a wider
  walk/balance switch sweep was inert.
- **Run command:** `./download_weights.sh && ./export_onnx.sh && make && make capture MJWARP_NWORLD=1`,
  then `./run.sh --policy gr00t_wbc_h066_p012 --engine mujoco --runids 0-0`.
  The MuJoCo path uses MuJoCo Warp and captured CUDA graphs, with CUDA,
  TensorRT and NVCC. It is not a CPU deployment path; the build and CUDA
  version stack make it a poor first experiment on a WSL2 RTX 4060 laptop.
- **License/model/fall boundary:** the MIT harness license does not replace
  the checkpoint's NVIDIA Open Model terms. It provides no direct reuse
  evidence for all 29 policy-controlled joints or zero-command standing.

## Additional inspected candidate not ranked for reuse

Community [`Nikerane/g1-switchstep`](https://github.com/Nikerane/g1-switchstep),
commit [`ba9785af5ccf524d0a7e930e572005fee8794722`](https://github.com/Nikerane/g1-switchstep/tree/ba9785af5ccf524d0a7e930e572005fee8794722),
wraps the current Apache-2.0 Unitree RL Lab 29-DoF ONNX policy and a pinned
`unitree_rl_mjlab` scene. Its CLI exposes x/y/yaw and its 4.06 s switch test
contains a short stand/walk/settle sequence, but the zero-command window is
too short to establish a stable hold. The wrapper has no repository license,
requires external scene and policy files, and its integration test contains
macOS-specific paths and FFmpeg assumptions. Treat it as a source reference,
not reusable code.

## Comparison with RobotSim's current baseline

RobotSim pins `unitree_rl_lab` at
[`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3)
and `unitree_mujoco` at
[`1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d).
The simulator model is full 29-DoF (`nq=36, nv=35, nu=29`) and the current
known run is stable only while the elastic band carries substantial weight.

Unitree RL Gym is better evidence for a simple unassisted zero-to-velocity-to-
stop cycle, but trades away full-DoF compatibility and exposes only a fixed
command in its stock MuJoCo script. MuJoCo Playground matches the 29-action
shape, uses CPU inference and has a straightforward gamepad path, but its
neutral command visibly fails the stationary hold and its model geometry is
different. The CUDA benchmark has better perturbation statistics, but does
not validate a zero hold and is costly to reproduce. None proves a successful
full-29-DoF free stand on RobotSim's pinned model.

## Recommended next local experiment

Reproduce the official Unitree RL Gym checkpoint and scene in a disposable
checkout with the stock forward command, then run the same headless
deployment loop with a short command schedule: 8 s zero, 4 s forward at
0.2 m/s, 4 s backward at -0.2 m/s, 4 s yaw at 0.4 rad/s, and 8 s zero.
Record root height, roll/pitch, planar displacement/speed, yaw rate and
finite-state status. The completed cloud headless reproduction used
`mujoco==3.2.3`, `torch==2.14.1+cpu`, NumPy and PyYAML and measured the values
reported above. Use it as a behavior reference only; test compatibility with
RobotSim's pinned 29-DoF model as a separate experiment.
