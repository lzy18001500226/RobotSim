# Issue #43 M0 Status

CURRENT STAGE: Launcher provenance and reproducibility fixes are in progress. The historical episodes below predate these fixes and are not final evidence for the current launcher revision.

The earlier box-dominant geometry and center-only placement run is superseded. All measurements below are historical evidence for the prior acceptance-gap code, not validation of the launcher changes now in progress.

## Historical Acceptance-Gap Revision

The earlier acceptance-gap runs used seed 42 and passed the task checks, but their artifacts do not contain the current run identity and mesh provenance contract. They are retained as historical evidence only. JSON, physics traces, videos, screenshots, and logs remain outside the repository:

- Run 1: `/tmp/robotsim-issue43-m0/pr45-final-1/`
- Run 2: `/tmp/robotsim-issue43-m0/pr45-final-2/`

The bottle consists of four cylindrical collision geoms: body radius 0.075 m / half-length 0.100 m; shoulder 0.060 / 0.025 m; neck 0.022 / 0.024 m; cap 0.026 / 0.008 m. Combined mass is 0.315 kg. Placement requires every projected bottle geom to clear the table edge by 0.030 m and retains the target-table contact requirement. The minimum observed per-geom clearance was 0.0700 m.

Across 14,080 physics steps per run, maximum translation was 0.0009113 m (limit 0.005 m), maximum angular jump was 0.004569 rad (limit 0.025 rad), and maximum penetration was 0.002428 m (limit 0.025 m). The measured-contact weld activated at 2.208 s with a 0.00003384 m / 0.00001140 rad pose change and released at 24.544 s with a 0.000001293 m / 0.00001276 rad change. Both events include exact before/after poses in the result JSON. The final object pose was `[0.294515, -0.344947, 1.099234]` m, target XY error 0.007386 m, with target contact and the complete footprint inside the configured margin.

CHOSEN BASELINE: ozkannceylan/humanoid_vla bimanual G1 MuJoCo controller.

UPSTREAM REPO/COMMIT: https://github.com/ozkannceylan/humanoid_vla @ 3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12; G1 meshes from https://github.com/unitreerobotics/unitree_mujoco @ 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d.

WHY SELECTED: MIT-licensed source provides the shortest usable path to a G1 29-DoF palm-pad model, physical bimanual contact, Jacobian IK, and PD torque control. RobotSim reuses its PhysicsSim and bimanual trajectory planner, then adds the bottle scene, measured-contact grasp attachment, transfer/lower/release sequence, rendering, and acceptance checks. Unitree meshes retain their separate BSD license.

## Phase 1 Candidate Comparison

| Repository / commit | License | Robot / hand / simulator | Controller, assets, task | Reproduction and adaptation |
|---|---|---|---|---|
| [NVlabs/humanoidmimicgen](https://github.com/NVlabs/humanoidmimicgen) `d82844dcec242c82d6b82628ccc45933c3ad5cbd` | Apache-2.0 source with separately documented RoboCasa/RoboSuite MIT and Unitree BSD assets; downloaded WBC weights have NVIDIA Open Model License | Unitree G1, three-finger hand, MuJoCo 3.2.6 | Nine G1 tasks including Drill PnP; public recorded-action replay and task-success checks; WBC and task data are available separately | Python 3.10; core pins MuJoCo 3.2.6, NumPy 1.26.4, RoboSuite 1.5.1 and robosuite-models 1.0.0. Recorded replay is CPU-capable; public replay data is an external download. Cylinder adaptation is moderate/high. |
| [ozkannceylan/humanoid_vla](https://github.com/ozkannceylan/humanoid_vla) `3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12` | MIT; its G1 meshes come from a separate Unitree checkout | G1 29 DoF with palm pads, MuJoCo 3.x | Scripted Jacobian IK demos cover cube place; ACT checkpoint is not in the checkout and the single-arm path uses a weld/kinematic update. Its separate bimanual demo uses physical friction contacts but only lifts a box. | Model assets need an external checkout; package declares NumPy, h5py, OpenCV, MuJoCo, Torch and TorchVision. Useful controller reference, but physical table-to-table placement needs adaptation. |
| [luckyrobots/g1-manipulation-challenge](https://github.com/luckyrobots/g1-manipulation-challenge) `0e9d1f9b61772905c9f7997b527f221dc7800fc9` | No repository license found; not reusable for code/assets | G1 29 DoF with Dex3, MuJoCo | Exact two-table cylinder scene; walker/croucher/rotator/reacher ONNX assets and manual hand toggle, but no automated pick-place solution | Small MuJoCo/ONNX dependencies and low scene adaptation, but no license and no completed task controller disqualify it as an adoption baseline. |
| [maxwellrobotics/g1-ros2](https://github.com/maxwellrobotics/g1-ros2) `ace298393ec6cadc1f4a66e70a3311e1d2c4d7ff` | BSD-3-Clause | G1 29 DoF with Dex3, `unitree_mujoco` | MoveIt and manipulation actions perform an in-reach cube pick/place; the broader mission also uses locomotion and navigation | ROS 2 Jazzy/Ubuntu 24.04 dev container, MoveIt, GPU-backed simulator and many packages. Exact task is useful evidence but the runtime/ROS integration is out of M0 scope. |
| [unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc` | Apache-2.0 | G1 29 DoF with Dex1 or Dex3, Isaac Sim / Isaac Lab | Existing cylinder pick-place tasks and simulation weights | Exact object/task match, but not MuJoCo; requires Isaac Sim/Isaac Lab and supported NVIDIA RTX hardware, so it is not a compatible baseline. |

Other investigated candidates included Robotics-Ark/ark_unitree_g1 `95775cad397e90ca95198b02c0cbeab4d7418bc3` (MIT; G1/Dex3 MuJoCo model and IK hand-motion sequence, but no object scene or task acceptance) and wiscohumanoids/manipulation_sim (generic arm, not G1; no license found).

REPRO COMMAND: From a fresh WSL2 checkout with `git`, `uv`, public network access for first fetch/install, and EGL/Mesa runtime available, run exactly `./scripts/run_m0_pick_place.sh`. The executable launcher verifies both pinned commits and clean checkout state, enforces Python 3.10.x, restricts mesh resolution to the verified Unitree G1 mesh tree, and creates a unique run directory beneath `/tmp/robotsim-issue43-m0/output`. Completed artifacts are JSON, physics trace, log, MP4, and PNG. CPU rendering uses offscreen EGL; no GPU, display, ROS 2, or Unity is required.

HISTORICAL RUN MEASUREMENTS: The unchanged upstream H-VLA bimanual demo lifted its object 7.6 cm over 174 frames. The prior revision runs completed GRASP at 2.208 s, LIFT at 4.320 s, TRANSFER at 14.592 s, RELEASE at 24.704 s, and PLACE at 28.160 s (882 control frames; 500 Hz physics, 31.25 Hz control). Final object position was `[0.294515, -0.344947, 1.099234]` m, target XY error 0.007386 m, and final linear/angular speeds were 0.008877 m/s and 0.115378 rad/s. Those runs passed the task checks but lack the new run identity/provenance fields.

WHAT FAILED: HMG recorded low-level action replay diverged from its reference trajectory (late maximum state divergence about 2.85) and did not reach task success; its separate WBC-goal replay passed and remains candidate evidence, not the selected adapter. Initial widened-table variants interfered with grasp or prevented stable support and were discarded. No vendor checkout was modified.

AUTOMATIC METRICS: PASS requires bilateral palm force contact (at least 2 N per palm for 5 frames), at least 5 cm lift, all projected bottle collision geoms inside the target tabletop with 3 cm margin, contact-free release for 5 frames, target-table contact, and at least 1.0 s stable time (33 consecutive samples, 1.024 s at 31.25 Hz). Physics-step limits are 5 mm translation and 0.025 rad rotation; weld-event limits are 2 mm and 1 degree; penetration must remain at or below 25 mm. Linear/angular speed limits remain 0.03 m/s and 0.20 rad/s; control-frame translation remains bounded by 0.20 m. These acceptance checks passed in the historical pair; the current launcher revision still requires two clean full runs.

FINAL DEMO COMMAND: `./scripts/run_m0_pick_place.sh`. Set `ROBOTSIM_M0_CANDIDATE_DIR`, `ROBOTSIM_M0_UNITREE_DIR`, `ROBOTSIM_M0_RUN_DIR`, or `ROBOTSIM_M0_OUTPUT_DIR` only to select cache/output locations. `ROBOTSIM_M0_MESH_DIR` is accepted only when its resolved path is inside the pinned Unitree checkout's `unitree_robots/g1/meshes` tree. Every invocation creates its own run ID and begins with a durable `PREFLIGHT`, `passed: false` result record.

KNOWN SHORTCUTS: The G1 base is fixed, object/table poses are deterministic ground truth, and the script uses a measured-contact runtime weld during grasp. The final stable pose is side-lying; upright orientation is reported but is not an acceptance requirement. The adapter runtime pins MuJoCo 3.2.6; the repository's separate G1 smoke suite was validated on MuJoCo 3.3.6.

FINAL STATUS: PENDING - rerun launcher regression checks and two clean fixed-seed full episodes after provenance, stale-PASS, Python-version, and latest-main integration changes. PR #45 remains Draft.
