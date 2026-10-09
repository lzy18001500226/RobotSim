# G1 Inspire DFQ Physical Grasp

This runner evaluates one fixed C2 right-hand grasp candidate in MuJoCo 3.3.6. It is a `SIMULATION_ONLY_M0` abstraction, not a claim of DFQ hardware-control equivalence. The prior C2 static/wrench result is only an initialization hypothesis; dynamic contact, measured load transfer, table separation, suspension, and free-physics release are gated separately.

The runner uses the accepted G1 body, corrected DFQ converter output, frozen DFQ meshes and wrist mounts, C2 finger targets, canonical free-jointed 0.570 kg bottle, accepted table, source joint limits, and the 0.531 Nm simulation-only effort cap. It does not write bottle qpos during active rollout, add a bottle constraint or mocap body, or command the left hand. The active velocity limit remains 0.5 rad/s. After measured contact hold, a bounded one-second torque-impedance/static-LP-feedforward preload runs with the wrist fixed; this is explicitly not a LOAD_READY claim. The same torque controller remains active during the existing quintic G1 arm lift and post-lift hold. The fixed-wrist hand-only LOAD_BUILD path is skipped and is not reported as LOAD_READY. No velocity is directly clamped and no follower state is written.

## Runtime

- WSL2 Ubuntu 22.04
- Python 3.10.x
- MuJoCo 3.3.6
- NumPy 1.26.4
- OpenCV headless 4.10.0.84 with MP4V writer support
- EGL/Mesa runtime for offscreen rendering
- Pinned, clean upstream source checkouts:
  - `unitreerobotics/unitree_ros` at `5994d4faef0a9cadd3287f8de0199a67eeb2a259`
  - `unitreerobotics/unitree_mujoco` at `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`

The generated MJCF and candidate JSON are committed under `fixtures/g1_dfq_m0_c2/`. Vendor meshes remain in the pinned upstream checkouts and are resolved at runtime. The runner verifies each upstream revision and a clean checkout before compilation.

## Reproduction

Create the runtime once:

```bash
python3.10 -m venv /tmp/robotsim-issue43-g1-physical-grasp/venv
/tmp/robotsim-issue43-g1-physical-grasp/venv/bin/pip install -r simulation/mujoco/requirements-g1-dfq-grasp.txt
```

The source checkouts may be placed anywhere and selected with the environment variables below. Each checkout must be clean and at the exact pinned revision. The exact command used for a run is copied into `m0_result.json`, `run.log`, `REPORT.md`, and `REPRODUCE.md`.

```bash
MUJOCO_GL=egl \
ROBOTSIM_MUJOCO_PYTHON=/tmp/robotsim-issue43-g1-physical-grasp/venv/bin/python \
ROBOTSIM_EVIDENCE_ROOT=/mnt/c/Users/HP/Desktop/Robot/reviews/issue43-g1-physical-grasp-priority-p0-20261009 \
ROBOTSIM_UNITREE_ROS_DIR=/tmp/robotsim-issue43-g1-physical-grasp/upstream/unitree_ros \
ROBOTSIM_UNITREE_MUJOCO_DIR=/tmp/robotsim-issue43-stock-arm-smoke/upstream/unitree_mujoco \
./scripts/run_g1_dfq_physical_grasp.sh
```

Every invocation creates a unique directory under `ROBOTSIM_EVIDENCE_ROOT/runs/`. A failed preflight or physical gate writes a durable `m0_result.json`, `run.log`, `REPORT.md`, and reproduction command for that invocation.

## Gate meanings

- `HAND_ACTUATION`: all six right-hand command channels show measured CLOSE-phase joint travel.
- `FINGER_BOTTLE_CONTACT`: thumb plus at least one opposing digit pass the one-second contact HOLD gate.
- `ISOLATED_PHYSICAL_LIFT_30MM`: the free-jointed bottle rises at least 30 mm and table support is absent.
- `HOLD_1S`: the bottle remains lifted for a completed one-second post-lift hold.
- `PHYSICAL_RELEASE`: the fingers open, hand contact clears, and the bottle settles on the table under free physics.
- `FULL_WORKCELL_PICKUP`: separate station-integration evidence; this runner does not mark it passed.

When an earlier gate prevents a later stage from running, the later gate remains failed/unverified with that prerequisite failure in its evidence. No unrun physical stage is reported as passed.
