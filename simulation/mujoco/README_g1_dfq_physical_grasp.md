# G1 Inspire DFQ Physical Grasp

This runner evaluates one fixed C2 right-hand grasp candidate in MuJoCo 3.3.6. It is a `SIMULATION_ONLY_M0` abstraction, not a claim of DFQ hardware-control equivalence. The prior C2 static/wrench result is only an initialization hypothesis; dynamic contact, measured load transfer, table separation, suspension, and free-physics release are gated separately.

The runner uses the accepted G1 body, corrected DFQ converter output, frozen DFQ meshes and wrist mounts, C2 finger targets, canonical free-jointed 0.570 kg bottle, accepted table, source joint limits, and the 0.531 Nm simulation-only effort cap. It does not write bottle qpos during active rollout, add a bottle constraint or mocap body, or command the left hand. The active velocity limit remains 0.5 rad/s. After measured contact hold, a constrained force allocation is recomputed from current force-bearing contacts, live contact normals/friction, current point Jacobians reduced through the official mimic ratios, and current driver effort. Support demand follows the measured table-load deficit and is rate-limited; total normal/contact/resultant and driver effort retain reserve below their unchanged hard caps. The controller remains active during the ordered 0.1 mm, 0.25 mm, and 0.5 mm transfer checkpoints, then the 1 mm, 5 mm, and 30 mm free-bottle lift, one-second hold, and physical release. Each transfer stage must show noise-aware table unloading matched by measured upward hand support while contact windows remain valid and the left hand stays clear. The fixed-wrist hand-only LOAD_BUILD path is skipped and is not reported as LOAD_READY. No velocity is directly clamped and no follower state is written.

## Runtime

- WSL2 Ubuntu 22.04
- Python 3.10.x
- MuJoCo 3.3.6
- NumPy 1.26.4
- SciPy 1.14.1 (`scipy.optimize.linprog` for bounded live-contact allocation)
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

Every invocation creates a unique directory under `ROBOTSIM_EVIDENCE_ROOT/runs/`. A failed preflight or physical gate writes a durable `m0_result.json`, `run.log`, `REPORT.md`, and reproduction command for that invocation. The dynamic contact Jacobians and bounded solver allocations are preserved in `live_contact_allocation_trace.json`; contact, effort, bottle-pose, arm, load-transfer, and lift-stage traces are written separately.

## Gate meanings

- `HAND_ACTUATION`: all six right-hand command channels show measured CLOSE-phase joint travel.
- `FINGER_BOTTLE_CONTACT`: thumb plus at least one opposing digit pass the one-second contact HOLD gate.
- `CONTACT_FORCE_WITHIN_CAP`: measured total normal force, individual and total resultants, and applied effort stay below the existing hard caps.
- `UPWARD_HAND_SUPPORT`: measured upward support from right-hand contacts increases during transfer.
- `TABLE_LOAD_TRANSFER`: table normal force falls by a noise-aware amount as hand support rises; `LOAD_READY` additionally requires airborne geometry and measured force/contact/mimic/velocity/effort checks.
- `AIRBORNE_1MM` and `AIRBORNE_5MM`: the free-jointed bottle reaches the indicated height with table support absent.
- `PHYSICAL_LIFT_30MM`: the free-jointed bottle rises at least 30 mm and table support is absent.
- `HOLD_1S`: the bottle remains lifted for a completed one-second post-lift hold.
- `PHYSICAL_RELEASE`: the fingers open, hand contact clears, and the bottle settles on the table under free physics.
- `PR_CI`: all required GitHub checks pass for the exact pushed PR head. A physical pass remains pending overall until this gate is recorded.

When an earlier gate prevents a later stage from running, the later gate remains failed/unverified with that prerequisite failure in its evidence. No unrun physical stage is reported as passed.
