# Issue #46: Reduced-Coordinate OmniHand Transmission

**Result: the affine mimic relations are representable as an exact reduced-coordinate constraint, but this prototype did not meet its close-hold settling gate.** No full-hand OPEN/CLOSE/OPEN rollout, bottle trial, GitHub update, or push was performed.

## Scope and provenance

- RobotSim branch: `codex/x2-omnihand-experimental-20261007`.
- Starting RobotSim HEAD: `ddc67037773de7242d69fa1d108c2f044237f5f5`.
- Official X2 source supplied for this experiment: `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`, `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf`.
- Source URDF SHA-256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.
- The repository's `third_party/LOCK.md` pins Unitree dependencies, not this X2 URDF; the X2 pin above comes from the task's verified source checkout.
- Runtime: Python `3.10.12`, MuJoCo `3.3.6`, timestep `0.002 s`, gravity `[0, 0, -9.81] m/s^2`.
- Compiled complete X2 dimensions: `nq=63`, `nv=63`, `nu=51`, `neq=12`; both native OmniHands and their source-derived visual/collision geometry were loaded. The experiment had one active reduced coordinate; other model coordinates were held at the initialized pose.
- No vendor asset or source was edited.

## Source-to-compiled mapping

The left thumb source chain is serial: `L_thumb_abad -> L_thumb_mcp -> L_thumb_pip -> L_thumb_dip`. All three URDF revolute axes are `[1, 0, 0]`; the importer/probe converts MuJoCo coordinates back into those source coordinates and applies a positive axis sign for these joints.

| Joint | Source relation in source coordinates | Source range (rad) | Effort limit |
|---|---|---:|---:|
| `L_thumb_mcp_joint` | driver `q` | `[-0.8552113334772214, 0]` | `33.5 Nm` |
| `L_thumb_pip_joint` | `1.33 q + 0` | `[-1.1344640137963142, 0]` | `33.5 Nm` |
| `L_thumb_dip_joint` | `1.30 q + 0` | `[-1.1344640137963142, 0]` | `33.5 Nm` |

The selected model relation is therefore exactly `q = [q_mcp, 1.33 q_mcp, 1.30 q_mcp]` and `qdot = [qdot_mcp, 1.33 qdot_mcp, 1.30 qdot_mcp]`. The two compiled thumb equality rows were disabled for the prototype; the other 10 source mimic equality rows remained active. No actuator was assigned to either selected follower.

## Representation feasibility

| Representation | What it guarantees | Assessment for this hand |
|---|---|---|
| Reduced coordinates, `q_full = q0 + Bx` | Exact kinematic dependency by construction; reduced inertia and forces are `B^T M B` and `B^T tau`. | Mathematically suitable for the affine URDF relations. For an exact serial-link model, FK, Jacobians, inertias, limits, and contact forces must all use the same map. |
| MuJoCo `equality/joint` | A solver residual coupling scalar joints with a polynomial. | Compliant solver constraint, not DOF elimination. The existing implementation and one prior 2x stiffness correction already missed the unchanged `0.003 rad` gate; no stiffness search was repeated. |
| Tendon equality or tendon actuator | Equality can couple tendon lengths; a tendon actuator maps scalar effort through moment arms into joint-space forces. | Equality remains solver-enforced. Actuation alone does not impose the mimic coordinate relation. The pinned URDF does not specify a physical tendon route, so inventing one would be a simulation-only mechanism. |
| Rigid body-tree joint removal | A child body with no joint is rigidly attached to its parent. | Exact only for fixed relative transforms; it does not by itself express these non-unit angular ratios through the serial thumb links. |

MuJoCo 3.3.6 describes equality constraints as residuals handled by the constraint solver, and describes joint equality as polynomial coupling. Its actuator model maps scalar actuator force through transmission moment arms to generalized forces. Those are useful compliant-constraint/force mechanisms, but neither is a native exact reduction of the independent follower coordinates. See the official [Computation documentation](https://mujoco.readthedocs.io/en/3.3.6/computation/index.html#equality) and [XML reference](https://mujoco.readthedocs.io/en/3.3.6/XMLreference.html#equality-joint).

The practical exact option is a reduced-coordinate dynamics implementation or an engine/model extension that actually removes dependent coordinates. A user-level wrapper can demonstrate the mathematics, but if it expands followers into full `qpos` for MuJoCo queries and integrates outside `mj_step`, it is not a native MuJoCo reduced model and must be labeled simulation-only. This prototype is that wrapper, not an accepted production transmission.

## Prototype and result

Added `scripts/research/issue46_reduced_coordinate_transmission.py` as a one-active-coordinate, full-X2-context diagnostic. It computes the projected inertia from the compiled model, uses one driver motor capped at the source effort limit, and explicitly expands the two followers from the reduced state before MuJoCo forward/dynamics queries. Each expansion's follower writes are counted and visible in the trace; there are no hidden writes or follower actuators. The integrator is custom semi-implicit Euler on the reduced coordinate, using MuJoCo `mj_forward`, mass, bias, passive, and constraint-force quantities. It does **not** call `mj_step`; all non-selected coordinates are reset to their initialized pose, so this is not full-body dynamic validation.

The single trajectory was source OPEN `-0.10 rad` hold -> source CLOSE `-0.45 rad` -> close hold. No gain, equality stiffness, or solver sweep was run.

| Gate / observation | Result |
|---|---|
| Gravity-loaded OPEN hold, 500 steps / 1 s | **PASS for this wrapper**: target error `0`, final-100 max speed `4.53e-16 rad/s`; driver torque `0.0253162 Nm` balanced projected gravity. This was active torque hold, not passive hold. |
| CLOSE motion, 750 steps | Executed; maximum transient target error `0.0995094 rad`, maximum speed `0.396352 rad/s`. |
| CLOSE_HOLD, 250 steps / 0.5 s | **FAIL**: tail-100 max position error `0.00411278 rad` passed the prototype's `0.010 rad` settle check, but tail-100 max speed `0.0253517 rad/s` exceeded its exploratory `0.020 rad/s` check. At the last sample `q_mcp=-0.4487903 rad`, speed `-0.00756831 rad/s`. |
| Mimic residual | Recorded maximum `0 rad`, below `0.003 rad`, but this is guaranteed by explicit state expansion and is **not an independent dynamic transmission test**. |
| Source joint limits | Maximum reported violation `0 rad`. |
| Follower actuation / expansion | `0` follower actuators; `6,000` explicit follower qpos expansion writes across the recorded steps. This is transparent map evaluation, but the nonzero writes are a reason this is not native MuJoCo reduced-coordinate acceptance. |
| Contact context | No bottle. Maximum penetration `0.000968815 m`, from three adjacent right-hand finger pairs. The probe's 2 mm abort bound was not exceeded; the pose is not collision-free. |
| OPEN after CLOSE / full-hand OPEN-CLOSE-OPEN | **NOT RUN** after close-hold failed. No separate all-joint rollout was started. |
| Bottle contact, grasp, lift | **NOT RUN**. |

The close-hold miss is a failure of this one custom controller/integrator trial's selected settle check, not proof that the affine relation is mathematically infeasible or proof of a hardware transmission defect. The equality gate is tautological here. Because the bounded prototype did not pass, this iteration stops without further parameter search.

## Evidence and reproduction

Evidence is external to the repository:

`C:\Users\HP\Desktop\Robot\reviews\issue-x2-dual-omnihand-20261010\reduced-coordinate-01\`

- `result.json`: status, dimensions, gates, and hashes.
- `reduced_coordinate_trace.jsonl`: per-step driver/follower coordinates, residuals, limits, contacts, and control effort; SHA-256 `a4496a02916886b6847af1a7e1c701ab61077bee812a8825520cfdbbeabcade0`.
- `reduced_coordinate_thumb.mp4`: actual MuJoCo-rendered 76-frame simulation video at 30 fps; SHA-256 `8efa00010071cc166cc58aa4c2b04516c791e6ece583a7037f0b024024dd8ada`.
- `reduced_coordinate_thumb_focus.mp4`: cropped/reframed derivative of those rendered frames for inspection; SHA-256 `e90e8b9fec8a1ebaaa45797f5b2f15e90efd90cee51847393cd7c0970ef9ec62`.
- `reproduction_command.txt`: command arguments recorded by the harness.

The framing is diagnostic, not a morphology review: the camera is not centered tightly on the moving left thumb and the arm partly occludes the hand. No claim of polished visual evidence is made.

Run from the RobotSim checkout with the pinned source checkout available:

```bash
cd /home/lzy18001500226/robotsim-x2-omnihand-experimental-20261007
export AGIBOT_X2_VENDOR_ROOT=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf
/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_reduced_coordinate_transmission.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-x2-dual-omnihand-20261010/reduced-coordinate-01
```

Focused checks:

```bash
/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -m py_compile \
  scripts/research/issue46_reduced_coordinate_transmission.py \
  tests/test_issue46_reduced_coordinate_transmission.py
/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -m unittest \
  tests.test_issue46_reduced_coordinate_transmission -v
git diff --check
```

The two focused unit tests passed. They check the reduced velocity basis and exclusion of unrelated coordinates; they do not independently validate the MuJoCo dynamic model. The actual simulation result is **FAIL** as described above.

## Credential and publication status

Credential-shaped-value scans found no matches in the local repository/evidence text or Windows PowerShell history. No credential value is included here. The previously reported command-error transcript itself cannot be retracted by this task. The maintainer's revocation and authentication restoration are not confirmed, so no authenticated GitHub operation, PR/Issue update, remote inspection, or push was attempted. PR #59 and Issue #46 were left untouched; their live status was not checked.

## Verdict

- Exact affine kinematic dependency: **FEASIBLE IN REDUCED-COORDINATE MATHEMATICS**.
- This custom wrapper's gravity OPEN hold: **PASS**, with active driver torque.
- Close-hold settling: **FAIL** against the wrapper's exploratory velocity criterion.
- Native MuJoCo `mj_step` reduced transmission: **NOT IMPLEMENTED / NOT VALIDATED**.
- SOURCE_FAITHFUL physical transmission or hardware semantics: **NOT ESTABLISHED**; the URDF supplies mimic kinematics, not the physical transmission construction.
- Complete X2 OPEN/CLOSE/OPEN, bottle contact, and lift: **NOT RUN**.
- Further correction/search: **STOPPED** at the first failed close-hold gate.
- GitHub publication: **BLOCKED** pending maintainer credential revocation and authentication restoration.
