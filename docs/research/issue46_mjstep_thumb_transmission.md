# Issue #46: Genuine MuJoCo Thumb Transmission

**Result: FAIL at the minimal gravity-loaded thumb gate.** The native MuJoCo equality/driver prototype called `mj_step`, but did not maintain the source mimic relations within `0.003 rad`, did not preserve the source joint range during the rollout, and did not track its commands. The full-X2 stage was not run.

## Scope and provenance

- Branch: `codex/x2-omnihand-experimental-20261007`.
- Starting HEAD: `ea7383d414835c5af6108c94f96309b7366f744e`.
- Official source: `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`.
- URDF: `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf`.
- URDF SHA-256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.
- Source checkout was clean at the pinned revision. No upstream file was modified.
- Runtime: Python `3.10.12`, MuJoCo `3.3.6`.

The task's X2 source revision is verified from the local research checkout. It is not a pin in `third_party/LOCK.md`.

## Representation choice

| Candidate | Mechanics | Decision |
|---|---|---|
| MuJoCo joint equality plus independent driver motor | Solver-enforced polynomial relation; compliant constraint, not exact coordinate elimination. | Selected for the one bounded prototype because it preserves the imported source joints and uses native MuJoCo stepping. |
| Tendon/coupling | Tendon force maps through moment arms; tendon equality remains a solver constraint. The URDF supplies no tendon path or pulley geometry. | Not selected; inventing routing would add unsupported assumptions without making the relation exact. |
| True reduced coordinates | Could encode the affine relation exactly by eliminating dependent coordinates. | Not implemented: the prior custom state-expansion integrator is explicitly rejected, and this URDF-to-MuJoCo path does not provide an exact reduced-coordinate `mj_step` model. |

MuJoCo documents joint equality as a solver constraint and describes actuator force as mapping through transmission moment arms; these are not equivalent to eliminating dependent coordinates. See the official [MuJoCo computation documentation](https://mujoco.readthedocs.io/en/3.3.6/computation/index.html#equality) and [joint equality reference](https://mujoco.readthedocs.io/en/3.3.6/XMLreference.html#equality-joint).

## Fixture and controller

The disposable diagnostic URDF contains the pinned right palm and complete right-thumb chain, including the source visual meshes, collision geoms, inertials, axes, ranges, and mimic tags. The compiled model has `nq=5`, `nv=5`, `nu=1`, `neq=4`, `ngeom=13`, and `nmesh=6`. The four active equalities are the two source MCP-to-PIP/DIP mimic relations plus fixture locks for the unused roll and abduction coordinates. Only `R_thumb_mcp_joint` has an actuator; neither follower has an actuator.

Source relation and ranges:

| Joint | Source coordinate | Source range (rad) | Axis |
|---|---|---:|---|
| `R_thumb_mcp_joint` | Driver `q` | `[0, 0.8552113335]` | `[1, 0, 0]` |
| `R_thumb_pip_joint` | `1.33 q` | `[0, 1.1344640138]` | `[1, 0, 0]` |
| `R_thumb_dip_joint` | `1.30 q` | `[0, 1.1344640138]` | `[1, 0, 0]` |

The source roll and abduction joints remain in the fixture with their original limits and axes; they are solver-locked at their reset references only to isolate the transmission. This is a fixture constraint, not a claim about hardware.

The controller is a declared SIMULATION_ONLY projected-gravity feedforward plus PD motor command (`kp=0.04 Nm/rad`, `kd=0.004 Nm s/rad`) with a `0.08 Nm` driver torque cap. The cap is below the source URDF effort limit `33.5 Nm`; it is an experiment bound, not a hardware motor specification. The equality solver remained at the existing direct setting `[-10000, -200]`; source hand joint-limit solver values were set to the same repository baseline. Timestep was `0.002 s`, integrator `implicitfast`, Newton solver, 100 iterations, tolerance `1e-8`, gravity `[0, 0, -9.81] m/s^2`.

During each controller call, the harness snapshots `qpos` and `qvel`, writes only `data.ctrl`, and asserts the state arrays were not changed. State advancement is exclusively through `mujoco.mj_step`; no follower state writes or equality disabling occur.

## Dynamic result

Primary result is the corrected `2 ms` run in `mjstep-thumb-04`. Diagnostic settling limits were declared before the run: over the final `0.2 s` of each hold, position error `<=0.01 rad` and speed `<=0.02 rad/s`; motion tracking error `<=0.08 rad`. These are experiment diagnostics, not pre-existing acceptance thresholds.

| Gate / measurement | Result |
|---|---|
| Initial OPEN hold | **FAIL**. MCP settled at `-0.011918 rad` against `+0.08 rad` (error `-0.091918 rad`); this is outside its source range. |
| First source-range crossing | At step `1`, `t=0.002 s`: MCP/PIP/DIP coordinates crossed below their source lower limits. The initial source-neutral state is at the lower limit. |
| First fixture-lock residual above `0.003 rad` | Abduction at step `9`, `t=0.018 s`, `-0.003104 rad`. |
| First source mimic residual above `0.003 rad` | PIP at step `15`, `t=0.030 s`, `+0.003071726 rad`; the corresponding MuJoCo EFC residual is also `+0.003071726 rad`. |
| Maximum source mimic residual | `0.009173972 rad` (PIP); DIP maximum `0.007867004 rad`. |
| Maximum unused-axis lock residual | `0.008248710 rad` (roll). |
| CLOSE motion / hold | **FAIL**. Maximum driver tracking error `0.407762 rad`; close-hold tail error `0.407686 rad`. The thumb did not reach the close target. |
| Return OPEN | **FAIL**. Final hold error `0.091918 rad`. |
| Driver effort | Peak `0.016523 Nm`, below the diagnostic `0.08 Nm` cap. |
| Limits / finite state | Finite state passed; source-range gate failed by as much as `0.011918 rad`. |
| Contact | No bottle. Maximum observed penetration was `0.001308 m`; this fixture did not establish a contact/grasp result. |

One targeted timestep check at `0.001 s` did not change the conclusion: maximum mimic residual was `0.009173973 rad`, lock residual `0.008248710 rad`, and maximum range violation `0.011918169 rad`. The first mimic threshold crossing shifted only from `0.030 s` to `0.031 s`. Equality stiffness and controller gains were not swept.

The evidence demonstrates failure of this selected native-equality/driver-only prototype with the measured source and solver settings. It does not establish that all MuJoCo transmission formulations or X2 hardware are impossible. Because the minimal gate failed, no full-X2 rollout was run.

## Evidence and reproduction

Primary `2 ms` evidence directory:

`C:\Users\HP\Desktop\Robot\reviews\issue-x2-dual-omnihand-20261010\mjstep-thumb-04\`

- `thumb_result.json`: dimensions, options, gates, metrics, and artifact hashes.
- `thumb_trace.jsonl`: 2,000-step record of targets, source positions/velocities/accelerations, bias, actuator and constraint generalized forces, source mimic residuals, actual MuJoCo EFC residual rows, contacts, and limits. SHA-256: `7e4be2f75597605407671cafaebaadb3a02efaf7cc5ed1155c786a31eccda3b7`.
- `thumb_mjstep.mp4`: 200 frames, H.264, 640x480, 30 fps, rendered from MuJoCo states advanced by `mj_step`. SHA-256: `e9c141e869debff28488beb40b932fd439facb3d98eb806f9ad901a190e6c413`.
- `official_right_thumb_fixture.urdf`: generated disposable fixture URDF; source mesh/inertial references retained.

The single half-timestep comparison is preserved separately under `mjstep-thumb-05`; its trace SHA-256 is `2cb20e34a6e230278fc5774b5dee1876359121fd8ec3ec82049300ffe984b970`, and video SHA-256 is `aa2d1ff398046fe7e8d06913d86c73eaa07e929937a1302fccd69fc2072c7dab`.

Reproduce the primary run from the RobotSim checkout:

```bash
cd /home/lzy18001500226/robotsim-x2-omnihand-experimental-20261007
export AGIBOT_X2_VENDOR_ROOT=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf
export MUJOCO_GL=egl
/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_mjstep_thumb_transmission.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-x2-dual-omnihand-20261010/mjstep-thumb-04
```

The conditioning comparison adds `--timestep 0.001` and changes the output directory to `.../mjstep-thumb-05`.

## Validation and publication

- Python compilation: PASS.
- Focused unit tests: 2 passed. They check source model dimensions/actuation, unchanged equality/limit settings, controller state-write exclusion, and that `mj_step` advances the model.
- Actual MuJoCo rollout: **FAIL**, as detailed above.
- `git diff --check`: PASS.
- Full repository suite and full-X2 validation: NOT RUN; the minimal dynamic gate failed first.
- No authenticated GitHub operation, remote query, PR update, or push was attempted. Credential revocation and reauthentication remain unconfirmed, so publication/PR status is **BLOCKED**. No implementation from the prior custom integrator was changed.

## Verdict

- Representation selected: native MuJoCo `mjEQ_JOINT` constraints with one independent driver motor.
- Genuine MuJoCo stepping: **PASS** (`mj_step` advanced the model and generated the preserved trace/video).
- Gravity-loaded source mimic gate: **FAIL** (`0.009174 rad` maximum versus the unchanged `0.003 rad` tolerance).
- Source range and commanded OPEN/CLOSE behavior: **FAIL**.
- Full X2 re-integration: **NOT RUN** because the minimal gate failed.
- Source-faithful mechanical transmission or hardware equivalence: **NOT ESTABLISHED**.
- Branch publication and PR update: **BLOCKED** pending credential remediation.
