# Issue #46 Thumb First-Step Diagnosis

## Verdict

The first-step wrong-way motion was caused by an infeasible diagnostic fixture reference pose, not by an actuator sign/transmission mapping error. At reset, the pinned palm and thumb MCP collision proxies overlap by 1.3081 mm while the MCP and its mimics are on their source lower limits. The resulting contact constraint force opposes the correctly signed OPEN command and drives the MCP below its source range on the first `mj_step`.

A single fixture-only correction excludes collision between `R_palm` and `R_thumb_mcp`; source geometry, joint limits, equality constraints, actuator, controller and `mj_step` are unchanged. This removes the limit crossing and keeps the mimic residual below 0.003 rad. The original OPEN -> CLOSE -> OPEN test still fails its initial OPEN settling-speed criterion, so the transmission is not accepted as passing.

## Provenance

- Starting local HEAD: `6c9a2d846141c983937dc088a432f32075d71dbb`.
- Pinned source: `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Source URDF SHA-256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.
- MuJoCo: `3.3.6`; fixture dimensions: `nq=5`, `nv=5`, `nu=1`, `neq=4`.
- Both ten-step diagnostic trials used `mj_resetData` from the same zero-qpos, zero-qvel source-reference state. No rollout qpos/qvel writes were used.
- Raw diagnostic JSON: `C:\Users\HP\Desktop\Robot\reviews\issue-x2-dual-omnihand-20261010\first-step-diagnostic-02\first_step_diagnostics.json`; SHA-256 `247a9229edd1c955cf8cba9af23a8c92ed465ea79ef204ee0e6cebb83a2095d5`.

## Source And Compiled Mapping

| Joint | Source range (rad) | Source/compiled axis | Reset qpos (rad) | Mimic |
|---|---:|---|---:|---|
| `R_thumb_roll_joint` | `[-0.1745329252, 0.8726646260]` | `[1,0,0]` | 0 | fixture locked |
| `R_thumb_abad_joint` | `[-1.745329252, 0]` | `[0,0,1]` | 0 | fixture locked |
| `R_thumb_mcp_joint` | `[0, 0.8552113335]` | `[1,0,0]` | 0 | independent driver |
| `R_thumb_pip_joint` | `[0, 1.1344640138]` | `[1,0,0]` | 0 | `1.33 * MCP` |
| `R_thumb_dip_joint` | `[0, 1.1344640138]` | `[1,0,0]` | 0 | `1.30 * MCP` |

The MCP, PIP and DIP reset coordinates are exactly at their source lower boundaries; abad is exactly at its upper boundary. The coordinates satisfy the source intervals, but the colliding palm/MCP geometry makes the complete reference pose dynamically infeasible in this fixture.

The MCP actuator is transmitted to the MCP joint (`trnid` joint id 2), with positive unit gear `[1,0,0,0,0,0]`, control and force limits `[-0.08, 0.08]` in the SIMULATION_ONLY fixture. The OPEN command is `+0.08 rad`; its commanded torque is positive (`+0.0032 Nm` without gravity, `+0.00340402 Nm` with gravity). CLOSE uses the opposite direction. No sign inversion was found. Source effort limit is 33.5 Nm; the fixture's smaller torque cap is unchanged.

The active source mimic equalities map PIP with coefficient `1.33` and DIP with `1.30`, zero reference offsets, and existing direct `solref` `[-10000,-200]`. No equality-stiffness change was made.

## Initial `mj_forward` And First Ten Steps

Before stepping, `mj_forward` found two palm-to-thumb-MCP contacts at distance `-0.001308099962 m`. There were no joint-limit constraint rows yet; the joints were at the inclusive source boundaries. The configured OPEN torque had the intended positive sign, but the contact/equality-constrained solve produced negative MCP acceleration. These are coupled generalized forces, so their scalar components should not be naively subtracted to infer acceleration:

| Quantity at commanded t=0 | Gravity off | Gravity on |
|---|---:|---:|
| MCP actuator force | `+0.00320000 Nm` | `+0.00340402 Nm` |
| MCP bias force | `0` | `-0.00021188 Nm` |
| MCP constraint force | `-0.00253405 Nm` | `-0.00320338 Nm` |
| of which contact | `-0.00276860 Nm` | `-0.00318712 Nm` |
| MCP acceleration | `-23.13220 rad/s^2` | `-21.27748 rad/s^2` |
| MCP change after first step | `-0.00009253 rad` | `-0.00008511 rad` |
| MCP change after 10 steps | `-0.00272298 rad` | `-0.00264909 rad` |

Both trials used timestep `0.002 s`, unchanged solver/equality configuration, the same reset state and ten `mj_step` calls. Gravity was the only trial condition changed. The full JSON records all five joint positions, velocities, accelerations, actuator/bias/constraint forces, active EFC rows, contacts and mimic residuals for the initial state and every step. The residuals began at zero and remained below `0.0021 rad` during these ten steps. Joint-limit force rows appeared only after MCP crossed below zero.

## Correction And Acceptance Replay

The minimal change is a body-pair collision exclusion between `R_palm` and `R_thumb_mcp`, enabled only for the minimal diagnostic fixture. It does not alter either body's geometry or the full-X2 model. The focused regression asserts that the unexcluded reference pose has exactly this contact pair and that the corrected pose has no initial contacts. MuJoCo body-pair exclusions apply to collisions between geoms on the named bodies ([MuJoCo 3.3.6 contact exclusion reference](https://mujoco.readthedocs.io/en/3.3.6/XMLreference.html#contact-exclude)).

The original gravity-loaded OPEN -> CLOSE -> OPEN replay was run once after the correction: 2,000 steps over 4 s. Overall result is **FAIL**, with one remaining gate:

- Initial OPEN hold: target error `0.003715 rad` (within diagnostic `0.01`), maximum speed `0.036054 rad/s` (above diagnostic `0.02 rad/s`); failed.
- CLOSE hold: PASS; max error `0.0000571 rad`, max speed `0.0008722 rad/s`.
- Return OPEN hold: PASS; max error `0.0002183 rad`, max speed `0.0041479 rad/s`.
- Source joint limits: PASS, maximum violation `0 rad`.
- Mimic residual: PASS, maximum `0.00205295 rad < 0.003 rad`.
- Motion tracking: PASS against the existing diagnostic bound; max error `0.0491835 rad < 0.08 rad`.
- Actuator effort: PASS, peak `0.00462753 Nm < 0.08 Nm`.
- Max penetration: `0 m`; gravity remained enabled.

These OPEN-hold thresholds are explicitly diagnostic criteria, not an established source/hardware acceptance specification. No second correction, gain change, timestep sweep, equality sweep, full-X2 run or bottle trial was performed.

Corrected run artifacts:

- `C:\Users\HP\Desktop\Robot\reviews\issue-x2-dual-omnihand-20261010\mjstep-thumb-06\thumb_result.json`; SHA-256 `b004ec462e6a18dc9ea19149b647cf4fcdf6f6af9ddba0728b66ec199bddffba`.
- `thumb_trace.jsonl` SHA-256 `e9c44bbfebedc62aae5d05f074967fc575b795296952667a260cdddabfaedfc5`.
- `thumb_mjstep.mp4` SHA-256 `90f008813d423747e17ddf1f455a50129057babddd8c22e08d6fb8256264e26f`.

Exact commands:

```bash
cd /home/lzy18001500226/robotsim-x2-omnihand-experimental-20261007
AGIBOT_X2_VENDOR_ROOT=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_thumb_first_step_diagnostic.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-x2-dual-omnihand-20261010/first-step-diagnostic-02

AGIBOT_X2_VENDOR_ROOT=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf \
MUJOCO_GL=egl \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_mjstep_thumb_transmission.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-x2-dual-omnihand-20261010/mjstep-thumb-06
```

The diagnostic script deliberately builds the unexcluded baseline for the two ten-step trials; the acceptance replay uses the corrected fixture default. Both remain local. No authenticated GitHub operation or push was attempted because credential revocation/authentication restoration has not been confirmed.
