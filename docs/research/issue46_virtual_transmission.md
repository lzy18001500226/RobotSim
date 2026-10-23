# Issue #46 Virtual Transmission Prototype

## Abstraction

This prototype is labeled **CONTROL-LEVEL / ACTUATED MIMIC EMULATION**. It does not claim to reproduce the OmniHand's physical internal transmission. The command interface exposes the 10 source active coordinates for each hand. The six mimic followers per hand are internal implementation details and are rejected by the public active-coordinate command API.

For each source relation, follower position and velocity targets are derived as:

```text
q_follower_target = multiplier * q_driver_target + offset
qdot_follower_target = multiplier * qdot_driver_target
```

Each follower has an internal bounded motor. The prototype contains no mimic equality constraints and does not write follower qpos during rollout. Object attachment, mocap carry, follow-hand logic, and object qpos writes are not used.

## Source Mapping

Pinned source: `AgibotTech/agibot_x2_urdf` at `575cc6b988f976c23550e0db85aa1e5475d3652d`.

| Follower | Driver | Multiplier | Offset |
| --- | --- | ---: | ---: |
| `L_thumb_pip_joint` | `L_thumb_mcp_joint` | 1.33 | 0 |
| `L_thumb_dip_joint` | `L_thumb_mcp_joint` | 1.30 | 0 |
| `L_index_dip_joint` | `L_index_pip_joint` | 1.097 | 0 |
| `L_middle_dip_joint` | `L_middle_pip_joint` | 1.097 | 0 |
| `L_ring_dip_joint` | `L_ring_pip_joint` | 1.097 | 0 |
| `L_pinky_dip_joint` | `L_pinky_pip_joint` | 1.097 | 0 |
| `R_thumb_pip_joint` | `R_thumb_mcp_joint` | 1.33 | 0 |
| `R_thumb_dip_joint` | `R_thumb_mcp_joint` | 1.30 | 0 |
| `R_index_dip_joint` | `R_index_pip_joint` | 1.097 | 0 |
| `R_middle_dip_joint` | `R_middle_pip_joint` | 1.097 | 0 |
| `R_ring_dip_joint` | `R_ring_pip_joint` | 1.097 | 0 |
| `R_pinky_dip_joint` | `R_pinky_pip_joint` | 1.097 | 0 |

The active coordinates are the 10 non-mimic revolute joints per hand in the pinned URDF. The runtime builds 12 internal follower motors across both hands. Model dimensions are `nq=70`, `nv=69`, `nu=63`, `neq=0` at MuJoCo 3.3.6.

## Simulation-Derived Profile

The current test profile uses `kp=0.02 N*m/rad`, `kv=0.0006 N*m*s/rad`, and a follower torque bound of `min(0.05 N*m, source effort limit)`. Active finger servos use the same gains and a 0.05 N*m maximum. These are simulation-derived values, not vendor parameters. No gain or force tuning was performed after the failed run.

## Recovery Result

The previously completed 13-trial kinematic/actuated-emulation comparison is preserved separately under `transmission-model/20261006-passive-transmission/actuated-emulation/`. It completed all requested steps but failed its 0.010 rad diagnostic in the comparison model; it is not evidence for this virtual-transmission controller.

The first virtual-transmission prevalidation stage was restarted from a clean model reset. `open_close_both_hands` failed at step 1 (`t=0.002 s`) on source joint limits. Twelve joints were outside their source range after the first step; for example, `L_thumb_mcp_joint` moved `0.0002891 rad` above its upper limit of zero, while several DIP followers moved approximately `0.00034-0.00037 rad` below their zero lower limits. The initial configuration had zero self-contact and zero penetration. At the failed step there were no contacts, no NaNs, and no velocity-limit violations. The maximum source mimic relation residual was `0.0005868 rad`; maximum follower target-tracking error was `0.0006269 rad`. Maximum absolute qacc was `205.9166 rad/s^2`.

The first failure is therefore the joint-limit gate during the initial open hold, before any bottle contact. The later driver perturbation, gravity, arm-motion, and fixed-object contact gates were not run. The bottle checkpoint was not run. No manipulation video or contact imagery was produced.

## Evidence And Reproduction

Raw result, trace, and log are under:

`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/prevalidation-restart-01/`

The result and log have SHA-256 `942f1d47b71ec574898ba836411f27be7ca5107f153151b05f097d22fb2ce90f`. The one-step trace has SHA-256 `936865558012962cd5cf09efb47db89c0d6a5fea29ce3c8d01c8884d7802c641`.

```bash
wsl.exe -d Ubuntu-22.04 -- bash -lc "env AGIBOT_X2_VENDOR_ROOT=/tmp/robotsim-issue46-virtual-transmission-vendor-20261007 ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/prevalidation-restart-01 PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -u /tmp/robotsim-issue46-virtual-transmission-resume-20261007/scripts/research/issue46_virtual_transmission_gate.py > /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/virtual-transmission-20261007/prevalidation-restart-01/run.log 2>&1"
```

Stop condition reached: `FAIL — source joint limits during open/close step 1`. No further simulation stage was run. PR #53 must remain Draft; do not merge.
