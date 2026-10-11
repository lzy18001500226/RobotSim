# Issue #46: OmniPicker Measured Preload Before Lift

## Result

**FAIL — the bounded preload controller reached the 5 mm bottle-slip stop before establishing the requested normal force. No lift was commanded.** The fixed-wrist bilateral HOLD gate remains PASS. This result applies to the tested isolated fixture, contact pose, and controller; it does not establish global OmniPicker infeasibility or source-faithful hardware performance.

## Baseline and one correction

The comparison baseline is RobotSim HEAD `a85d5e795a44dd97a5c60e08725ef9619d2fccdf`. Its one-second HOLD had 100% bilateral contact, with mean narrow/wide pad normal forces of `0.433465/0.391212 N`. The earlier load-transfer attempt produced only `1.412 N` peak upward hand support, `0 mm` lift, and approximately `50 mm` slip.

One simulation-only correction was added: a bounded, slow driver-reference ramp with force feedback from the compiled pad contacts. The follower target, velocity, and acceleration are derived from the source ratio (`q_follower = -q_driver`). The normal-force target is `1.25 * m*g/(2*mu)`, using runtime effective `mu=1.4`, giving `2.496295 N` per side. Reference speed and acceleration are capped at 10% of the already validated close trajectory. The controller stops on slip above `5 mm`, source-limit violation, unexpected contact, penetration above the existing `2 mm` cap, instability, or a saturation plateau. No friction, mass, source geometry, joint range, mounting, gain, effort cap, or exclusion was changed.

| Measure | Baseline at `a85d5e7` | Bounded preload run |
|---|---:|---:|
| HOLD bilateral contact fraction | `100%` | `100%` before and during preload |
| HOLD mean normal force, narrow / wide | `0.4335 / 0.3912 N` | same HOLD endpoint |
| Required ideal force per side, no margin | `1.9970 N` | `1.9970 N` |
| Preload target per side, 25% margin | not used | `2.4963 N` |
| Peak normal force before safe stop, narrow / wide | not applicable | `1.5074 / 1.4571 N` |
| Force at first slip stop, narrow / wide | not applicable | `1.4831 / 1.4214 N` |
| Maximum ideal friction-capacity upper bound | not measured in baseline | `4.0822 N`, below `5.5917 N` bottle weight |
| Peak applied torque, driver / follower | HOLD `0.06164 / 0.04548 Nm` | `0.14277 / 0.13935 Nm` |
| Effort cap, driver / follower | `0.22499 / 0.22508 Nm` | unchanged; zero saturation |
| Maximum bottle lift | `0 mm` | `0 mm`; lift not run |

The force-capacity value is a Coulomb-friction upper bound computed as `mu*(N_narrow + N_wide)`. It is not measured vertical support. At the stop, measured upward hand support was only `0.00851 N`, while the table supported `5.57868 N`; the bottle remained table-supported with bottom separation `-0.595 mm`.

## First limiting event

At simulation step `4288`, `t=8.576 s` (`1.576 s` into preload), relative bottle-to-hand slip crossed `5.003 mm`, so the safety stop fired. At that sample:

- The driver target had advanced from `-0.327497` to `-0.292927 rad`; the follower target was `+0.292927 rad`. The operational closed targets remained `-0.001/+0.001 rad`, leaving substantial legal closure travel.
- Actual driver/follower positions were `-0.307262/+0.306785 rad`; target errors were `+14.335/-13.858 mrad`. Both were inside the existing `20 mrad` tracking bound.
- Requested and applied torques were `+0.140407/-0.134998 Nm`, below the unchanged `0.224987/0.225078 Nm` caps. Neither actuator saturated; bias compensation was negligible.
- Narrow contact (`R_hand_narrow3_Link`) was at `(0.269864, -0.162914, 0.929345) m`, with normal force `1.48309 N`, normal `(-0.8843, 0.4670, 0.0000)`, and `1.028 mm` penetration.
- Wide contact (`R_hand_wide3_Link`) was at `(0.331039, -0.162401, 0.929807) m`, with normal force `1.42136 N`, normal `(0.8818, 0.4716, 0.0000)`, and `0.723 mm` penetration.
- The contact normals were about `124 degrees` apart, rather than opposing. Their measured force-on-bottle vectors summed to approximately `(0.0603, -1.1311, 0.0085) N`: the contact geometry drove lateral bottle motion while preload was increasing. Maximum penetration remained `1.0285 mm`, under the pre-existing `2 mm` limit.

Thus the measured first failure is **lateral escape from this fixed contact geometry before adequate preload**, not source-limit exhaustion, actuator saturation, or an excessive penetration gate. At the largest observed normals, the ideal friction-capacity upper bound was still below bottle weight. The run does not prove that a different contact location or fixture geometry could not work.

## Integrity and scope

- Static corridor and moving-fixture no-contact regression: PASS. They were reused only after exact runtime, native-library, source, generated-model, fixture-pose, and model-inventory identity checks against the prior passing packet.
- Bilateral HOLD: PASS for one second; preload: FAIL; 50 mm lift: NOT RUN; full-X2 integration: NOT RUN.
- Zero source-limit violations, unexpected contact bodies, NaN state, bottle qpos writes, bottle attachment, mocap carry, or hidden support. No object force was applied outside MuJoCo contact dynamics.
- The fixture/controller is SIMULATION_ONLY and remains subject to the 11 historical SIMULATION_ONLY collision exclusions. No source-faithful acceptance is claimed.

## Reproduction and provenance

The exact executed command is recorded in the evidence packet's `REPORT.md` and `P0_ISOLATED_RESULT.json`. It reused the passing static/moving preflight and ran a clean physical reset for contact and preload:

```bash
cd /home/lzy18001500226/robotsim-issue46-omnipicker-p0-20261011
PYTHONPATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib/python3.10/site-packages:/tmp/robotsim-issue46-omnipicker-p0-pin:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages \
LD_LIBRARY_PATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib:/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib64 \
MUJOCO_GL=egl python3 scripts/research/issue46_omnipicker_p0_isolated.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-p0-measured-preload-20261011-run05-safe \
  --reuse-preflight /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-p0-measured-preload-20261011-run05
```

- Evidence directory: `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-p0-measured-preload-20261011-run05-safe\`.
- RobotSim source branch/head used for the run: `research/issue46-omnipicker-p0-automation-20261011` / `a85d5e795a44dd97a5c60e08725ef9619d2fccdf`; the reproduced runner version is identified by its SHA-256 below.
- AgiBot source: `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`, `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- MuJoCo Python/native `3.3.6`, Python `3.10.12`, timestep `0.002 s`; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Runner SHA-256 `5182bb51b68b815807eca98adea2764d4b57fa362a7abf5f63f08c97c10a3fe9`; generated subtree URDF SHA-256 `20d5221580c003e8d05e3f35985a75a61c25f8f81483b82f7b41a878c9c82786`.
- Review video: `C_preload_feedback.mp4`; per-step contact and control evidence: `C_PRELOAD_TRACE.jsonl`; machine-readable summary: `C_RESULT.json` and `P0_ISOLATED_RESULT.json`.
- SHA-256 of the preserved key evidence files:
  - `C_RESULT.json`: `34c9f43f2cf5b39b8e61c0a01b7404073d84c83c6d9637cc0487001d40787407`
  - `P0_ISOLATED_RESULT.json`: `3b3536b39dce8fe879155a30bcaf95e09a0029046ae11d6b725d8681f248d9b1`
  - `C_PRELOAD_TRACE.jsonl`: `17f1f28fdf75de54f240c947f20f711470ef326f3e733cbc970a548242512918`
  - `C_preload_feedback.mp4`: `645846a4262d3e67810b5b5ac3571a8f0ba45fc8aae2c6ce92965f7f24727399`
  - `PRELOAD_RESPONSE.png`: `d539d7793b2cb634f75e1ec7e5b89ac990e74691732ab82a077f3a48bdaa6afa`

## Recommended next diagnostic

Do not lift this free-bottle setup and do not broaden controller tuning. The smallest useful follow-up is a separately named **SIMULATION_ONLY fixed-object pad-force diagnostic**: constrain the same cylinder only in that diagnostic fixture, retain the original jaw collision meshes, joints, controller caps, and contact parameters, then measure normal force versus commanded closure before either the `2 mm` penetration or source-limit bound. This would distinguish force-generation/contact-solver limits from the observed free-body lateral escape. It would not count as a physical grasp, load transfer, or lift.
