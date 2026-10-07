# Issue #46 X2 OmniPicker Dynamic Checkpoint

**Result: FAIL - Stage 3 no-contact OPEN HOLD failed at the first physics step.**

The pinned AgiBot source defines one commanded coordinate per picker. For the right side, `right_claw_joint` is the driver with range `[-1, 0]` rad; `R_hand_wide1_joint` follows it with multiplier `-1` and offset `0`, range `[0, 1]` rad. Both axes are `[0, 0, -1]`. The source writes `velocity="0"` and `effort="0"` on both joints. There are no URDF transmissions. The left side has the mirrored name pair and same limits/relation.

The source revision is `AgibotTech/agibot_x2_urdf` commit `575cc6b988f976c23550e0db85aa1e5475d3652d`; URDF SHA256 is `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`. MuJoCo 3.3.6 drops the mimic annotations when importing the URDF. The research runner adds two native joint equalities with coefficients `[0, -1, 0, 0, 0]`, preserves the source position ranges, and exposes only one right-driver actuator (`nu=1`, `neq=2`). No follower actuator or post-initialization follower `qpos` write is used. Runtime was WSL2 Ubuntu 22.04, Python 3.10.12, MuJoCo Python/native 3.3.6, `0.002 s` timestep, EGL; native library SHA256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.

## First Failure

The no-contact OPEN HOLD was run for one 2 ms step. The bottle and table were absent, no contacts occurred, and no object state was written. All four picker coordinates had literal source velocity limit zero, so any nonzero velocity violates that unchanged source value. At this first step the right open pair moved at `0.2659169` and `-0.2660020` rad/s while its target remained at `-1` rad and the actuator force was zero. The right pair stayed within position ranges.

The unused left picker started at its closed endpoint (`0/0` rad) with no actuator. At 2 ms its driver was `+2.3195e-5` rad, above its source upper bound `0`, and its follower was `-1.9747e-5` rad, below its source lower bound `0`. The equality residual was `3.4478e-6` rad. No NaN occurred. The one-step trace therefore fails both literal source velocity and left-side source position limits before CLOSE begins.

A bounded static `mj_forward` audit at the same initial state (zero `mj_step` calls) shows nonzero `qfrc_bias`, zero actuator torque at the exact initial targets, and only small equality forces. This supports gravity/bias loading of the hard-stop initialized jaws; the mimic relation itself remains close at the first step. It is not evidence of a complete dynamic synchronization test.

## Stage Status

- Stage 1 source audit: PASS.
- Stage 2 native equality construction: PASS as a compiled representation, not as a full motion validation.
- Stage 3 no-contact open/close: FAIL at OPEN HOLD, step 1 / 2 ms. The close/reopen sequence was not run.
- Stage 4 fixed-object contact: NOT RUN.
- Stage 5 bottle hold: NOT RUN.
- Stage 6 lift: NOT RUN.

No MP4 was produced because motion did not pass the no-contact gate. The accepted static overview, front, side, and right-picker close-up PNGs are copied into the evidence directory; they are unchanged copies of the maintainer-approved static packet.

## Simulation-Only Values And Limitations

The driver servo is an internal diagnostic actuator, not a vendor actuator model. Its `kp=5 Nm/rad`, `kv=0.1 Nm*s/rad`, and `+/-0.5 Nm` force bound are simulation-derived. The source's zero effort value does not provide a positive hardware effort rating. No gain or coupling sweep was run. The full close/reopen trajectory, fixed-object contact, bottle HOLD, and lift remain unvalidated.

The first blocker is an authority question: confirm whether the literal zero velocity fields are hard source limits or placeholders for unspecified limits, and provide an authoritative positive velocity/effort rating if motion is intended. RobotSim must not replace them silently. Separately, the unused left picker needs a valid initial hold configuration/controller before the no-contact gate can pass; the current static closed-at-zero pose has no actuator support under gravity.

The accepted static morphology was not changed. The bottle/table/floor and robot scene definitions were not changed. No OmniHand work was reopened.

## Reproduction And Evidence

Exact dynamic command is in the external `experiment_command.txt`; the source audit and compiled model identity are in `result.json`, and the one-step state is in `step_trace.jsonl`. All four compiled position ranges matched the source values in `compiled_limit_audit.json`. The static force audit is in `initial_force_audit.json` with its exact command in `initial_force_audit_command.txt`.

Evidence directory (WSL): `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-dynamic/20261007-no-contact-first-failure-v2/`

Evidence directory (Windows): `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-dynamic\20261007-no-contact-first-failure-v2\`

Research runner disposition: `KEEP-AS-TOOL` for this pinned-source, offline diagnostic only. It is not imported by production runtime code.
