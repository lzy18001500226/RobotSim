# Issue #23: Free-Base G1 Locomotion Prototype Evidence

Status: `PARTIAL / NOT YET REUSABLE FOR THE FULL REQUESTED SEQUENCE`.

This report-only task branch is based on latest `main` at `bbf2090b0ccb7b50190dac33d7b04997361c9fd5`. No production, manipulation, PR #45, or upstream vendor source was modified.

## Pins

- Upstream `unitreerobotics/unitree_rl_gym`: `276801e46c5d433564f24658bac64f254b7d2d4b`, clean detached checkout.
- G1 policy `motion.pt`: SHA-256 `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`.
- Scene SHA-256: `482d49902ca2b9fdc49d84ef5d8779fa69d66bc53ef440b4bf50079d1bac9995`.
- Model SHA-256: `747ede40aa726b7352bae8353e95d0d0f908cec2257a27cbd78bc6e5a2d5a314`.
- Config SHA-256: `73044e7d355c61915695c16d6e09eb3efef46eec1e3d708fd3eb9157dfe3bbbb`.
- Runtime: Python 3.10.12, MuJoCo 3.2.3, PyTorch 2.3.1+cpu, NumPy 1.26.4; local `robotsim:humble-dev` image `sha256:5dae0a80a15aa111b122ae3d0ad5a4022384f7b0bfcb84ddc6dd406e0f37d1f9`.

## Model and Support

Compiled `nq/nv/nu=19/18/12`; 12 leg actuators, no sensors. It has one free root joint, zero equality constraints, zero mocap bodies, and zero applied generalized/body forces throughout the policy run. There was no band, weld, fixed base, or hidden world support. This is a reduced 12-DoF model, not RobotSim's full 29-DoF G1.

## Results

| Gate | Result | Measured result |
| --- | --- | --- |
| Free-base run without external support | PASS | 32 s simulated; no safety abort or fall. At least one foot contacted floor in all samples. |
| Zero-command stationary standing, 10 s | FAIL | XY drift `0.225 m`; planar world speed p50/p95 `0.099/0.217 m/s`; both-foot contact fraction `11.2%`, indicating ongoing gait. |
| Forward `+0.2 m/s`, 3 s | PASS | XY displacement `(+0.540, -0.089) m`; body forward velocity p50 `+0.192 m/s`. |
| Backward `-0.2 m/s`, 3 s | PASS | XY displacement `(-0.391, -0.012) m`; body forward velocity median `-0.166 m/s`. |
| Yaw `+0.4 rad/s`, 3 s | PASS | Yaw change `+1.103 rad`; yaw-rate median `+0.369 rad/s`. |
| Stop and settle, 8 s | FAIL | XY drift `0.186 m`; body planar speed p50/p95 `0.096/0.215 m/s`; cyclic motion remained. |
| Static PD at upstream default pose, separate probe | FAIL | Fell at `1.26 s` (`z=0.490 m`, pitch `1.001 rad`), despite bilateral floor contact. |

The policy run kept pelvis height above `0.767 m`, peak absolute roll/pitch below `0.071/0.059 rad`, and all state/action values finite. Safety abort limits were `z < 0.42 m`, `abs(roll/pitch) > 1.0 rad`, or any non-finite value. Forward, backward, and yaw motion are useful; idle/stop are not stable. No training or gain tuning was attempted.

## Evidence

Raw evidence and the reviewable video remain outside the Git checkout at the local external Issue #23 evidence directory `issue23-g1-locomotion-20261003/evidence/local-rl-gym-20261004/`:

- `g1_12dof_locomotion.mp4`: 640x360 H.264, 321 decoded frames, 10 fps, 32.1 s; SHA-256 `304ea4b153f29892a93756151f7e7ce1727897b52efaa42dccb58f3720236f44`.
- `locomotion-contact-sheet.png`: SHA-256 `f1ca295fe01c38e285b47473b2b4146719433508fa49d276214379b14957ba8e`.
- `timeseries.csv`: SHA-256 `6c89bbcf27eebac2da4ea3e9f51a8891aa5c7fe5cc7c2b831152aef8b3b9291d`.
- `summary.json`, `stand_probe.csv`, `stand_probe.json`, and external harness scripts preserve the remaining run data.

The video was decoded and visually inspected. Capture took `102.9 s` wall time for `32 s` simulation because EGL rendering was enabled; this is not a real-time or performance claim. The complete local `REPORT.md` preserves the exact local commands and additional per-phase data without adding large binaries to this branch.

## Reuse Decision

Use this policy only as an isolated 12-DoF commanded-gait baseline for forward/backward/yaw experiments. It is not yet reusable for the requested full stand-to-walk-stop sequence and cannot be directly applied to the RobotSim 29-DoF model. Next, validate an unsupported idle/stand mode compatible with the selected model, then test both stand/policy transitions before architecture integration.
