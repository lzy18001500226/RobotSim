# Issue #46 Dual OmniHand Transmission Tracking

**Result: BLOCKED at the no-bottle dynamic OPEN gate.** The sole reduced SIMULATION_ONLY transmission prototype exceeded the unchanged `0.003 rad` mimic threshold during the fourth physics step. CLOSE, bottle contact, load transfer, and lift were not run.

## Scope and Provenance

- RobotSim branch: `codex/x2-omnihand-experimental-20261007`; experiment started from `0ae880b846567b9db0a7b955e75483de60353671`.
- Official source: `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`, `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf`.
- URDF SHA-256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.
- Vendor worktree was clean. Runtime: Python `3.10.12`, MuJoCo `3.3.6`, timestep `0.002 s`, implicit-fast integrator.
- Full pinned X2 and both native OmniHands were loaded; all 12 source mimic equality constraints were retained. No bottle was included. No upstream file, hand geometry, inertial, joint limit, or collision geometry was modified.

The historical `physical-probe-02/physics_contact_trace.jsonl` was absent on this machine when the task resumed. The original recorded digest was `6ba5917abbe873426bada5336f44ead561277605f3d611162251e78ddc0a28d7`, but it could not be reverified. The baseline values below are the event values retained from the earlier trace inspection; a complete all-joint event CSV could not be regenerated. Do not treat the historical values as a newly verified raw-trace replay.

## Historical Dynamic Failure

The source/compiled hinge axes aligned (`dot = +1`) and the 12 compiled driver/follower names, ratios, and offsets matched the pinned URDF. The affected thumb relation is `thumb_mcp -> thumb_pip (1.33)` and `thumb_mcp -> thumb_dip (1.30)`; the middle relation is `middle_pip -> middle_dip (1.097)`. The recorded failure is therefore not supported as an axis-sign or mapping defect.

The `0.1411207 rad` OPEN error was on `L_thumb_mcp_joint`, whose commanded OPEN target was the source neutral `-0.100 rad`. It peaked at step 55 with actual position about `-0.241121 rad`; at the end of the 250-step hold it remained at `-0.215403 rad` (error `0.115403 rad`, velocity about `-0.0081 rad/s`). This is persistent static droop, not a bad initial target or a transient that settled. The legacy driver servo was `kp=0.2 Nm/rad`, `kv=0.006 Nms/rad`, without gravity feedforward. Its recorded projected gravity load was `0.025316 Nm`; the linear static deflection estimate `load/kp = 0.1266 rad` is close to the observed hold error. The best-supported cause is the simulated actuator/hold design.

Focused values retained for the right middle-finger path:

| Event | Joint | Target (rad) | Actual (rad) | Velocity (rad/s) | Mimic residual (rad) |
|---|---|---:|---:|---:|---:|
| OPEN, step 1 | `R_middle_pip_joint` | 0.050000 | 0.049939 | -0.030481 | n/a |
| OPEN, step 1 | `R_middle_dip_joint` | 0.054850 | 0.054751 | -0.049298 | 0.000032 |
| OPEN hold end, step 250 | `R_middle_pip_joint` | 0.050000 | 0.050549 | -0.000034 | n/a |
| OPEN hold end, step 250 | `R_middle_dip_joint` | 0.054850 | 0.055596 | -0.000048 | 0.000143 |
| Preshape start, step 251 | `R_middle_pip_joint` | 0.050495 | 0.050539 | -0.005068 | n/a |
| Preshape start, step 251 | `R_middle_dip_joint` | 0.055393 | 0.055733 | 0.068415 | 0.000291 |
| First bottle contact, step 678 / 1.356 s | `R_middle_pip_joint` | 0.261860 | 0.255391 | 0.253917 | n/a |
| First bottle contact, step 678 / 1.356 s | `R_middle_dip_joint` | 0.287260 | 0.280724 | 0.509126 | 0.000560 |
| First mimic violation, step 725 / 1.450 s | `R_middle_pip_joint` | 0.285125 | 0.286995 | 0.331000 | n/a |
| First mimic violation, step 725 / 1.450 s | `R_middle_dip_joint` | 0.312782 | 0.318150 | 0.627519 | 0.003317 |

The first recorded right-hand/bottle contact was on `R_middle_dip` at step 678. The mimic threshold was first exceeded 47 steps later, so the historical middle-DIP breach was contact-loaded dynamic coupling, not a pre-contact mapping error. The raw contact-force trace is unavailable, so the relative contributions of contact impulse and equality-constraint compliance cannot be separated further. The full per-joint historical event table is **DEFERRED** because the source trace is missing.

## One Reduced Prototype

The sole prototype used independent torque motors only on source-independent driver joints, with the original 12 MuJoCo joint equalities retained and **zero follower actuators**. It is explicitly SIMULATION_ONLY, not a claim about the physical OmniHand transmission. Driver torque used projected gravity compensation plus an inertia-derived critically damped PD term (`omega_n = 8 rad/s`), with a shared torque cap of `0.076351 Nm`. No follower qpos was written during rollout. Source limits were unchanged.

To avoid the clearly excessive ring/pinky overlap in the first setup, the right ring OPEN abduction target was changed from `0.10` to the source-valid `0.0 rad`. That removed the 5.79 mm failure but left shallow adjacent-finger contacts up to `0.969 mm` in the next run. Those contacts remain a fixture limitation; they are not claimed collision-free.

The diagnostic trace was extended to record follower target, actual source position, velocity, and mimic residual. With the above pose and unchanged control parameters, the full no-bottle model failed on step 4 in `OPEN_HOLD`:

| Step | `L_thumb_mcp` target / actual / velocity | `L_thumb_pip` expected / actual / velocity / residual | `L_thumb_dip` expected / actual / velocity / residual | Maximum mimic residual |
|---:|---|---|---|---:|
| 1 | -0.100000 / -0.099903 / 0.048484 | -0.133000 / -0.133233 / -0.116699 / 0.000362 | -0.130000 / -0.130322 / -0.160757 / 0.000448 | 0.001068 (`L_ring_dip`) |
| 2 | -0.100000 / -0.099696 / 0.103422 | -0.133000 / -0.133830 / -0.298224 / 0.001234 | -0.130000 / -0.130671 / -0.174943 / 0.001066 | 0.001691 (`L_ring_dip`) |
| 3 | -0.100000 / -0.099575 / 0.060646 | -0.133000 / -0.134601 / -0.385718 / 0.002167 | -0.130000 / -0.131077 / -0.202889 / 0.001630 | 0.002167 (`L_thumb_pip`) |
| 4 | -0.100000 / -0.099539 / 0.018179 | -0.133000 / -0.135478 / -0.438514 / **0.003092** | -0.130000 / -0.131546 / -0.234656 / 0.002146 | **0.003092 (`L_thumb_pip`)** |

At the same failing step, the active right middle DIP residual was `0.000541 rad`; its target/actual/velocity were `0.054850 / 0.053703 rad / -0.304474 rad/s`. The violating relation was on the *other* hand, under gravity and its explicit hold control. The model's compiled equality uses direct `solref=[-10000, -200]` at scale `1` for this run. This points to the compliant MuJoCo equality/driver-load dynamics in this SIMULATION_ONLY realization, not a thumb axis or target-direction error. Since the gate is a maximum-through-motion bound and the unchanged threshold was exceeded before OPEN settled, this prototype **FAILS**; the planned CLOSE and return-to-OPEN were not attempted.

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| Source axis and mimic mapping | PASS | Prior compiled inventory: all hand axes aligned; all 12 names/ratios/offsets matched. Historical raw trace unavailable. |
| Historical OPEN target | FAIL | Persistent `L_thumb_mcp` droop; `0.141121 rad` peak and `0.115403 rad` at hold end. |
| Historical contact-loaded mimic | FAIL | `R_middle_dip` first exceeded `0.003 rad` after first bottle contact. |
| SIMULATION_ONLY no-bottle OPEN/CLOSE/OPEN | FAIL | Step 4 OPEN mimic residual `0.003092 rad`; no CLOSE reached. |
| Joint-limit gate | PASS through recorded steps | Maximum violation `0 rad` in the 4-step run. |
| Penetration gate | PASS only against this probe's 2 mm abort bound | Maximum `0.969 mm`; shallow adjacent finger contacts remain and are disclosed. |
| Bottle contact / support / lift | NOT RUN | Phase 3 did not pass. |
| Stable settled tracking | NOT RUN | Failure occurred after 8 ms, before any hold window could settle. |

No further transmission model, solver setting, gain, or contact experiment was tested. The next step is to stop and review the equality-compliance/actuation representation; another model requires a separate bounded authorization. The source-faithful transmission is **NOT ESTABLISHED** and no physical support claim is made.

## Reproduction and Artifacts

The run-07 evidence is external to the repository at:

`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-x2-dual-omnihand-20261010/transmission-motion-07/`

It contains the 4-step JSONL trace, result JSON, one-frame diagnostic MP4, and command record. The MP4 is not an OPEN/CLOSE motion demonstration because the simulation aborted before movement.

Exact command:

```bash
cd /home/lzy18001500226/robotsim-x2-omnihand-experimental-20261007
AGIBOT_X2_VENDOR_ROOT=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf \
/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
scripts/research/issue46_dual_omnihand_transmission_probe.py \
--output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-x2-dual-omnihand-20261010/transmission-motion-07
```

Run-07 trace SHA-256: `e4b540daa522ccd1f71f32014851891df83513c17215f749a824deb28500f49b`.
Run-07 video SHA-256: `74f894df5803fa7bf5aa78244e5c32bf75e77b16dceb5742b8e83e3f8b2be850`.
Probe SHA-256 at run time: `0475888c3ca1c49c43e121f12a72ff676298e6ee8d94482c729326f4019d7229` (the later command-recording fix changes the working-tree file hash).

## Validation

- L0: `python -m py_compile scripts/research/issue46_dual_omnihand_transmission_probe.py tests/test_issue46_dual_omnihand_transmission.py` — PASS.
- L1: `python -m unittest tests.test_issue46_dual_omnihand_platform tests.test_issue46_dual_omnihand_transmission -v` — PASS, 10 tests.
- L0 whitespace check: `git diff --cached --check` — PASS.
- L2: MuJoCo headless dynamic run — **FAIL**, exact first failure recorded above. This failed gate is the experiment result, not a passing physics validation.
