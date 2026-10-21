# Issue #46 — OmniPicker Functional Simulation Review

Draft PR #84 preserves the source-derived D-point/sign mapping and adds an isolated motor-only Candidate B simulation. Official AgiBot X2 v1.4.0 URDF/STL assets, source STEP, official root axes/limits, and opposite-root mimic remain unchanged.

The per-side Candidate B relation `(q_root - qpos0_root) + q_B - q_E = 0` is used only as an `UNVERIFIED_EXPERIMENTAL` MuJoCo coupling. With one right-root actuator, the corrected source-range model has 8 velocity coordinates, equality rank 7, and 1 effective DOF. In CAD q `[-0.10,0.00]`, a 4.6-second real `mj_step` open/close cycle passed: D gap ≤7.11 μm, coupling residual ≤0.000586 rad, joint speed ≤0.486 rad/s, and passive qpos writes after initialization =0.

Physical object validation did not pass. The measured maximum source-range aperture is 35.939 mm; the 36 mm canonical bottle neck has no clearance and was not tested. A free 18 mm × 20 mm, 30 g cylinder made symmetric jaw contact, but normal force peaked at 0.269/0.273 N per side, below the 0.5 N-per-side lift gate. The jaws' vertical force remained downward, the shelf supported the object, and no lift was attempted. Do not report a grasp or load-transfer PASS.

The maintainer's q=+0.05 visual alignment frame remains outside saved STEP motion samples; it is analytical continuation. The pinned left/right URDF joint frames match after side-name substitution, so no mirror-origin correction was justified. See [the detailed report](issue46_motor_only_omnipicker.md) for the q-frame audit, test traces, video, force breakdown, and exact reproduction commands.

Latest run artifacts and verified path manifest: `reviews/issue-46/candidate-b-test-cylinder-20261021T145844Z/`. The actual motor stroke video is `single_motor_open_close.mp4`; `test_cylinder_contact.mp4` records the failed object-contact/lift attempt. The first default-softness failure and the successful corrected-solref run remain preserved in their own run folders.

**Status:** `SOURCE-RANGE MOTOR STROKE PASS` (SIMULATION_ONLY / UNVERIFIED_EXPERIMENTAL) and `PHYSICAL GRASP/LIFT BLOCKED`. No vendor fidelity or canonical bottle grasp is claimed. PR #84 remains Draft.
