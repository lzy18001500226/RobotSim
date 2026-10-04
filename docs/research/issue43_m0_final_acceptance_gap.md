# Issue #43 M0 final acceptance gap review

Reviewed the Issue #43 branch at `ddcab12057fcd84d64b979d49d91771302668280` (merge commit; includes M0 implementation commit `517c1c2`). This is a read-only implementation audit. No simulation was run for this review.

## Acceptance matrix

| Check | Status | Current evidence and remaining gap |
|---|---|---|
| Actual bottle/cylinder object | **PARTIAL** | The body is a free joint, but the main 0.20 m body is a box (`box_geom`, half-sizes 0.100/0.075/0.100 m); its shoulder is also a box. Only the neck and cap are cylinders. This is a bottle-colored composite, not a cylindrical bottle body. See `OBJECT_GEOM_SPECS` in `simulation/mujoco/m0_pick_place.py`. |
| Physical grasp | **PARTIAL** | Bilateral palm contact and force (2 N per palm for 5 samples) are measured. Once that gate passes, `_engage_grasp_weld` activates a MuJoCo weld from the right wrist to the object; the left hand retreats. Thus the contact gate is physical, but the continuing grasp is an equality constraint rather than a friction/contact grasp. |
| Lift | **PASS** | The monitor requires the free-body center to rise at least 5 cm after the bilateral grasp gate. `M0_STATUS.md` reports a 7.6 cm lift and a passing LIFT stage. The cited run JSON is not available here for independent confirmation. |
| Transfer | **PARTIAL** | The object reaches the target footprint while the runtime weld is active. This demonstrates a simulated constrained transfer, not transfer held by hand contact/friction. |
| Release | **PASS** | The weld is disabled before retreat/settling. RELEASE requires five control samples with no palm contact and no active weld; subsequent motion is stepped in MuJoCo. |
| Target-region placement | **PARTIAL** | The predicate checks only the object center against table half-extents minus 3 cm; it does not project/check the object's footprint. Reported final center `[0.4528, -0.4876]` m is 0.1528 m from the target-table center in X. The main box has at least 0.075 m projected half-width in X, so its X edge is at least 0.5278 m while the tabletop ends at X=0.5000 m: at least 2.78 cm overhang. The center-only predicate can pass this. |
| Settle | **PASS** | PLACE requires target-table contact, center in the accepted region, speeds below 0.03 m/s and 0.20 rad/s, and 33 consecutive samples (1.024 s) within a 2 cm position radius. This is a useful free-physics settle check after release. |
| Finite-state checks | **PASS** | Each control observation checks all `qpos`, `qvel`, and `qacc`, plus object position, quaternion, and simulation time for finiteness. A broad `max(abs(state)) <= 1000` bound is also present. Controls/sensors are not included in that state vector. |
| Penetration check | **PARTIAL** | The monitor records contact distance for contacts involving the listed object geoms and fails above 2.5 cm penetration. It samples after each control frame, not every 2 ms physics substep, so a transient worst penetration between observations can be missed. |
| No active-rollout teleport | **PARTIAL** | The monitor rejects an object-center step above 0.20 m between control observations. Code inspection found no direct object-qpos assignment in the active motion sequence; IK helpers temporarily change robot qpos and restore it. The 20 cm/control-frame check does not prove zero object-qpos writes and can miss smaller or between-observation jumps. |
| Deterministic seed/input | **PASS** | Default seed is 42; initialization noise is explicitly zero; planning uses the seeded NumPy generator; scene geometry, timing, and upstream commits are fixed. `M0_STATUS.md` reports two runs with matching stage times/final state, though those result files are not present for review. |
| One-command run | **PASS** | `./scripts/run_m0_pick_place.sh` pins/checks both external revisions, creates the Python 3.10 environment, installs the M0 requirements, runs the demo, and writes outputs. It requires `git`, `uv`, network access for initial checkouts/package installation, and headless EGL support. |
| Result JSON | **PASS** | The runner writes a structured JSON with stage predicates, safety checks, metrics, seed, trace, and artifact paths. It writes under `/tmp/robotsim-issue43-m0/output` by default. The cited run JSON is absent from this review environment, so the reported pass is not independently auditable here. |
| Visual artifact/replay | **PARTIAL** | Code writes an MP4 and final PNG; the JSON retains a short post-release trace. `M0_STATUS.md` points to files in `/tmp/.../final-verified` and `/tmp/.../final-repeat-verified`, but those files are not present or attached to the reviewed branch. There is no durable artifact for the requested final visual review in the material available here. |
| License boundary | **PASS** | The adapter identifies the Humanoid VLA source as MIT and the referenced grasp source as BSD-3-Clause; `M0_STATUS.md` separately records Unitree mesh BSD terms. The runner keeps upstream checkouts and generated/derived scene files outside the RobotSim tree and checks upstream checkouts are clean. Preserve those attributions if assets are later redistributed. |

## Current stage

This is a reproducible M0 scripted pick/place prototype with a measured-contact gate, weld-assisted carry, free-body release/settling, numeric checks, and artifact-generation code. The branch is not yet at final acceptance for an actual bottle grasp-and-place: the object body is box-dominant, transfer depends on the weld, and the reported final placement overhangs the table while passing a center-only region test.

`M0_STATUS.md` currently says `FINAL STATUS: PASS - ready for final visual review.` That overstates what a reviewer can verify from this branch: the run outputs it cites are outside the repository and unavailable here, and the three acceptance gaps above remain. Keep final status pending until those are closed and review evidence is accessible.

## Shortest path to final review

1. Replace the box-dominant bottle body with a cylindrical bottle/cylinder collision body while keeping the same pinned G1/controller setup.
2. Carry the object through transfer using the measured hand contacts/friction; do not let an active weld supply the carry. Keep the existing weld-free release and settling path.
3. Place the whole collision footprint inside the tabletop with the configured edge margin; make the placement predicate account for all bottle geoms, not only the object center.
4. Make the no-teleport and penetration evidence cover the active rollout at physics-step resolution (or otherwise prove there are no object-state writes/jumps between sampled control frames).
5. Run the pinned one-command demo twice with seed 42 from clean output directories. Publish each result JSON, log, MP4, and PNG as reviewable PR artifacts; then make the single visual review.

No topology, transport, Unity, perception, VLA, navigation, or architecture decision is needed for these acceptance gaps.
