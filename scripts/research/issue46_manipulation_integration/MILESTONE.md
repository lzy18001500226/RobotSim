# Issue #46 / PR #91 — Full-X2 PR90 grasp milestone

Updated: 2026-10-23 18:52 UTC
Branch: codex/issue46-manipulation-integration-20261023
Baseline source HEAD before this audit: 97a1e8a0ca5fa19a34415a2d370a06b5663d77b8
PR #91: OPEN / Draft. Issue #46: OPEN.

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| GRIPPER_BASELINE | PASS | PR90 isolated unloaded MuJoCo 3.3.6 cycle: 17,500 steps, 12.831–90.999 mm five-section aperture, root tracking error 0.015576 rad. This is not a loaded grasp. |
| MOUNTING_AND_SOURCE_GEOMETRY | BLOCKED for tested bottle poses | Full-X2 uses the pinned official X2 Ultra URDF and exact right-wrist mount. Source STL triangles still enter the bottle envelope in every refined target tested. |
| PREGRASP_IK | BLOCKED as a usable witness | Static IK converged for 5/24 coarse poses and 5/7 lateral refinements, but none paired with a collision-clear endpoint and full path. |
| APPROACH_COLLISION | BLOCKED | No candidate passed the recursive swept-clearance gate. |
| PAD_FIRST_CONTACT_GEOMETRY | BLOCKED | Each refined reachable endpoint was already obstructed with the gripper OPEN; no pad-first bilateral midbody contact was established. |
| FULL_X2_DYNAMIC_CONTACT | NOT RUN | Static gates did not pass. |
| BOTTLE_LOAD_TRANSFER | NOT RUN | Static gates did not pass. |
| 50_MM_LIFT | NOT RUN | Static gates did not pass. |

The single authorized full-X2 physical trial remains unused. No bottle qpos, weld, mocap, hidden support, or collision suppression was used.

## Frozen model and static method

- PR90 HEAD: facdaa0d6e5f8152168d2e812d864b660eac9f96.
- PR90 isolated model SHA-256: 32353b81b9893fb049da55962b72a65bbae865189e1d404579502140515d6c45.
- Official X2 URDF pin: 575cc6b988f976c23550e0db85aa1e5475d3652d, source SHA-256 35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d.
- Full-X2 PR90 overlay SHA-256: fad9ae69ba99f1c8e5efd00f78a9571e9ccc04f48c7a10513ae711187718c302. The overlay changes only the two q2 equality polynomials; original masks/exclusions, wrist mount, source mesh files, object size/mass/inertia/friction remain fixed.
- Bottle: free-jointed upright cylinder, diameter 70 mm, height 230 mm, mass 0.550 kg, center [0.300, 0.040, 0.915] m; bottom rests at table top z=0.800 m.
- Collision method avoids MuJoCo mesh-pair distance false-zero behavior. It uses active compiled mesh vertices, conservative projection-gap witnesses, exact box bounds, and a circumscribed 96-sided bottle cylinder. A positive tested-axis gap is a safe lower bound; missing witnesses fail closed as unresolved.
- Swept path checks use recursive joint-space arc-length bounds, with a 2 mm clearance target and a bounded 255-node/depth-7 interval budget. No interval is accepted if its clearance cannot be proved inside that budget.
- Original source-STL surface checks transform compiled visual triangles into the full-X2 wrist frame and test the finite cylinder volume. They are supplementary to the active collision proxy gate.
- Exact scene/model identity is from the full mounted-X2 model; the previously observed isolated-fixture mesh-frame discrepancy was not reused.

## Candidate evidence

1. The first coarse packet used a 200 mm side standoff. That side-entry distance was not supported by the saved grasp plan and is retained only as a diagnostic attempt:
   /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-full-x2-bounded-24-static-20261023T175503Z/
2. Corrected 24-pose coarse screen (side standoff 39.85 mm from the saved pregrasp_point/grasp_point):
   - Run ID 52882211-82a8-4e67-a291-1321ca842d0c.
   - Evidence: /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-full-x2-bounded-24-static-20261023T180937Z/
   - Physics steps: 0; 426 mj_forward calls. Five IK witnesses, zero swept-path passes. The only completed top-down/oblique targets were blocked at their OPEN endpoint. Side-entry candidates at the saved 39.85 mm stand-off had enabled wide-pad/loop bottle intersections up to about 30.7 mm.
3. Seven evidence-derived lateral refinements:
   - Run ID 1f4d6d20-4b53-4c4d-8b23-6af30d46c591.
   - Evidence: /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-full-x2-lateral-7-refine-static-20261023T181534Z/
   - Physics steps: 0; 235 mj_forward calls. The +Y 6 mm candidate was the closest reachable endpoint; it still had -4.712 mm active proxy overlap and no clear OPEN state. Its exact source-STL check found the base surface entering the bottle cylinder by 8.76 mm and the narrow/wide loop surfaces by 3.57/4.44 mm. At +8 and +10 mm, the bounded IK solver hit the unchanged source joint limit before producing a witness.
   - At the +Y 6 mm pose, active CoACD proxies R_hand_narrow2_Link__narrow2_Link_coacd_09 and R_hand_narrow_loop_Link__narrow_loop_Link_coacd_04 overlapped by 4.712 mm. The original narrow2/narrow-loop source triangles produced zero triangle-AABB candidate pairs with a 1 mm broad-phase expansion, so their source surfaces are separated by more than 1 mm at this pose; the exact minimum was not computed. The -4.712 mm active proxy overlap is a collision-proxy artifact at this pose. The proxy remains active and cannot be ignored.
4. Across the five refined coarse poses, the first closure sample was already 100% OPEN; loop/base or pad proxies contacted the bottle before any closing command. No bilateral pad-first state was found. Thus no static contact candidate advanced to a dynamic trial.

The 2 mm swept gate also conservatively marked the otherwise near-clear top-down and oblique initial arm intervals unproven at the bounded subdivision limit. Those are not reported as collisions, but they are not certified clear.

## Fail-closed dynamic-trial gate

- The one-shot preflight read the corrected coarse result and all seven lateral refinements: run 8a44759c-4474-4c12-a662-ad2685e78aa5 at /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-dynamic-trial-gate-20261023T183444Z/.
- Decision: BLOCKED_STATIC_GATES; 31 candidate records were checked, none was authorized. It did not load or step MuJoCo. The maximum trial budget remains 30 simulated seconds / 30,000 steps, with source position/velocity/actuator limits, contact persistence, table unloading, slip and nonfinite-state stops unchanged.
- Regression checks: the fail-closed gate has 4 focused tests; the existing manipulation integration suite has 28 tests. These check authorization logic only and are not physical validation.

## X2 Ultra versus Ultra Plus source identity

- Read-only source comparison: run b4d3b187-50bc-4668-8f21-b70cbe0860d7 at /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-x2-ultra-variant-audit-20261023T184520Z/.
- The currently accepted X2 Ultra source pin remains the objective target. The local Ultra Plus IK URDF is a distinct robot description: 14 of 14 shared arm joints differ in at least one origin/limit field; it contains no collision/visual mesh elements. Its mount delta alone cannot be tested on the current full-X2 body as a source-faithful correction.
- Verdict: NOT_DIRECTLY_INTERCHANGEABLE. No variant mount or arm geometry was changed.

## Dynamic contact instrumentation prepared

- Added pr90_contact_trace.py as a read-only frame collector for a future authorized runner. It records geom/body pair identities, active-contact flags, contact dimension and friction coefficients, contact-frame force, force-on-object with the equal-and-opposite geom sign, per-role hand/table support, object pose/velocity, and relative tangential slip.
- Force/frame convention was checked against the official MuJoCo computation/API docs and upstream issue #1314: frame normal points geom0 to geom1; mj_contactForce reports the contact-frame force on contact.geom[1] (the second geom); invert it for contact.geom[0], with equal/opposite forces on the pair. The collector stores both the raw force and signed world force on the bottle.
- The module only reads MjModel/MjData after a caller step. It was not run on a physical trial; no dynamics ran in this goal turn.
- Contact-math tests: 4/4 pass in the pinned MuJoCo 3.3.6 environment. The general manipulation test suite: 29 tests, 28 pass and the optional NumPy contact-trace module is skipped on the system Python that lacks NumPy. The four contact-trace tests pass in the pinned environment.

## Source reuse audit / fallback

Existing local references do not provide a drop-in, physically validated PR90 bottle grasp:

- reference/AgiBot-Community/raicom2026-agibot-X2 at a97761820b398b6cab12b632ce9f313c394eeb7e contains an IK-only x2_ultra_plus_omnipicker_omnipicker.urdf (SHA-256 9a0ef40bd9c19d301fdaeb0a78eb167e736ecb8a2c86f7bf3260ad94f6187ff5). It uses the same single claw driver / wide1 mimic (-1 ratio), and its narrow/wide secondary and loop joints are fixed. The simplified URDF has no visual/collision elements or mesh references. The separate source audit found 63 joints versus 60 in pinned X2 Ultra, 14/14 common arm joints with differing frame or limit fields, and root robot names x2t2.5 versus X2-Ultra. Its right OmniPicker root mount is z=-0.046 m versus pinned X2 Ultra z=-0.062 m. It is not directly interchangeable and cannot validate the current robot or serve as a contact model.
- reference/AgiBot/raicom2026-x2-mujoco-scene at 62095bbdda86d6e95e00749fa64ab742e6e915ee provides the competition MuJoCo scene and object scenes; its original x2.xml has no OmniPicker claw joints.
- reference/fofarup/Raicom2026-X2-Sim at ddbd3fd926846160424ff7fb74daba42b9d2ff5f adds a simulation-only palm and two box slide-finger geoms, an equality relation, and a private command relay. This changes end-effector geometry and is not the frozen PR90 gripper. Its arm/approach/close/lift sequence is a planning reference only; repository text/code alone is not independent load-bearing evidence for PR90.
- Existing AGILINK/STEP analysis in reviews/issue-46/omnipicker-mechanism-reconstruction-20261023T055242Z/REPORT.md found matching joint frames but unresolved gears, springs, loop laws, and physical couplings. The old G2-to-MuJoCo conversion failed static closure and is not a newly discovered solution.
- The previously audited AgiBot-Community grasp demo provides ROS/AimDK planning and command sequencing, but its bundled URDF is simplified IK data without collision geometry and it is not a MuJoCo contact backend.

Recommended next human decision: confirm whether the physical robot and acceptance target are the X2 Ultra or X2 Ultra Plus variant, then explicitly authorize either (a) a separate static test of the source-identified Ultra Plus wrist mount with its matching robot model, or (b) continued use of the current pinned X2 Ultra mount with a source-supported installation/contact specification. No source model or mount has been changed here.


## Reproduction

    cd /home/lzy18001500226/robotsim-issue46-manipulation-integration-local02-20261023
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_pr90_bounded_24_candidate_search.py
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/refine_pr90_lateral_offsets.py
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_x2_variant_compatibility.py
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_x2_variant_compatibility.py
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python -m unittest scripts.research.issue46_manipulation_integration.tests.test_pr90_contact_trace -v
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/pr90_trial_gate.py --static-result /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-full-x2-bounded-24-static-20261023T180937Z/result.json --static-result /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-full-x2-lateral-7-refine-static-20261023T181534Z/result.json

Generated evidence remains external under the paths listed above. All three manifests were verified; Windows UNC access was checked for the lateral refinement and trial-gate directories. PR #91 remains OPEN/Draft; Issue #46 remains OPEN.
