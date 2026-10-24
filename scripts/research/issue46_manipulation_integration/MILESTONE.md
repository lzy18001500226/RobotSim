# Issue #46 / PR #91 — Full-X2 PR90 grasp milestone

Updated: 2026-10-24 00:38 UTC
Branch: codex/issue46-manipulation-integration-20261023
Baseline source HEAD before this audit: 97a1e8a0ca5fa19a34415a2d370a06b5663d77b8
Contact metrics and source-variant audit implementation commit: 91aeb2465c3e76bfa9f5ce189af2a68637d4905d
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
- The module only reads caller-provided MjModel/MjData. It was not used on a dynamic frame; no dynamics ran in this goal turn.
- Static API smoke: run b198014a-50ce-497e-8d6a-5b9e62fc4989 at /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-contact-trace-api-smoke-20261023T190032Z/; 0 physics steps, 1 mj_forward, 35 active contact records. The packet stores pair identities and distances only; force values from this penetrated static frame are deliberately omitted and not interpreted as support.
- Contact-math tests: 4/4 pass in the pinned MuJoCo 3.3.6 environment. The general manipulation test suite: 29 tests, 28 pass and the optional NumPy contact-trace module is skipped on the system Python that lacks NumPy. The four contact-trace tests pass in the pinned environment.

## Source reuse audit / fallback

Existing local references do not provide a drop-in, physically validated PR90 bottle grasp:

- reference/AgiBot-Community/raicom2026-agibot-X2 at a97761820b398b6cab12b632ce9f313c394eeb7e contains an IK-only x2_ultra_plus_omnipicker_omnipicker.urdf (SHA-256 9a0ef40bd9c19d301fdaeb0a78eb167e736ecb8a2c86f7bf3260ad94f6187ff5). It uses the same single claw driver / wide1 mimic (-1 ratio), and its narrow/wide secondary and loop joints are fixed. The simplified URDF has no visual/collision elements or mesh references. The separate source audit found 63 joints versus 60 in pinned X2 Ultra, 14/14 common arm joints with differing frame or limit fields, and root robot names x2t2.5 versus X2-Ultra. Its right OmniPicker root mount is z=-0.046 m versus pinned X2 Ultra z=-0.062 m. It is not directly interchangeable and cannot validate the current robot or serve as a contact model.
- reference/AgiBot/raicom2026-x2-mujoco-scene at 62095bbdda86d6e95e00749fa64ab742e6e915ee provides the competition MuJoCo scene and object scenes; its original x2.xml has no OmniPicker claw joints.
- reference/fofarup/Raicom2026-X2-Sim at ddbd3fd926846160424ff7fb74daba42b9d2ff5f adds a simulation-only palm and two box slide-finger geoms, an equality relation, and a private command relay. This changes end-effector geometry and is not the frozen PR90 gripper. Its arm/approach/close/lift sequence is a planning reference only; repository text/code alone is not independent load-bearing evidence for PR90.
- Existing AGILINK/STEP analysis in reviews/issue-46/omnipicker-mechanism-reconstruction-20261023T055242Z/REPORT.md found matching joint frames but unresolved gears, springs, loop laws, and physical couplings. The old G2-to-MuJoCo conversion failed static closure and is not a newly discovered solution.
- The previously audited AgiBot-Community grasp demo provides ROS/AimDK planning and command sequencing, but its bundled URDF is simplified IK data without collision geometry and it is not a MuJoCo contact backend.

### Official X2 Ultra OmniPicker data audit

- Read the official [OmniPicker User Manual for X2 Ultra](https://x2-aimdk.agibot.com/zh-cn/v0.8.1/_downloads/9179a383c97b0f37b1f0fbf51ed2cb35/omni_picker_introduction_for_x2.pdf) (21 pages; SHA-256 `a27f68422b94d114089d0e8bb4b4839babe66c8d6b8260f310c64c7253d0b173`; PDF metadata date 2025-10-26). Its closing note says hardware Ver1.2 and software Ver3.2.8. The manual says X2 Ultra but does not mark `X2 Ultra -N`; the pinned [v1.4.0 model README](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/README_CN.md) maps this URDF set to the `X2 Ultra -N` nameplate. Exact applicability of the manual's hardware installation to that nameplate remains unconfirmed.
- X2 Ultra installation pages 3-5 specify the end-cap/adapter assembly: remove four M2.5x8 screws, fix the adapter using five M3x6 screws, reinstall the four M2.5x8 screws, then fix the gripper to the wrist end using four M4x8 screws. The manual says both sides install the same way, with each wide finger toward the body's outside, and warns to preserve adapter orientation. This is a screw sequence and qualitative orientation rule, not a numerical wrist-to-gripper transform.
- The pinned X2 URDF remains the numeric source for the accepted mount: `R_omnipicker_joint` is fixed under `right_wrist_roll_link` at xyz `(0.00196, 0.00035, -0.062) m`, rpy `(3.14, 0, 0)`; the left mount is xyz `(0.00259, 0.00035, -0.062) m`, rpy `(3.14, 0, 3.14)`. Source pin and URDF SHA-256 remain `575cc6b988f976c23550e0db85aa1e5475d3652d` and `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`. The manual does not supersede these frames; no mount or geometry change was made.
- Manual typical values are 0.43 kg product mass, 30 N maximum gripping force, 1.5 kg maximum recommended payload, 120 mm maximum stroke, 0.7 s open/close time, +/-0.05 mm repeatability, and 24 V DC. The external-dimension figures label an outer envelope (about 125 mm wide in the open drawing and 88.1 mm in the closed drawing); these are not pad-to-pad aperture measurements and must not be compared directly with the PR90 five-section source-STL aperture of 90.999 mm or treated as the meaning of the 120 mm stroke.
- The manual describes an adaptive one-active-DoF gripper and force control. `set_pos(x)` uses normalized opening `0..1` (0 closed, 1 fully open); CAN control carries byte-valued position, force, velocity, acceleration and deceleration commands (`0..0xFF`, with `0xFF` documented as maximum). It does not document a conversion from the force command to newtons or a calibrated force-vs-position curve.
- The manual identifies replaceable soft pads and replaceable fingers, but supplies no pad material, friction coefficient, contact-pressure distribution, per-pad normal-force calibration, bottle-specific contact data, or internal passive-link transmission law. The published maximum gripping force is therefore not proof of this bottle's load-bearing contact capacity and does not validate PR90's simulation-only quartic coupling.

The objective remains the pinned X2 Ultra with PR90 frozen. The public manual now resolves the installation sequence, qualitative hand orientation, outer dimensions and command semantics, but it does not supply the contact/force calibration or internal transmission data needed to infer bottle support or validate the simulation-only coupling. Existing static full-X2 collision gates remain BLOCKED; the single authorized physical trial remains unused, and GRASP/LIFT/TRANSFER/RELEASE/PLACE/SETTLE remain NOT RUN. No model, mount, contact parameter, or physics behavior was changed in this source audit. The Ultra Plus IK file is not directly interchangeable.

## Source-backed route and runner gate audit — 2026-10-24

Decision: `SOURCE_BACKED_ALTERNATIVE_PREFERRED`, with explicit static-margin and simulation-only restrictions below. No new bottle-grasp `mj_step` was run in this task. PR #91 remains the owner; PR #90 and PR #92 were read-only inputs, and Issue #46 remains open.

### Frozen OmniPicker result

- Task-pinned PR #92 input `6f2700052e7e80049069a04223d7ef6a8d1da021` identifies `X2-OmniPicker-Sim-Functional-v0.1`, SHA-256 `5e33d30e212ee72271ef3cf0a790ebfb46cd31e775c578a10f1a2fb1f33c93e7`. The loader/equivalence check passes against the frozen PR #90 functional model; it does not add independent mechanism evidence. Four mesh-frame registrations still require independent fidelity review. GitHub PR #92 has since advanced to `5e2ed1dcbfac2e357fd19191733b377034ea09f5`; that newer head was not used or modified in this audit.
- The completed PR #91 screens remain the controlling evidence: 24 bounded full-X2 poses plus 7 lateral refinements; none has a collision-clear open approach and bilateral pad-first contact. Refined targets intersect original loop STL surfaces or active collision proxies before pad contact. No further duplicate OmniPicker poses were evaluated.
- Result: OmniPicker is not statically qualified for the current 70 mm bottle on the examined bounded approach set. No new dynamics were launched.

### Official X2 OmniHand alternative

- Source: `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`, `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf`, SHA-256 `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`; license Mulan PSL-2.
- Prior source-geometry audit tried 720 static preshape/pose combinations; 440 had IK witnesses and none produced a plausible static enclosure of the 70 mm / 0.570 kg bottle. The closest pose had same-side thumb/index/middle normals and ring penetration, not opposing force closure. Historical evidence: `/home/lzy18001500226/projects/RobotSim/reviews/issue-46/grasp-pose-preshape-20261016/REPORT.md`.
- The URDF contains bilateral visual/collision geometry, 10 independent active channels and 6 mimic followers per hand, but no MuJoCo actuators/transmissions; a direct URDF import does not preserve mimic constraints. Official X2 command documentation supports a 10-axis position array, but that interface is not a validated physical controller for this MuJoCo model.
- Result: genuine official hand geometry, but no current 70 mm static grasp candidate and no dynamic evidence. It is not a shorter route than the existing Robotiq diagnostic result.

### Preferred source-identified reference alternative: Robotiq 2F-85

- Source geometry: Google DeepMind MuJoCo Menagerie commit `0059d4335f8156206f63a35662313385f7ad6d74`, `robotiq_2f85/2f85.xml`, SHA-256 `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`, BSD-2-Clause. X2 body source remains `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Existing complete mounted-X2 evidence is stronger than an isolated gripper lift: `full_episode_x022_attempt03` used the canonical 0.570 kg, four-geom, 70 mm bottle and reports 1.0 bilateral-pad contact fraction, 55 mm maximum bottle COM rise (50 mm gate passed), table support lost for 675 samples, 80.88 mm lateral transfer, physical release, and 3 s free-physics settle. The object had a free joint; the trace reports zero post-start bottle qpos writes, no weld/equality, no mocap/follow, no hidden support, and no left-hand assistance. Existing packet: `/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/overnight-transfer-20261010/full_episode_x022_attempt03/`.
- This is still `SIMULATION_ONLY_DIAGNOSTIC`, not a source-faithful X2 product assembly. The recorded wrist adapter transform is translation `[0.00196, 0.00035, -0.062] m`, quaternion WXYZ `[0.5495276713, -0.4449936387, 0.5495276713, -0.4449936387]`; this transform is derived from the X2 tool mount, not a verified Robotiq manufacturer adapter. The diagnostic model excludes the original `head_pitch_link`/`torso_link` self-contact and uses a simulation-derived 0.001 rad coupler activation margin; the source-faithful coupler model exceeded its stop. The existing demo therefore supports only the clearly labeled simulation-only reference path.
- Recommended configuration to preserve for static work: fixed base station `[0.300, 0.400, 0.680] m`, yaw `-90 deg`; table center `[0.300, -0.100] m`; bottle root `[0.300, 0.045, 0.9175] m`, upright and free; grasp height 50 mm above the recorded bottle reference; right-arm pregrasp joint vector `[-1.227920, -0.036858, 0.831438, -0.436518, 0.592329, -0.057963, 1.263898] rad`. Use Menagerie `rq_fingers_actuator` over `[0,255]` with the existing `rq_split` tendon and equality constraints; these are simulation semantics, not validated X2 hardware commands.
- The earlier read-only 101-sample static replay reported a 0.005701 mm conservative world-AABB lower bound at sample 20 for `rq_right_spring_link`/`unnamed_geom_102` to `bottle_body`. A same-pose geometry reconciliation in the later 20261024 audit shows this is not actual source or compiled-proxy proximity: compiled-hull SAT lower bound 29.439789 mm, MuJoCo geom distance 29.454623 mm, source STL bound 20.033924–29.454840 mm. The old AABB number is not the path's actual limiting clearance. A separate 101-state recursive swept check later found the actual external path minimum near 2.0001 mm on a pad/body pair, only marginally above the 2 mm gate.
- Final static replay run ID `ca9648d1-a938-4caa-8e2c-f4d4f58efe93`; evidence `/home/lzy18001500226/projects/RobotSim/reviews/issue-46/robotiq-current-bottle-static-20261024T045914Z/`. Windows UNC access and artifact manifest were verified. Earlier same-day directories are retained: `044448Z` and `044514Z` contain result JSON only because report packaging stopped before manifest creation; `044852Z` and `045210Z` use earlier clearance accounting. They are superseded and are not final evidence packets.
- Result at that earlier stage: source-identified alternative with historical mounted full-X2 contact/lift evidence, not a static-margin PASS or production endorsement. The sample-20 0.005701 mm AABB value is now resolved as conservative; current same-pose source and compiled-proxy distances are well above 2 mm. The actual approach path is only marginally clear elsewhere, and a fixed internal spring/pad clearance lower bound remains below 2 mm. See the 20261024 full-X2 qualification below; no physical rerun was performed.

### Measured episode gate and validation

- Added `episode_gate.py` as a transport-neutral, model-independent stage evidence validator. It consumes ordered post-step measurements and exact model identity; it rejects missing bottle pose/orientation, missing source-limit evidence, left-hand contact, bottle qpos writes, attachment/mocap/hidden support, missing load-transfer contact/slip, insufficient upward support, missing table reaction, lift below 50 mm, table-supported lift, and hold below 1 s. Missing physical phases stay `NOT_RUN`; synthetic test samples are not physics evidence.
- Focused regression: 11 tests pass with pinned MuJoCo 3.3.6 environment Python. `py_compile` and `git diff --check` pass. The gate is not yet wired to a candidate-specific dynamic runner, and no dynamic stage was run in this task.
- Static audit reproduction:

      cd /home/lzy18001500226/robotsim-issue46-manipulation-integration-local02-20261023
      PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_robotiq_current_bottle_static.py --diagnostic-model /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/overnight-transfer-20261010/full_episode_x022_attempt03/x2_robotiq_2f85_simulation_only.xml --approach-samples /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/overnight-transfer-20261010/full_episode_x022_attempt03/raw/static_approach_path_samples.json --historical-result /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/overnight-transfer-20261010/full_episode_x022_attempt03/result.json

Initial GitHub state was checked read-only: PR #90 OPEN/Draft at `facdaa0d`, PR #92 OPEN/Draft at task-pinned `6f270005`, PR #91 OPEN/Draft at starting head `9fd3417f`, and Issue #46 OPEN. PR #92 advanced concurrently to `5e2ed1dc` during this task; that newer model was not consumed or changed. No concurrent PR was modified.

Implementation identities: `audit_robotiq_current_bottle_static.py` SHA-256 `efe6a9ca1c9831dd519842ccac33c9739c9dc7ba15779ba3800444169444e1f0`; `episode_gate.py` SHA-256 `9b8ca6f2b0da5e4c44c5f384e691cdffeea6938d63b11695d4f70147bb5d8c47`; gate test SHA-256 `6c4ffb1a8a06e4eabbdb0109113ed45709ad2f539eff7fe85451f7530fab885f`.


## Reproduction

    cd /home/lzy18001500226/robotsim-issue46-manipulation-integration-local02-20261023
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_pr90_bounded_24_candidate_search.py
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/refine_pr90_lateral_offsets.py
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_x2_variant_compatibility.py
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_x2_variant_compatibility.py
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python -m unittest scripts.research.issue46_manipulation_integration.tests.test_pr90_contact_trace -v
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/smoke_pr90_contact_trace_api.py
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/pr90_trial_gate.py --static-result /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-full-x2-bounded-24-static-20261023T180937Z/result.json --static-result /home/lzy18001500226/projects/RobotSim/reviews/issue-46/pr90-full-x2-lateral-7-refine-static-20261023T181534Z/result.json

Generated evidence remains external under the paths listed above. All three manifests were verified; Windows UNC access was checked for the lateral refinement and trial-gate directories. PR #91 remains OPEN/Draft; Issue #46 remains OPEN.

## Full-X2 Robotiq approach qualification and historical gate replay - 2026-10-24

This bounded task changed only the full-X2 right-arm retreat path and analysis/reporting code. No hand mesh, bottle, table, collision mask, coupling, actuator parameter, source limit, or physics behavior changed; no new mj_step ran.

### Static candidate

- Candidate: 25 mm pregrasp retreat along the existing grasp-axis pose; 101 linearly interpolated right-arm configurations; historical final grasp vector preserved.
- Candidate pregrasp IK: PASS, residual 3.51e-16 m position / 2.58e-8 rad orientation; minimum joint-limit margin 0.108846 rad.
- Swept lower bounds: all robot/bottle 2.000361 mm; non-pad robot/bottle 2.000631 mm; robot/table 14.253024 mm; external robot/self 2.000003 mm. Recursive midpoint subdivision uses depth 11 and a 5,000-node budget.
- The 2 mm external margins are only micrometres above threshold and do not include measured tracking error. The historical rollout maximum right wrist yaw error was 0.036843 rad; shoulder pitch was 0.022342 rad. The new candidate is a static witness, not a robust dynamic clearance certificate.
- Strict internal spring/pad clearance remains unproven against the same 2 mm threshold: the SAT projection lower bound is 1.571048 mm for rq_right_spring_link/unnamed_102 to rq_right_pad/rq_right_pad2, invariant across all arm states, with no active contact. Arm-only changes cannot alter that fixed relative geometry. This is a failed clearance certification, not evidence of penetration.
- The earlier 0.005701 mm value is reconciled below: it was a conservative world-AABB lower bound at sample 20. At the same model and qpos, compiled-hull SAT lower bound is 29.439789 mm, MuJoCo distance is 29.454623 mm, and source STL distance is bounded 20.033924-29.454840 mm. It is not actual source-surface proximity.
- Historical assumptions remain diagnostic-only: head_pitch_link/torso_link exclusion despite overlap in all 10,201 tested head poses (best -6.168 mm); derived Robotiq wrist adapter xyz 0.00196, 0.00035, -0.062 m and rpy 3.14, 0, 0 without verified manufacturer provenance; and simulation-derived +0.001 rad coupler margin. The source-faithful coupler configuration exceeded its stop at first step.

Static evidence: /home/lzy18001500226/projects/RobotSim/reviews/issue-46/robotiq-arm-approach-qualified-20261024T065752Z/ (run ID d3d13099-3791-47be-ba01-70d692ced230, 0 physics steps). Contains both complete 101-point joint paths, stage gates, exact sample-20 geometry comparison, report, reproduction command, and verified artifact manifest.

### Historical episode gate

The current episode gate was applied to all 14,350 saved physics samples. The raw integer step field repeats twice because it truncates floating timestamps; the audit verified a uniform 1 ms sample interval and reconstructed unique post-step indexes without dropping samples. Results: CONTACT, PLACE, RELEASE pass their recorded gates; PREGRASP, APPROACH, LOAD_TRANSFER, LIFT_50MM, HOLD, and TRANSFER are BLOCKED_MISSING_EVIDENCE; SETTLE fails the strict maximum angular-speed gate at 0.087526 rad/s against 0.05 rad/s, although terminal angular speed is 0.030392 rad/s.

The 1,000-sample contact hold shows 100% simultaneous force-bearing pad contacts and mean table reaction 5.204 N. During the 195-sample table-supported load-transfer prefix, mean table reaction is 3.894 N, a 1.310 N reduction. The record does not contain contact-frame orientation, so upward world force and tangential world forces are missing. Right-arm positions and velocities pass for the seven recorded joints, but whole-model source limits, actuator effort/limits, and follower residuals were not saved.

Historical gate evidence: /home/lzy18001500226/projects/RobotSim/reviews/issue-46/robotiq-historical-episode-gate-audit-20261024T070116Z/ (run ID a02ccca0-9456-4df1-ad51-542ce55c027e, 0 physics steps). It contains stage statuses, missing-field counts, force-direction limits, exact sample-20 geometry reconciliation, input hashes, reproduction command, and verified artifact manifest.

Next physical experiment is prepared but not run: after the fixed internal pair is measured exactly or accepted under an explicitly separate internal-mechanism margin, and after mount/coupler safety assumptions are resolved, use the 25 mm retreat candidate with a controller that demonstrates clearance under measured tracking error. Record all source joint qpos/qvel, per-actuator control/force, follower residuals, contact frame basis and per-contact world force. Run SETTLE -> PREGRASP -> APPROACH -> PAD CONTACT -> LOAD TRANSFER -> 50 mm LIFT -> HOLD -> TRANSFER -> PLACE -> RELEASE -> SETTLE. Gate the 50 mm lift on measured upward hand support, measured table unloading, bilateral contact persistence, table separation, bounded relative slip, and full-model safety limits. No attachment, bottle qpos writes, mocap/follow carry, hidden support, collision suppression, or left-hand assistance.

### Reproduction

    cd /home/lzy18001500226/robotsim-issue46-manipulation-integration-local02-20261023
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/qualify_robotiq_static_approach.py --diagnostic-model /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/overnight-transfer-20261010/full_episode_x022_attempt03/x2_robotiq_2f85_simulation_only.xml --historical-result /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/overnight-transfer-20261010/full_episode_x022_attempt03/result.json --historical-approach-samples /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/overnight-transfer-20261010/full_episode_x022_attempt03/raw/static_approach_path_samples.json --contact-trace /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/overnight-transfer-20261010/full_episode_x022_attempt03/raw/contact_trace.jsonl --spring-link-stl /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/mujoco_menagerie/robotiq_2f85/assets/spring_link.stl --previous-static-result /home/lzy18001500226/projects/RobotSim/reviews/issue-46/robotiq-current-bottle-static-20261024T045914Z/result.json --retreat-m 0.025
    env PYTHONDONTWRITEBYTECODE=1 /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python scripts/research/issue46_manipulation_integration/audit_historical_robotiq_episode.py --historical-result /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/overnight-transfer-20261010/full_episode_x022_attempt03/result.json --physics-trace /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/overnight-transfer-20261010/full_episode_x022_attempt03/raw/physics_trace.csv --contact-trace /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue-46/issue-46-x2-single-hand/overnight-transfer-20261010/full_episode_x022_attempt03/raw/contact_trace.jsonl --static-result /home/lzy18001500226/projects/RobotSim/reviews/issue-46/robotiq-arm-approach-qualified-20261024T065752Z/result.json

Current decision: no new physical experiment was run. Keep PR #91 Draft until the strict internal clearance, practical tracking margin, and source-faithful model assumptions are resolved.
