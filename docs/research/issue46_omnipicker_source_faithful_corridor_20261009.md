# Issue #46 X2 Source-Faithful Grasp Corridor

## Decision

**SOURCE-FAITHFUL STATIC GRASP: FAIL for both tested configurations.** The distal jaw surfaces can be placed on opposite sides of the canonical bottle body, but original wrist-roll and loop collision surfaces occupy the same bottle volume. The OPEN, approach, clean closure, and static 30 mm clearance gates fail.

**PHYSICAL BILATERAL CONTACT: NOT RUN.**

**30 MM PHYSICAL LIFT: NOT RUN.**

**DESIGN APPROVAL REQUIRED: YES** before applying any alternate collision representation, source geometry, bottle, or end effector to production.

Classification: **B. Genuine structural interference for the tested poses.** This bounded result does not show that every source-faithful OmniPicker pose is infeasible.

## Provenance

- Requested research baseline: `cbaecbfcb7e991a0f5a1c4375d2581b27209f52f`; experiment checkout HEAD: `54d3ee6a4de3430d338ee9a96acf63190ef8e883` (the baseline is an ancestor).
- X2 source: [AgibotTech/agibot_x2_urdf](https://github.com/AgibotTech/agibot_x2_urdf), pin `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- URDF: `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`; SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- License: Mulan PSL v2.
- Runtime: Python 3.10.12; MuJoCo Python/native 3.3.6; `MUJOCO_GL=egl`.
- Native library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Corridor runner SHA-256: `8304c076b79478f904bac41a7cd82dfdfaaf8b52e0e0d9daaf2eb74c973edde0`.
- Source-surface checker SHA-256: `155140bbbbf39d8aaf9452a103c1301318c776de19ca9cccf5d599b90812a5d6`.
- The runtime JSON contains all 12 source collision STL hashes, triangle counts, scales, sizes, converter/audit hashes, and canonical helper hash. The key interfering collision mesh hashes are: wrist roll `f6ed0a2a01407c6f314b6b2c04e5df8914f60963f3b546622f6734f03d487c72`; narrow loop `63b895ff646b799c303c41c924ea4c77cf00f9c5d357e2fb4f6761c649237b17`; wide loop `31aa42a993b3364e16efa657700ffe621eeaa5994d415bbe1b77f0687419b2b8`.
- The canonical bottle was unchanged: 70 mm diameter, 244.5 mm overall height, 0.570 kg, with body, shoulder, neck, and cap collision parts.

## Method

The investigation reused the measured centered-jaw pose and tested only two mechanically derived alternatives: `right_wrist_pitch_joint` at its exact source lower and upper limits, -0.523599 and +0.523599 rad. Wrist yaw and roll targets remained zero. For each orientation, the rigid test-mount translation was solved from the compiled distal jaw witness midpoint, then the source jaw aperture was bisected to opposing contact on the bottle-body cylinder. No random seeds, xyz sweep, collision decomposition, source-limit edit, or production model edit was used.

The operational grasp TCP is the midpoint of the closest-point segment between compiled `R_hand_narrow3_Link` and `R_hand_wide3_Link` collision geoms at simultaneous contact. It is centered on the bottle-body cylinder near world `[0.300, 0, 0.8775]` m. This is an experiment-derived frame, **not** a vendor-defined TCP. The jaw-normal axis points almost along world +X. The insertion axis points from the distal jaws toward `right_wrist_roll_link`; it has a large upward component because the tested wrist is above the bottle. Exact root poses, joint targets, solver bisections, frame axes, and per-pair traces are preserved in `result.json` and `robot_bottle_pairwise_clearance.csv`.

Compiled MuJoCo distances cover the bottle body, shoulder, neck, and cap. A companion source-surface check transformed every original collision-STL triangle vertex, edge midpoint, and centroid through the pinned collision scale/origin and compiled link FK, then evaluated them against analytic bottle primitive signed-distance functions. A negative sample proves a source-surface intersection; a nonnegative sample would not prove separation. For the body cylinder, signed distance is exact at those sample points. Reported source-surface minima are sampled values, not exact global minima. Supplemental views draw original collision STL facets, not vendor visual meshes; the bottle is drawn from the canonical primitive collision geometry.

This is static geometry only: no `mj_step`, controller run, or physical force calculation occurred. The extracted subtree is rooted at `right_elbow_link` with a rigid test-mount transform, so the result does not certify full humanoid arm IK or whole-body reachability.

## Candidate Measurements

| Candidate | Wrist pitch | OPEN jaw-to-body clearance | OPEN compiled minimum | Distal contact geometry | Source STL samples at bottle body |
| --- | ---: | ---: | --- | --- | --- |
| Lower limit | -30 deg | narrow +16.754 mm; wide +16.309 mm | wrist roll/body -53.399 mm | aperture 0.580300539; jaw/body +0.001369 / -0.001369 mm; witness offsets -35.010 / +35.028 mm | wrist roll -34.915 mm; narrow loop -28.172 mm; wide loop -26.693 mm |
| Upper limit | +30 deg | narrow +16.973 mm; wide +16.541 mm | wrist roll/body -45.907 mm | aperture 0.580222646; jaw/body -0.000178 / +0.001000 mm; witness offsets -34.995 / +34.999 mm | wrist roll -34.742 mm; narrow loop -27.719 mm; wide loop -25.248 mm |

The jaw witness pairs are geometrically on opposite sides of the 70 mm body. This does not make the complete corridor clear. At lower pitch, compiled wrist-roll distances to shoulder and neck are -39.462 mm and -17.712 mm; at upper pitch they are -15.491 mm and +7.200 mm. The cap is clear in both tested poses. Source STL samples independently confirm wrist-roll and loop intersections with the body at both OPEN and contact geometry.

Along the lower-pitch insertion axis, the body projects approximately -84.978 to +84.978 mm from the TCP; the shoulder occupies +55.890 to +111.816 mm. The narrow and wide loops occupy approximately -8.8 to +35.4 mm, while wrist-roll collision geometry spans -24.3 to +181.9 mm. Axial interval overlap is descriptive rather than collision proof; the negative compiled distances and negative source-surface samples establish actual intersections. Exact intervals for both poses and all bottle parts are in `result.json`.

| Static gate | Lower pitch | Upper pitch |
| --- | --- | --- |
| Source joint pose within source limits | PASS | PASS |
| All robot/bottle surfaces clear in OPEN | FAIL | FAIL |
| 80 mm insertion path clear | FAIL | FAIL |
| Distal jaws reach opposite sides geometrically | PASS | PASS |
| Closure clear except intended gripping contacts | FAIL | FAIL |
| Common 30 mm static clearance corridor | FAIL | FAIL |

The tested pitch endpoints do not remove the proximal overlap. The 30 mm check is only a rigid geometric corridor query, not a physical lift test.

## Prior Wrist Penetration

The earlier full-robot run measured -38.843732 mm between the compiled `right_wrist_roll_link` collision geom and the bottle body while the distal jaw frame was centered. That value is wrist-envelope penetration, not a 38.84 mm error in the jaw midpoint/TCP. Here, the jaw witness midpoint reconstructs onto the bottle-body center, source-root FK reconstruction error is zero, and the opposing jaw gap is about 70.0 mm, yet wrist-roll and loop intersections remain. The source STL sample test confirms these intersections are present on the pinned source collision surfaces, not only in a MuJoCo convex hull. This explanation applies only to the centered reference and the two tested pitch configurations.

## Evidence and Reproduction

The complete packet is at `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\grasp-corridor-20261008\run-04\`.

- `result.json`: exact candidates and solver iterations, limits, compiled distances, bottle-part results, closure states, static lift path, source hashes, runtime identity, and gates.
- `robot_bottle_pairwise_clearance.csv` and `state_clearance_summary.csv`: compiled clearance traces.
- `source-surface-validation/source_triangle_clearance.json` and `.csv`: sampled original collision STL surfaces against bottle parts, including the method limits and pose reconstruction.
- `wrist_pitch_{lower,upper}_limit_{front,side,top}.png`: OPEN projections.
- `wrist_pitch_{lower,upper}_limit_closeup.png`: contact-geometry close-ups.
- `source-surface-validation/annotated_collision_views/`: separate OPEN and CONTACT_GEOMETRY original-collision-surface views.
- `source/isolated_omnipicker_source_elbow_root.urdf`: extracted source subtree used for geometry compilation.
- No MP4 was produced because physics was not run.

See `REPRODUCE.md` for exact commands. The result changes no production model and does not claim global OmniPicker infeasibility.
