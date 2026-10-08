# Issue #46 Gripper Root-Cause A/B/C Benchmark

**Overnight continuation result: BLOCKED — the source-faithful OPEN static preflight still fails. Current overall classification: E, inconclusive between source-geometry interference and an over-conservative collision representation.** The earlier bounded classification C applied to the configurations in the initial comparison and did not establish global gripper/bottle incompatibility. The new source contact-frame and collision-representation evidence below narrows the cause but does not establish the intended physical envelope of the wrist collision asset. No X2 physics rollout was run because no tested OPEN configuration passed static preflight.

No production model, controller, accepted scene, or shared contract was changed. Issue #46 remains open and PR #58 remains Draft. The experiment used isolated source checkouts and external evidence storage.

## Comparison

| Candidate | Geometry/contact result | Physics result | Interpretation |
| --- | --- | --- | --- |
| A. Menagerie Robotiq 2F-85 | Four real pad collision geoms contact the canonical bottle; zero initial contacts | PASS: 100% bilateral contact for the 1 s hold; 33.045 mm maximum bottle lift; bottle is unsupported by the table at the end | Validates this bottle and the physical-contact harness. It is only a diagnostic control, not X2 acceptance. |
| B. Pinned X2 OmniPicker | The old target is laterally off-center at the distal jaws. A derived tip-centered pose still has active wrist/loop/body contacts before jaw closure. | Not run: static OPEN preflight fails | Tested geometry has a real wrist/loop interference; no conclusion about other wrist orientations or approach directions. |
| C. X2 collision-only counterfactual | Six source-triangle convex pieces replaced the collision meshes on the two loop links and wrist-roll link. The large collisions remain. | Not run: static OPEN preflight still fails | This diagnostic decomposition does not justify a production collision-model replacement. |

## A: Reference Gripper

Source: [MuJoCo Menagerie Robotiq 2F-85](https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74/robotiq_2f85), pinned at `0059d4335f8156206f63a35662313385f7ad6d74` (BSD-2-Clause). The benchmark retained the official 2F-85 gripper and its pad collision boxes, split actuator, and link constraints. A disposable, force-limited vertical carriage moved the gripper mount; it did not contact or constrain the bottle.

The canonical bottle was loaded through the accepted scene helper: 70 mm body diameter, 244.5 mm overall height, 0.570 kg, multipart body/shoulder/neck/cap geometry, geom friction `[1.4, 0.02, 0.001]`; gravity was 9.81 m/s^2 and timestep 0.002 s. The table and bottle definitions were not changed. The initial bottle state was a free joint on the table. The gripper began open with no robot-bottle contacts.

Protocol: 0.5 s OPEN, 0.5 s smooth CLOSE, 1.0 s HOLD, 0.5 s carriage lift, 1.0 s post-lift hold. First bilateral pad contact was at 0.736 s. Bilateral pad contact persisted for 100% of the hold. Maximum lift was 0.0330452 m. No bottle qpos write, bottle equality, or bottle mocap was used; the final 100 lift steps had no bottle-table contact. Peak recorded pad normal forces were 48.02 N (`left_pad1`), 75.64 N (`left_pad2`), 48.11 N (`right_pad1`), and 77.70 N (`right_pad2`).

The first carriage attempt exposed a test-fixture issue: the 1.05261 kg gripper subtree required 10.3261 N just to hold itself, and the original 1,000 N/m carriage servo settled short. The final disposable reference fixture used a 10,000 N/m, 200 N s/m carriage servo, +/-50 N force bound, and a 35 mm target. The gripper actuator and bottle properties were not changed. This is a fixture setting, not a gripper parameter or a RobotSim controller recommendation.

## B: Source OmniPicker

Pinned source: [AgibotTech/agibot_x2_urdf at `575cc6b988f976c23550e0db85aa1e5475d3652d`](https://github.com/AgibotTech/agibot_x2_urdf/tree/575cc6b988f976c23550e0db85aa1e5475d3652d), file `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, Mulan PSL v2.

The source right jaw has one active revolute `right_claw_joint` from `R_omnipicker_base_link` to `R_hand_narrow1_Link`, axis `(0, 0, -1)`, range `[-1, 0]`. `R_hand_wide1_joint` is the opposing revolute follower, axis `(0, 0, -1)`, range `[0, 1]`, source mimic multiplier `-1`, offset `0`. The distal functional collision surfaces are `R_hand_narrow3_Link` and `R_hand_wide3_Link`; the URDF does not define separate pad collision primitives. The narrow and wide loop links are fixed to the OmniPicker base in this tree. The test applied the source relation `q_wide = -q_claw` exactly once for each static kinematic sample. No dynamic follower-actuator behavior was tested.

The corrected elbow-root extraction matched 14 source joints and 15 link bodies. Source-vs-compiled forward-kinematics maximum position error was `1.22e-17 m` or less; maximum orientation error was `3.6500e-8 rad`. Source mesh scale and URDF collision origins were applied when sampling original STL triangles. The narrow tip's source-surface and compiled-hull distance agreed within 1 micrometre. These checks do not show a joint-tree, scale, or broad link-origin conversion defect in the tested state. They also do not validate dynamic mimic behavior.

The previous candidate used aperture `0.5843397233`, with `q_claw = -0.5841710438 rad` and `q_wide = +0.5841710438 rad`. At that state:

| Surface | Compiled MuJoCo signed distance | Sampled original STL signed distance |
| --- | ---: | ---: |
| narrow3 tip | +6.030 mm | +6.030 mm |
| wide3 tip | -7.632 mm | -7.411 mm |

Thus the tip pair is both off-center and slightly over-closed at that target: the signed clearances differ by 13.662 mm, with a 6.831 mm midpoint offset, while their sum indicates about 1.602 mm excess closure. The exact original target and pose are preserved in `x2-geometry-8/result.json`.

One evidence-derived correction translated the wrist root `6.911743 mm` along the measured jaw axis `(0.999997326, -0.002312414, 0)` and solved aperture to `0.6025301219` (`q_claw = -0.6023250617`, `q_wide = +0.6023250617`). The compiled tip distances then became `+7.278` and `-7.278` micrometres. The original STL samples at that same state were `+7.268` and `+165.072` micrometres. The wide-tip collision hull therefore reports a small overlap where the sampled source surface remains slightly clear; this local collision-model error is approximately 0.172 mm here.

The corrected pose is not a valid pregrasp. It has an active `right_wrist_roll_link`/bottle-body contact at `-44.035 mm`, plus active loop and wrist contacts. At the original target, the wrist-roll hull contact is `-37.143 mm`; the source STL surface sample is `-34.735 mm`. The narrow and wide loop source surfaces are `-22.395 mm` and `-30.697 mm` at the original target. At the centered candidate they are `-29.265 mm` and `-26.083 mm`; wrist-roll source surface is `-34.975 mm`. These source-triangle samples were compared against the canonical bottle body cylinder; compiled `mjData.contact` rows additionally show active contacts with the bottle shoulder, neck, and cap.

At OPEN, the candidate configurations already have active non-tip bottle contacts. At the corrected candidate the distal tips have positive clearances of `+17.729 mm` and `+12.655 mm`, yet wrist/loop collision rows are already present. Across the 101-state aperture profile, tip clearance transitions from OPEN to overlap; for the original candidate the wide tip first reaches the cylinder around aperture ratio `0.79` and narrow tip around `0.45`. The centered pose makes the two tip distance queries nearly equal at one exact aperture, but MuJoCo reports only `R_hand_wide3_Link` as an active jaw contact there. The two functional tips therefore do not establish bilateral physical contact in a collision-free configuration.

The X2 model has one explicit body exclusion and the default parent filter is enabled. `mj_geomDistance` measurements are reported as geometric queries; they are not treated as proof of an active contact. The bottle rows above were independently checked in `mjData.contact`; they have `excluded=0` and active constraint addresses. Thus the large wrist/loop values are not merely filtered adjacent-link overlap.

## C: Collision Representation Counterfactual

The separate diagnostic URDF keeps source visuals, joints, transforms, all non-target collisions, and the canonical bottle. Only the collision meshes on `R_hand_narrow_loop_Link`, `R_hand_wide_loop_Link`, and `right_wrist_roll_link` were replaced by six convex hull pieces each, constructed by binning source STL triangles along each mesh's principal axis. This is a localized source-derived decomposition, not a production model.

The summed local-piece/single-hull volume ratios were 1.609 for the narrow loop, 1.600 for the wide loop, and 0.992 for wrist-roll. At the original pose, the minimum non-jaw distance remained `-37.143 mm`; at the centered pose it remained `-44.035 mm`. Active bottle contacts persisted on wrist, loop, and jaw structures. The decomposition therefore neither opens a collision-free grasp channel nor resolves the measured interference. Because loop-piece volumes expand substantially, this particular decomposition is not suitable as a production collision model.

No X2 contact-force trace or physical X2 MP4 exists: dynamics were deliberately not run after the OPEN collision preflight failed. The MP4 in this packet is the successful A reference-gripper physical lift.

## Root-Cause Assessment

1. **Grasp TCP/contact definition:** The original jaw-gap target was not centered on the two compiled functional tips. The one-step geometric correction fixes the tip-center mismatch at a static pose, but not the larger obstruction.
2. **URDF-to-MuJoCo kinematics/tree:** No evidence of a conversion error in the tested right-elbow-root state. Static FK matched; the source mimic coordinate was applied with the correct sign. Dynamic mimic fidelity remains untested here.
3. **Wrist/loop interference:** Confirmed for both tested configurations, including the source STL surface samples. It is the dominant blocker for these configurations.
4. **Collision representation:** The wide3 convex hull has a small local overreach (about 0.17-0.22 mm in these samples). The tested loop/wrist decomposition did not improve the dominant intersections.
5. **Source gripper vs bottle incompatibility:** Not established. Two pose configurations are insufficient to rule out a different wrist orientation/approach that preserves the source geometry.

**Primary classification: C, for the tested configurations only.** The next justified step is a maintainer-reviewed wrist orientation/approach hypothesis derived from the source STL free channel, followed by static OPEN active-contact validation before any X2 physics. Do not change the production collision model based on this diagnostic.

## Runtime Identity and Reproduction

Python `3.10.12`; MuJoCo Python and native library `3.3.6`; `MUJOCO_GL=egl`; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.

Key source hashes are in each run's `runtime_identity.json`, including all X2 source mesh hashes. Key identities:

- X2 URDF SHA-256: `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`
- X2 `narrow3_Link.stl`: `0fb55604e97d18c9545567e082e071746a968f624a0aa31ece3a0cc4fc1217b0`
- X2 `wide3_Link.stl`: `a06d0c7e16ddd06e28a8e75c5a27c2e5fa025c00b79d973983f389cd08a380fb`
- Menagerie `2f85.xml`: `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de`
- A generated scene XML: `a1d1f1605a00b08776734503da25ffecee8a27f332b98e55f2f9a63d0722b4ac`
- Accepted canonical scene helper: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`
- A runner SHA-256: `7bb480019aa1e74ce60e61d743a6f80c4e6a3b2da8096e2951c21510071f878d`
- B/C runner SHA-256: `b35511be923260fa80bd12a684f77a610caf047281b677db4617220ff2ee8a9a`

From the isolated research branch checkout:

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_root_cause_abc.py --phase reference \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/root-cause-abc-20261008/reference-6

MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_root_cause_abc.py --phase x2 \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/root-cause-abc-20261008/x2-geometry-8
```

The external evidence directory contains full JSON identities/results, the 101-state closure CSVs, the generated diagnostic URDF and source-derived meshes, and A's physical trace/video/images. No X2 physical rollout was attempted.

## References Reviewed

- [SO-101 Issue #2167 and complete thread](https://github.com/strands-labs/robots/issues/2167). The original hull-overlap explanation was corrected in later thread comments: the jaw pair is parent-filtered, so raw `mj_geomDistance` overlap did not establish active self-contact; a roughly 9 mm channel existed. Later comments also corrected the probe-radius interpretation and the selected body-vs-site TCP mismatch (98.4 mm), and a subsequent remeasurement found the site close to/on the static hull while offset from the channel. The thread's later corrections, not its initial interpretation, are the relevant reference.
- [Menagerie Issue #205](https://github.com/google-deepmind/mujoco_menagerie/issues/205). The issue is a user reproduction request with zero comments; its cube is 5 g and its scripted controller is not a comparable physical success. Its pasted model illustrates the 2F-85 pad boxes, tendon and equality construction.
- [Menagerie Robotiq 2F-85 at the tested pin](https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74/robotiq_2f85).
- [MuJoCo 3.3.6 collision selection and filtering](https://mujoco.readthedocs.io/en/3.3.6/computation/index.html#collision-detection): parent/child filtering and `contype`/`conaffinity` are separate from geometric distance queries.
- [MuJoCo 3.3.6 mesh collision reference](https://mujoco.readthedocs.io/en/3.3.6/XMLreference.html#asset-mesh): mesh geoms use their convex hull for collision; non-convex geometry needs convex decomposition or another supported representation.

## Overnight Continuation: Contact Frame and Collision Representation

The earlier known-good Robotiq 2F-85 control remains PASS and was not repeated: bilateral contact persisted through the 1 s hold and a real 33.045 mm lift, with the canonical 0.570 kg bottle unsupported by the table at the end. This validates the bottle and the basic physical-contact/carriage harness. It is not evidence of X2 capability.

### Why the prior full-robot target put the wrist into the bottle

The previous full-arm solver transferred the isolated bottle-relative transform of `R_omnipicker_base_link` to the full scene, then solved the seven source-limited arm joints for that endpoint. The saved endpoint residual was approximately 0.054–0.056 mm and 0.00865–0.0122 degrees. Thus the reported 38.841 mm wrist-roll overlap was not an IK convergence error or an arm-joint frame conversion error: the arm reached the requested tool pose, and that candidate pose placed the wrist-roll collision envelope 38.841 mm into the bottle body, with the narrow/wide loop links also inside by 28.155/28.092 mm while the jaws were OPEN. The collision-aware endpoint gate rejected that candidate; it was not accepted as a valid pose or path.

The previous target was a geometric bottle-relative `R_omnipicker_base_link` pose, not a vendor-defined TCP. Its functional jaw region could span the bottle, but the chosen gripper orientation also placed the wrist/loop envelope through the bottle. The source-frame follow-up independently reconstructed orientation from compiled opposing jaw witnesses and the fingertip-to-base chain, then solved the source surfaces directly against the canonical cylinder. The jaw gap is 70.000293 mm at aperture ratio 0.580188882; the jaw normal (narrow to wide) is `[0.013146293, -0.999878744, -0.008346935]`, and the finger axis (tip to base) is `[0.000132166, -0.008345918, 0.999965163]`. The resulting right-elbow root pose is position `[0.297878691, 0.024764212, 1.244412551]` m with rotation matrix:

```text
[[ 0.013146293, -0.999878744, -0.008346935],
 [ 0.999913575,  0.013146938, -0.000022432],
 [ 0.000132166, -0.008345918,  0.999965163]]
```

At the solved contact aperture, the compiled narrow/wide distal surfaces are 1.1748/1.0000 micrometers from the bottle-body cylinder. The surface-centering translation correction was only `-1.6810e-7 m`. This directly verifies the target frame and jaw-center solution. It does not make the source-faithful OPEN state valid: OPEN still has 11 active, unfiltered robot-bottle contacts across wrist, loop, and proximal jaw structures; the minimum compiled distance is -58.086 mm at `right_wrist_roll_link`. OPEN table clearance is +51.784 mm, so table placement is not the cause of this preflight failure. The functional distal surfaces are not the structures creating the first OPEN collision.

### Wrist mesh and collision counterfactuals

The pinned URDF assigns distinct meshes to `right_wrist_roll_link`: collision uses `right_wrist_roll_link.stl` (35,822 triangles, extents 99.633 x 71.043 x 204.628 mm), while visual uses `right_wrist_roll_extend_link.stl` (62,884 triangles, extents 83.700 x 45.151 x 78.367 mm). Their URDF origins and scales are both zero/one. The collision mesh convex-hull volume is 3.3611 times the visual mesh hull volume, and 69.0% of collision vertices lie outside the visual convex hull. In the source-frame OPEN pose, sampled source collision triangles reach -34.792 mm relative to the bottle-body cylinder, with 35.0% of samples inside; sampled visual triangles remain +23.423 mm clear, with none inside. This is strong evidence of a collision-envelope discrepancy, but the URDF alone does not establish whether the extra collision geometry models an intentionally hidden extension or is over-conservative.

Three static collision variants were compared at the same derived frame. None is an approved physical model:

| Variant | OPEN active bottle contacts | Minimum distance | Result |
|---|---:|---:|---|
| Pinned source collision meshes | 11 | -58.086 mm, right wrist roll | FAIL |
| Diagnostic wrist collision replaced by its official visual STL | 10 | -35.489 mm, right wrist roll | FAIL; loops and proximal links still contact, and MuJoCo convex-hull collision still reports overlap |
| Diagnostic one-convex-hull-per-exact-vertex-connected-source-mesh-component across the right elbow/wrist/gripper subtree | 38 | -56.153 mm, right wrist roll | FAIL; all source triangle components retained, but the decomposition did not open the corridor |

The component-hull diagnostic decomposed 81 mesh-connected components across 12 collision links. It changed collision approximation only; source visuals, triangle assets, mesh scale, link origins, joint tree, source ranges, and bottle remained. The larger contact-row count is not interpreted as improved physical fidelity. No wrist/loop geometry was disabled. These static results complete the authorized collision-representation comparison but do not justify replacing the production model.

### Stage result and remaining decision

- A. Robotiq baseline: PASS, completed in the earlier run and not repeated.
- B. OmniPicker contact frame and interference: source kinematics/tree remain consistent; the source-derived jaw contact frame is centered to micrometer scale. Source-faithful OPEN still fails from wrist, loop, and proximal-link contacts.
- C. Collision representation: the visual/collision wrist mesh envelope differs substantially; neither the visual-mesh substitution nor the source-component hull diagnostic passes OPEN. Both are diagnostic-only.
- X2 bottle contact, hold, and lift: NOT RUN. The static OPEN preflight is the stopping gate; no active-rollout object state writes or X2 dynamics occurred in this continuation.

**Classification: E — inconclusive.** The exact unresolved distinction is whether the pinned wrist collision STL's larger envelope is intended hardware geometry or an over-conservative/incorrect collision asset, and how the loop/proximal collision surfaces should be interpreted if the wrist asset is corrected. The immediate observed blocker is a source-faithful OPEN collision, not controller/contact-force behavior. The result does not prove global source gripper/bottle incompatibility.

The next maintainer decision is to confirm the intended physical/collision interpretation of the two pinned wrist meshes and loop geometry before authorizing any production collision-model correction or further source-faithful grasp search. No collision model, accepted scene, controller, source limit, or shared contract was changed.

Exact continuation command (already executed; the raw output is preserved and was not rerun during closeout):

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python scripts/research/issue46_root_cause_frame_followup.py --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/root-cause-abc-20261008/overnight-frame-collision-20261008/frame-2
```

The output includes `result.json` with runtime identity and source mesh hashes, `source_contact_frame.json`, `static_state_comparison.csv`, `wrist_visual_collision_comparison.json`, both PNGs, the extracted source subtree, the diagnostic URDFs and generated component OBJ meshes. The runner SHA-256 is `d04d658a20967ad8545dd7e7a6cee2faecefa33ea61b291163335ab026d9296e`; native MuJoCo is 3.3.6 with SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`; X2 source remains pinned to `575cc6b988f976c23550e0db85aa1e5475d3652d` under Mulan PSL v2. Raw evidence is under `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\root-cause-abc-20261008\overnight-frame-collision-20261008\frame-2\`. The 105-file raw-output integrity manifest is `frame-2.evidence_manifest.sha256` beside that directory (SHA-256 `58846ec51e4ab3e7f91c425a8dd39788f8291eaccef111cc7a5108e960e2dd95`).
