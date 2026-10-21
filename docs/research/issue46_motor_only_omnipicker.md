# Issue #46 — Motor-Only OmniPicker Result

**Kinematic visualization:** `PASS` (candidate motion only; zero physics steps)
**Root-only passive mechanism rollout:** `FAIL`
**Bottle contact/lift:** `NOT RUN`
**Model class:** `SIMULATION_ONLY / UNVERIFIED_EXPERIMENTAL`

## What was built

The isolated experiment uses the pinned `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d` visual meshes and inertias. The vendor URDF, its STL files, and the source STEP remained unchanged. The derived overlay applies the PR #84 D-site coordinates, represents the six measured B/C/E/loop pivots as hinges, retains the two D point closures and official `mimic=-1` root relationship, and omits the unverified per-side rate equality.

The right-hand kinematic preview evaluates CAD q `+0.20` to `-0.10 rad` and back. The saved source motion samples cover candidate q `-0.32..0.00 rad`; therefore the preview's positive-q segment is an analytical continuation, not measured STEP motion. The static comparison used source-backed q `-0.10..0.00`, which also stays within the official root range. Measured nearest vertices of the two jaw collision meshes range from approximately `45.0 mm` at q=+0.20, `21.6 mm` at q=0, and `6.39 mm` at q=-0.10. A single cross-section through the official visual meshes measures a `47.32 mm` inner-surface opening at q=+0.20, `39.93 mm` at q=+0.10, and `9.97 mm` at q=-0.10. These are mesh measurements, not a certified pad aperture. The canonical bottle body is `70 mm` and does not fit that demonstrated range; its `36 mm`, `0.57 kg` neck is used only as the proposed contact target.

The preview was opened in the Codex in-app browser. Its slider and Play control work, and all eight right OmniPicker visual links are shown with visibility toggles. [The preview](</home/lzy18001500226/projects/RobotSim/reviews/issue-46/motor-only-passive-20261021T133518Z/viewer.html>) is a kinematic visualization with source meshes; it does not depict physical MuJoCo actuation.

## Actuation and mobility evidence

The locally available RAICOM X2 `scripts/omnipicker_hand.py` at `AgiBot-Community/raicom2026-agibot-X2@a97761820b398b6cab12b632ce9f313c394eeb7e` publishes one `HandCommand` named `left_claw_joint` or `right_claw_joint`. Its public `open`/`close` commands map to normalized values `1.0`/`0.0`. The official v1.4.0 URDF also provides only the active root and opposite-root mimic; the narrow/wide B, C, and loop joints are fixed in that description. This supports the motor-only external API, but does not specify internal follower motion.

The experimental MJCF used two 3-row D-point connects and the official root mimic, with no Candidate B `q_root + q_B - q_E` tendon equality and no passive actuators. At the initialized pose, MuJoCo 3.3.6 reports 8 velocity coordinates and constraint Jacobian rank 5, leaving 3 system degrees of freedom. Each side has four hinge coordinates; its D pin closure has rank 2 on those coordinates, leaving **2 local degrees of freedom per side**. The official mimic couples left and right roots but removes no additional per-side internal freedom.

Only the right root position servo was actuated (`kp=0.5`, `kv=0.05`, force limit `±0.5 Nm`). Passive coordinates were initialized once from the measured CAD pose and never written after stepping began. During the first `0.055 s`, while the root target was still held at the open position, passive joints drifted substantially: narrow B/C/E ranges were `0.286 / 0.671 / 0.265 rad`; wide B/C/E ranges were `0.322 / 0.773 / 0.312 rad`. The root moved only `0.00749 rad`; the largest speed reached `25.039 rad/s`, so the safety gate stopped the rollout. D-site error stayed below `0.367 mm`, the Candidate B rate-law residual grew to `0.02899 rad`, and no contacts remained after the four scoped internal pair exclusions. This is a direct demonstration that the existing passive hinge graph does not follow one root command by itself.

The saved [root-only MuJoCo segment](</home/lzy18001500226/projects/RobotSim/reviews/issue-46/motor-only-passive-20261021T133518Z/single_motor_open_close.mp4>) contains the short failure interval; it does not show a completed open/close stroke. The separate 121-frame preview is not a physics video.

## Collision and wrist handling

The run first recorded the original collision baseline: at the mapped open pose, the four root/loop and distal/loop groups had convex-hull penetrations from `3.75–4.25 mm`. The isolated diagnostic model used the already generated collision-only CoACD decomposition and excluded only those four internal groups. All other internal pairs stayed enabled; bottle, finger, and table collisions remained enabled in the contact candidate. Because the motor-only physics gate failed, the free-bottle contact and lift stage did not run. No spring was added.

The right wrist is not included in the isolated gripper fixture. Existing static evidence for the pinned URDF reports wrist-roll triangle contacts at q=0 with `R_hand_narrow1_Link` (91 triangles), `R_hand_narrow3_Link` (2,021), `R_hand_narrow_loop_Link` (585), and `R_hand_wide1_Link` (679). The collision STL `right_wrist_roll_link.stl` has SHA-256 `f6ed0a2a01407c6f314b6b2c04e5df8914f60963f3b546622f6734f03d487c72`; transformed into the OmniPicker mount frame its bounds are approximately `[-45.96,-44.48,-76.91]` to `[53.67,26.67,127.70] mm`. The separate visual extension mesh has SHA-256 `94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de` and ends at about `+1.40 mm` on that axis. This large collision/visual extent difference is a likely source of the apparent protrusion, but the existing triangle intersections have not been classified as intended fit versus invalid collision geometry. No wrist collision was removed.

## Exact missing source relationship

The AP214 STEP includes assembly geometry/occurrences but no `KINEMATIC_PAIR`, `GEAR_PAIR`, rack-and-pinion, or other kinematic relation entity. The public command script sends only the root motor command and provides no B/C/E follower law. Thus the experiment cannot reduce each four-hinge side from two local degrees of freedom to the one controlled by the root without one additional source-defined relationship per side.

The missing input is the manufacturer’s relation between the middle link and its loop/palm for each side: for example, the verified gear pair’s ratio and phase, or the actual spring/tendon/coupler behavior and its parameters if that mechanism supplies the missing motion. The prior `theta_middle - theta_loop = source phase` / `q_root + q_B - q_E = 0` law remains unverified and was not used in the physical model.

## Maintainer visual review: passive-joint alignment

The screenshot at viewer `CAD q=+0.050 rad` shows an apparent offset near the passive links. A focused frame check found no left-to-right URDF mirror-origin defect: in the pinned v1.4.0 URDF (SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`), the paired root, B, C, and loop joints have matching parent-relative origins, axes, and corresponding parent/child names after `L_`/`R_` substitution. The generator's right-side name substitution therefore does not explain the offset.

The displayed positive candidate interval is not supported by the saved STEP motion samples. The source CSV spans candidate root q `-0.32..0.00 rad`; the current viewer analytically continues the linkage to `+0.20 rad`. For the source-backed poses q `-0.05` and `-0.10 rad`, a rigid registration fit at q=0 followed by independent comparison of the MuJoCo B/C/E joint anchors to the STEP-derived A/B/C/E pivots gives maximum per-pivot errors of `0.018 mm` (A/B/C/E: `0.012/0.0005/0.0091/0.0174 mm` at q=-0.05; `0.012/0.0006/0.0111/0.0174 mm` at q=-0.10). At the screenshot's q `+0.05 rad`, there is no STEP-sampled pose to validate the passive anchors; the displayed B/C locations are continuation results, not source-measured connection locations. The internal D constraint closes numerically, but that alone does not validate the unsampled B/C/palm pose.

At q `+0.05`, the current MuJoCo candidate places the right-side B/C/E anchors at `(-0.314,-43.384,83.434)`, `(-0.172,-32.530,104.839)`, and `(0,-21.633,73.870) mm` in the model frame. Triangle-surface distances from those axes to the official visual meshes are approximately `0.048/2.469/2.445 mm` on their respective child meshes; these are axis-to-surface distances, not a measured open gap and not proof of a pin fit. No source-grounded local nudge is justified by them.

**Disposition:** no coordinate/sign/parent correction was made. The visible alignment concern is unresolved on the viewer's extrapolated positive-q segment; the available source-backed negative-q sample segment aligns within `0.018 mm`. Resolving the screenshot state requires a source-measured positive-q STEP pose with B/C/E pivot frames, or the manufacturer's passive transmission/phase definition. The single-motor physics gate remains failed independently of this visual check.

## Evidence and reproduction

All generated files and their verified path manifest are under `/home/lzy18001500226/projects/RobotSim/reviews/issue-46/motor-only-passive-20261021T133518Z/`. The physical result is `SINGLE_MOTOR_PHYSICS_FAIL`; bottle contact/lift is `NOT RUN`. The report does not claim a working motor-only mechanism or grasp.

```bash
cd /home/lzy18001500226/robotsim-issue46-dpoint-evidence-20261021
PYTHONPATH="$PWD/scripts:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages" \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python \
  scripts/research/issue46_cad_driven_gripper.py test
```

The run used the same pinned Python environment and the existing governed run directory. `mj_step` stopped at step 55 for runaway velocity; no bottle motion or qpos teleportation occurred.
