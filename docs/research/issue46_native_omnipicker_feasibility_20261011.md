# X2 Native OmniPicker Feasibility

## Verdict

**No source-valid full-arm grasp configuration was established in the two bounded candidates at the existing canonical workcell.** Both candidates failed before the collision-path or closure gates. Physical bilateral contact and load transfer were therefore **NOT RUN**. This is a bounded result for the tested station and approach directions, not a claim of global X2/OmniPicker infeasibility.

The nearest optimizer states are diagnostic only. They are not accepted pregrasp or grasp poses and their collision measurements must not be read as clearances for a valid path.

## Provenance and model

- X2 source: [AgibotTech/agibot_x2_urdf](https://github.com/AgibotTech/agibot_x2_urdf), commit `575cc6b988f976c23550e0db85aa1e5475d3652d`, Mulan PSL v2.
- URDF: `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`; SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- MuJoCo Python/native: `3.3.6`, 2 ms timestep, Euler, 100 solver iterations; native library SHA-256 `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Canonical scene: 70 mm x 244.5 mm free-jointed multipart bottle, 0.570 kg, friction `[1.4, 0.02, 0.001]`; 400 mm square G1 table, top z `0.800 m`. Bottle start body position `[0.300, 0.000, 0.9175] m`; grasp TCP `[0.300, 0.000, 0.92225] m`.
- X2 base in the tested workcell: world position `[0.38, 0.32, 0.68] m`, yaw `-pi/2`.
- Original source collision/visual meshes were retained. No source joint range, friction, mass, collision geometry, bottle state, or collision exclusion was changed. The 11 historical SIMULATION_ONLY body-pair exclusions were retained; no additional exclusion was added.
- The URDF contains zero `<transmission>` elements. `right_claw_joint` has source range `[-1, 0] rad`, axis `[0, 0, -1]`, effort/velocity metadata `0`; `R_hand_wide1_joint` has range `[0, 1] rad`, same axis, and mimics the driver with multiplier `-1`, offset `0`. The baseline test model's paired-jaw servos are SIMULATION_ONLY controller emulation, not source actuators.

Right arm source joint ranges (rad):

| Joint | Range | Source axis |
|---|---:|---:|
| `right_shoulder_pitch_joint` | `[-3.080506, 2.033309]` | `[0, 1, 0]` |
| `right_shoulder_roll_joint` | `[-3.045600, 0.061087]` | `[1, 0, 0]` |
| `right_shoulder_yaw_joint` | `[-2.556907, 2.556907]` | `[0, 0, 1]` |
| `right_elbow_joint` | `[-2.356194, 0]` | `[0, 1, 0]` |
| `right_wrist_yaw_joint` | `[-2.556907, 2.556907]` | `[0, 0, 1]` |
| `right_wrist_pitch_joint` | `[-0.523599, 0.523599]` | `[0, 1, 0]` |
| `right_wrist_roll_joint` | `[-0.724312, 1.509710]` | `[1, 0, 0]` |

The compiled joint ranges matched the source limits. IK states below remained inside those limits, including at active bounds.

## Finger and wrist geometry

The actual compiled distal collision geoms use the pinned narrow and wide source STLs. They are not proxy primitives. Each distal link's visual and collision mesh use the same STL at identity origin and no scale:

| Link / mesh | SHA-256 | Triangles | Local extents (mm) |
|---|---|---:|---:|
| `R_hand_narrow3_Link` / `narrow3_Link.stl` | `0fb55604e97d18c9545567e082e071746a968f624a0aa31ece3a0cc4fc1217b0` | 16,542 | `52.95 x 27.67 x 26.00` |
| `R_hand_wide3_Link` / `wide3_Link.stl` | `a06d0c7e16ddd06e28a8e75c5a27c2e5fa025c00b79d973983f389cd08a380fb` | 28,236 | `52.92 x 27.61 x 60.38` |

The source wrist-roll visual and collision assets differ despite sharing identity URDF origins. The visual `right_wrist_roll_extend_link.stl` has extents `83.70 x 45.15 x 78.37 mm` (62,884 triangles; SHA-256 `94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de`). The collision `right_wrist_roll_link.stl` has extents `99.63 x 71.04 x 204.63 mm` (35,822 triangles; SHA-256 `f6ed0a2a01407c6f314b6b2c04e5df8914f60963f3b546622f6734f03d487c72`). Its triangle-mesh convex-hull volume is `0.000639 m^3`, versus `0.000190 m^3` for the visual mesh (3.36x). This is a verified source-model envelope difference; available evidence does not establish whether it represents a real assembly envelope or a conservative collision proxy. No modification was made.

## Two tested configurations

The TCP was anchored at the bottle's cylindrical body center and the open-channel local frame was measured from the compiled source distal collision meshes. A round-trip FK check reconstructed the source EE pose with `0 m` translation error and `5.2e-17 rad` rotation error. The full X2 was solved with bounded least-squares using the source joint ranges; the initial elbow seed was the source-range midpoint `-1.178097 rad` because neutral `0 rad` is exactly the source upper stop.

| Candidate | Insertion / jaw span | Result at pregrasp target | Nearest-state findings |
|---|---|---|---|
| A: front | insertion `-Y`, jaw span `+X`; pregrasp TCP `[0.300, 0.120, 0.92225] m` | IK optimizer terminated at a bounded local optimum; EE translation residual `14.73 mm`, orientation residual `0.14335 rad` (`8.21 deg`). `right_wrist_roll_joint` and `right_wrist_yaw_joint` are at their upper source limits; elbow is `69.86 mrad` above its lower limit. | Nearest failed state: wrist-roll to bottle-body clearance `+34.24 mm`; wrist/table `+58.55 mm`; but torso-to-wrist-yaw penetration `38.44 mm`, torso-to-wrist-pitch `30.72 mm`, torso-to-wrist-roll `15.33 mm`. This is not a valid endpoint. |
| B: orthogonal side | insertion `-X`, jaw span `+Y`; pregrasp TCP `[0.420, 0.000, 0.92225] m` | EE translation residual `178.52 mm`, orientation residual `1.58937 rad` (`91.09 deg`). Shoulder-roll and wrist-pitch are at upper source limits; wrist-roll is at its lower source limit. | Nearest failed state: wrist-roll penetrates bottle shoulder by `29.78 mm`; reported wrist/table distance is `0 mm`. Pads remain at least `69.99 mm` from the bottle body. This is not a valid endpoint. |

Because neither exact source-limited endpoint was attained, neither candidate has a valid continuous approach, jaw closure, or simultaneous opposing contact result. The nearest-state collision values are provided to locate the solver's failure neighborhood, not as candidate clearance passes. The full raw qpos/FK/limit/contact records are in `result.json` and each candidate's `*_geometry_trace.jsonl` and `*_ik_iterations.jsonl`.

## Contact-normal evidence

The previous source-geometry contact-frame experiment on the same pinned distal STLs remains the applicable physical-contact reference; it is not a result from the failed full-arm IK candidates. At that measured bilateral contact frame, the two object-side contact-force directions differed by `124.023 deg`, with normal forces `1.483 N` and `1.421 N`. Their measured resultant was `[0.060, -1.131, 0.0085] N`: lateral resultant `1.133 N`, vertical support only `0.0085 N`. Thus that measured patch pair did not form opposing normals suitable for a stable friction pinch. This does not prove that every possible source-surface patch is unsuitable; the present full-arm candidates did not reach contact, and one prior geometry-derived offset collided with the wrist.

## Decision and next action

- Source-faithful static full-arm grasp: **FAIL for both tested configurations at the existing station**.
- Physical bilateral contact/load transfer in this run: **NOT RUN**; the static IK/path gate failed first.
- Global source-gripper infeasibility: **NOT established**.
- Ideal-gripper friction-cone A/B findings are independent diagnostics and were not used as native OmniPicker success/failure evidence.

The next discriminating step is a single scene-first station adjustment that brings the canonical bottle into the right arm's source-limited workspace, followed by one collision-aware front-approach solve. The current data do not justify further finger-geometry changes or a dynamics run.

## Execution identity

Runner SHA-256: scripts/research/issue46_native_omnipicker_feasibility.py = 86e7f6ad6d58d1a4582a05bcb2d34c0078a7ad606ee51966c840eb8e78ea0693. Baseline model builder SHA-256 = 9ed3523b397d8cc009d437e445aeea8e027451dfd4846ead06cbb2a43629b28f. Endpoint acceptance required at most 1 mm translation and 1 degree rotation residual.

## Reproduction

Run from the RobotSim branch `research/issue46-omnipicker-p0-automation-20261011` in WSL2. Exit status `2` is the runner's expected result when neither candidate passes static feasibility.

```powershell
wsl.exe -d Ubuntu-22.04 --exec bash -lc "cd /home/lzy18001500226/robotsim-issue46-omnipicker-p0-20261011 && MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -B scripts/research/issue46_native_omnipicker_feasibility.py --x2-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf --canonical-helper /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robosuite-bottle-20261011/robotsim_reference/canonical_manipulation_assets.py --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/native-omnipicker-feasibility-20261011-scipy-ik-instrumented"
```

Primary machine-readable result: `result.json`. Static nearest-failure render packet: `A_front_approach_minus_y_nearest_failed_ik_*.png` and `B_orthogonal_side_approach_minus_x_nearest_failed_ik_*.png`. No physics steps were run by this feasibility runner.
