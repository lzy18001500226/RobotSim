# Issue #46 Native OmniPicker Workcell Recovery

## Result

**FAIL — no tested candidate passed the complete source-limited pregrasp and approach gate. Physical contact and lift were not run.** This is a bounded result for the three tested front-workcell layouts, not proof of global X2/OmniPicker infeasibility.

The experiment starts from RobotSim `cfcad8e6e3f0a0563677b90f5758a467727ee156`, branch `research/issue46-omnipicker-p0-automation-20261011`. The official AgiBot source is pinned at `575cc6b988f976c23550e0db85aa1e5475d3652d`, URDF `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`. The accepted canonical asset helper hash is `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`. MuJoCo Python/native are both 3.3.6, timestep 2 ms, Euler integrator.

All movable compiled joint ranges used by the solver match the source URDF. Fixed URDF joints that carry placeholder limit attributes are not compiled DOFs and are excluded from the range comparison. The 11 historical simulation-only collision exclusions were retained; no new exclusions, collision edits, source-limit changes, bottle/table edits, or active-rollout state writes were made. The jaw controller remains `SIMULATION_ONLY_M0` paired-jaw actuation.

## Bounded Layouts

The canonical bottle remains 70 mm diameter, 244.5 mm tall, 0.570 kg at `[0.300, 0.000, 0.9175]` m. The G1 table remains centered at `[0.300, -0.100]` m with 0.200 m half-extents and top at 0.800 m. Each base position aligns the right shoulder pivot to the listed front work line; no robot or object geometry was translated during rollout.

| Candidate | Base position (m) | Yaw | Heading error | Pregrasp | First approach failure |
| --- | --- | ---: | ---: | --- | --- |
| A front nominal | `[0.442920, 0.323260, 0.680]` | -90 deg | 18.66 deg | 14.497 mm, 5.577 deg residual; elbow reaches lower bound and wrist roll/yaw reach upper bounds | Not attempted after source-limited pregrasp failed |
| A front, 10 deg toward table | `[0.441315, 0.348392, 0.680]` | -100 deg | 7.49 deg | 0.002 mm, 0.002 deg; collision clear | Path sample 11/25, TCP `[0.300, 0.065, 0.92225]` m |
| A front, 15 deg toward table | `[0.438894, 0.386158, 0.680]` | -105 deg | 0.94 deg | <0.001 mm, <0.001 deg; collision clear | Path sample 11/25, TCP `[0.300, 0.065, 0.92225]` m |

All three reset layouts had zero active robot self/table contacts and positive minimum whole-robot/table clearance: 112.173 mm, 109.374 mm, and 132.106 mm respectively. Candidate 2 and 3 pregrasp poses remained inside all source limits. Their nearest torso/wrist clearances at pregrasp were 7.975 mm and 27.763 mm. The earlier torso-to-wrist penetration from the prior station is therefore absent in these tested poses.

## First Blocking Geometry

Both continuous straight front approaches fail at the same TCP waypoint. In the best constrained pose found there, translation residual is 5.111 mm for Candidate 2 and 5.117 mm for Candidate 3, while orientation residual is only 0.013 deg and 0.007 deg. The SLSQP solve terminates with “Positive directional derivative for linesearch.” It is not an exact-pose/path pass.

The active clearance constraint is the compiled source collision pair:

| Layout | Pair | Signed clearance | Witness on wrist mesh (m) | Witness on bottle (m) | Pad-to-bottle clearance | Wrist-to-table clearance | Torso-to-wrist clearance |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| Candidate 2 | `right_wrist_roll_link/geom_89` to `m0_bottle/bottle_body` | 1.000 mm | `[0.2869609, 0.0324805, 0.9178835]` | `[0.2865883, 0.0334085, 0.9178835]` | 48.216 mm | 58.877 mm | 36.825 mm |
| Candidate 3 | `right_wrist_roll_link/geom_89` to `m0_bottle/bottle_body` | 1.000 mm | `[0.2869644, 0.0324819, 0.9179689]` | `[0.2865919, 0.0334099, 0.9179689]` | 48.234 mm | 58.946 mm | 65.475 mm |

Thus the current straight insertion brings the wrist-roll collision envelope to its 1 mm safety margin while both functional pads remain about 48 mm from the bottle. Table clearance and torso/wrist collision are not the blockers at this segment. The contact geometry audit records the closest compiled geoms and witness points in each candidate geometry trace.

The prior source-mesh audit found that the wrist-roll visual and collision assets differ substantially: visual extents about `83.70 x 45.15 x 78.37 mm`; collision extents about `99.63 x 71.04 x 204.63 mm`, with a compiled convex-hull volume about 3.36 times the visual-mesh volume. This supports treating the envelope as potentially conservative, but there is no authoritative CAD or physical measurement proving the collision mesh is wrong. The collision mesh was preserved.

## Physical Gates

- Source-valid pregrasp: PASS only for Candidates 2 and 3.
- Continuous collision-free approach: FAIL for Candidates 2 and 3; Candidate 1 fails pregrasp.
- Bilateral dynamic contact/load transfer: NOT RUN.
- 50 mm object lift and one-second unsupported hold: NOT RUN.
- Weld, object teleport/qpos write, mocap attachment, hidden support: not used.

Static clearance and IK do not establish a physical grasp. No manipulation PASS is claimed.

## Evidence and Reproduction

Complete raw JSON, candidate IK iteration traces, geometry traces, four-view PNGs, exact command and generated report are under:

`C:\Users\HP\RobotLab\evidence\issue46-native-omnipicker-workcell-20261011\attempt-94e0b6ec-94bf-4fd1-a1ed-a3c0a779e241-retry3\`

The reproduction command is recorded verbatim in `experiment_commands.txt`. The output `result.json` records the runner SHA-256, source identities, layout geometry, all candidate gates, and marks physical contact/lift `NOT RUN`.

## Next Discriminating Experiment

Keep Candidate 2's exact source-valid pregrasp and final bottle grasp target. Change only the approach path profile: use one collision-derived high/oblique staging path that keeps the wrist-roll envelope outside the bottle until both jaw pads enter the grasp corridor. Recheck the complete swept path with the original compiled collision mesh. Do not alter the collision mesh or add exclusions. Proceed to contact physics only if the entire path and closure pass.
