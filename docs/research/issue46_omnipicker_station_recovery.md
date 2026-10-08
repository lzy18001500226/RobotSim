# Issue #46 X2 OmniPicker Station Recovery

**Result: BLOCKED — the two authorized front-approach layouts did not produce a collision-free static corridor.** No physics rollout was run. This is a result for these two configurations, not a claim that X2 OmniPicker grasping is globally infeasible.

## Scope and provenance

- Starting RobotSim HEAD: `521eaadd48edb26fe344e2696f73659bc04b228a`
- RobotSim branch: `codex/issue46-omnipicker-1dof-m0-20261007`
- X2 source: `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`
- URDF: `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`
- License: Mulan PSL v2
- Python: 3.10.12; NumPy: 1.26.4; MuJoCo Python/native: 3.3.6 / 3.3.6; `MUJOCO_GL=egl`
- Native MuJoCo library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`
- Accepted controller/baseline runner SHA-256: `9ed3523b397d8cc009d437e445aeea8e027451dfd4846ead06cbb2a43629b28f`
- Geometry helper SHA-256: `feb04ed9e0d5691ccdb659d583ac8b8fe03a535cd769263cc70d9078a7ff6a2e`
- Canonical table/bottle scene helper SHA-256: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`
- Executed recovery runner SHA-256: `1623f7ab84fe84f74d48f7961e5e92382efd5e8cb71a408c3b2430d0fa69ff86` (preserved as `station_recovery_executed.py` in the external evidence folder)
- Physics steps: `0`

## Solver and frame audit

The compiled pelvis frame maps local `+X` forward to world `-Y` and local `+Y` to world `+X`. The G1 table at `[0.30, -0.10, 0]` is in front of the robot. The previous world `+Y` 100 mm pregrasp backoff was therefore directed from the bottle toward the robot, not sideways.

The canonical bottle root is `[0.30, 0, 0.9175]`; the compiled `bottle_body` collision cylinder center is `[0.30, 0, 0.8775]`. The cylinder is 70 mm in diameter and 155 mm high, so its collision center is 40 mm below the multipart bottle root. The tested grasp-height target was the center of this cylindrical body, not the bottle root.

The compiled jaw surface gap is 103.090858 mm OPEN. The aperture ratio that brings the compiled opposing jaw surfaces to the 70 mm bottle diameter is `0.580188882`. The gap midpoint at that aperture differs from the OPEN gap midpoint by 13.453578 mm. The recovery solver derives separate OPEN and closure-aperture points; the earlier solver reused the closure point for OPEN pregrasp.

All seven right-arm source axes and hard position ranges exactly match the compiled MuJoCo axes and ranges. The finite-difference check of the complete arm point Jacobian differs from `mj_jac` by at most `1.4116e-9`. The pose optimizer converged within the 1.5 mm / 1 degree target tolerance for both candidates. It uses source ranges as bounds and records every evaluation; collision is checked on the resulting poses and sampled route, not included as an optimizer residual.

| Joint | Source and compiled axis | Source and compiled range (rad) |
|---|---:|---:|
| `right_shoulder_pitch_joint` | `[0, 1, 0]` | `[-3.080506, 2.033309]` |
| `right_shoulder_roll_joint` | `[1, 0, 0]` | `[-3.045600, 0.061087]` |
| `right_shoulder_yaw_joint` | `[0, 0, 1]` | `[-2.556907, 2.556907]` |
| `right_elbow_joint` | `[0, 1, 0]` | `[-2.356194, 0]` |
| `right_wrist_roll_joint` | `[1, 0, 0]` | `[-0.724312, 1.509710]` |
| `right_wrist_pitch_joint` | `[0, 1, 0]` | `[-0.523599, 0.523599]` |
| `right_wrist_yaw_joint` | `[0, 0, 1]` | `[-2.556907, 2.556907]` |

The previous candidate-3 seed was inside source limits but began with 41.774 mm table-top, 52.625 mm front-left-leg, and 2.156 mm jaw-link penetration. Its old line search rejected colliding iterates, so it could not recover from that seed. The present candidates start from reset neutral arm state. Collision checks include all active robot collision geoms and compiled `mjData` contacts; the former right-side-only clearance list omitted torso and other robot bodies.

## Candidate results

Both candidates keep the accepted table, bottle, gripper, floor, and robot geometry unchanged. Both keep yaw at `-pi/2`, so the torso still faces the table. Candidate B’s base X was derived from the compiled neutral right-shoulder position to align the shoulder with the bottle; candidate A moves the base 40 mm forward along the verified facing direction. Both reset stations have zero unwanted initial contacts and zero initial source-limit violations.

| Candidate | Base position (m), yaw (rad) | Static result and first geometric failures |
|---|---|---|
| A: forward shifted | `[0.38, 0.28, 0.68]`, `-pi/2` | Pregrasp pose is source-valid and reaches the target within 0.054 mm / 0.0122 degrees, but wrist yaw/pitch/roll intersect the torso by 27.800 / 31.762 / 29.348 mm. The sampled approach also intersects hip/wrist geometry and the front-left table leg. At the OPEN grasp arm pose, the wrist overlaps the bottle body by 38.844 mm and narrow/wide loop links by 28.160 / 28.088 mm. |
| B: shoulder aligned | `[0.443, 0.32, 0.68]`, `-pi/2` | Pregrasp is source-valid, contact-free, and reaches the target within 0.056 mm / 0.00865 degrees. The sampled neutral-to-pregrasp route intersects the front-left table leg by 6.925 mm. At the OPEN grasp arm pose, the wrist overlaps the bottle body by 38.841 mm and narrow/wide loop links by 28.155 / 28.092 mm. |

At the 70 mm closure aperture, the narrow and wide jaw surfaces are respectively 1.103 micrometers inside and 1.791 micrometers outside the bottle-body cylinder in candidate B (candidate A is within 5.3 / 6.0 micrometers). This confirms that the jaw surfaces can span the nominal diameter; it does not make either whole-hand pose valid because the wrist and loop geometry already intersect the bottle while OPEN.

Neither configuration passes the required full static corridor. The best candidate’s failure is therefore not a source-limit or pose-convergence failure: its OPEN hand/wrist/link geometry collides with the bottle, and its sampled approach also clips a table leg. No no-contact OPEN/CLOSE regression, bottle hold, lift, trajectory, or MP4 was run because the static gate did not pass. No global reachability conclusion is drawn; the sampled joint-linear route does not rule out every possible path.

## Reproduction and evidence

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python scripts/research/issue46_omnipicker_station_recovery.py --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-station-recovery/20261008-run3
```

Complete candidate JSON, every solver evaluation JSONL, route samples, source/runtime identity, console log, original overview, candidate overview/front/side/hand/table-clearance views, and the exact executed script are under `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-station-recovery\20261008-run3\`. The first harness attempt stopped before candidate evaluation because the renderer requested 800 pixels from a 640-pixel framebuffer. The second stopped after candidate A's solve because a NumPy boolean was not JSON serializable. Those incomplete attempts are retained in sibling `20261008-run1` and `20261008-run2` folders; only run3 contains complete two-candidate results.
