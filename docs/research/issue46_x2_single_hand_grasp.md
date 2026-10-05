# RobotSim Issue #46: X2 Single-Hand Bottle Grasp

**Status: FAIL.** The corrected station, vendor appearance, canonical table/bottle combination, and collision-free pregrasp are recorded. The right hand did not establish a stable multi-finger grasp, lift the required 50 mm, or place and settle the bottle at the target. This is not ready for X2 single-hand visual review.

## Source and runtime

- Vendor: [AgibotTech/agibot_x2_urdf](https://github.com/AgibotTech/agibot_x2_urdf), exact commit `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Model: `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf`; SHA256 `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.
- License: Mulan PSL v2; pinned `LICENSE` SHA256 `d5777bea50dc2cd111845d8a9c51ab990d188ce343b843e5ad3e653b5fa8fdab`.
- Runtime: WSL2 Ubuntu 22.04, Python 3.10.12, MuJoCo 3.3.6, NumPy 1.26.4, `MUJOCO_GL=egl`.
- RobotSim harness commit tested: `b62d63174950cef9e1dd8a24da22348c5077b096`. The run identity marks the worktree dirty because four pre-existing untracked `debug_issue46_*.py` scratch files were present; they were not staged or changed. The tracked harness files were at the recorded commit.
- The vendor checkout remained clean. No vendor asset was copied into RobotSim or edited.

## Canonical scene and appearance

The source table uses the G1 PR #45 M0 table definition: top height `0.80 m`, source center `[0.30, 0.00] m`, source half-extents `[0.20, 0.12] m`, target center `[0.30, -0.24] m`, target half-extents `[0.20, 0.10] m`, `0.02 m` gap, and `0.04 m` target-marker offset. The dark blue-gray table color is retained from that scene. The object is the current X2 bottle: `70 mm` body diameter, `244.5 mm` total height, `0.57 kg`; the previous G1 bottle is not used.

The X2 URDF contains 40 visual material color entries, all white (`1 1 1 1`) or black/transparent (`0 0 0 0.5`), and no URDF texture entries. The vendor visual meshes and authored colors are retained, including the black chest logo. A light neutral fallback is applied only to uncolored hand visual geoms. The vendor scene checker used equal RGB colors; although the replacement checker texture is assigned in the compiled model, it renders flat at this EGL overview scale. The latest scene therefore uses a restrained neutral floor with 34 thin, collision-disabled grid geoms. Headlight ambient and diffuse were lowered; specular is `0.45` to retain mesh highlights. The overview camera is closer while keeping the robot, table, and bottle in frame.

The X2 fixed base is `[0.38, 0.32, 0.68] m`, yaw `-pi/2`, so its forward axis is world `-Y` and its torso faces the canonical table edge. Walking is not used. The table and bottle lie in front of the robot and within the right arm workspace. The palm frame comes from the vendor `R_palm_joint`; its pregrasp normal is approximately world `-Y`. Measured pregrasp position error is `8.70 mm`, rotation error `0.032 rad`; no preclosure bottle contact, right-hand/table contact, or opposite-hand contact occurred. Bottle translation before closure was `0.186 mm`.

The table values in this scratch scene match the G1 M0 source, but no shared table/bottle-definition refactor was made: the current task leaves G1 PR #45 untouched. Consolidating reusable cross-robot definitions remains follow-up work.

## Rollout result

The run used a `0.002 s` timestep and lasted `18.8 s` simulation time (`40.27 s` wall time). The ordered phases were approach/preshape `0-1 s`, finger close `1-2 s`, closed-hand approach `2-3 s`, grasp hold `3-4 s`, lift `4-6 s`, transfer `6-11.6 s`, lower `11.6-13.6 s`, release `13.6-14.6 s`, retreat `14.6-15.8 s`, and free-physics settle `15.8-18.8 s`.

| Gate or measure | Result |
| --- | --- |
| Open-hand approach pushes bottle | No; no hand/bottle contact before closure, `0.186 mm` bottle drift |
| Right-hand/table contact | None during the rollout |
| First right-hand/bottle contact | `2.968 s`, `R_middle_pip` |
| Grasp-hold digit families | Middle only (`R_middle_pip`); thumb + two other families gate failed |
| Lift phase contact | Middle only for `91.6%` of lift steps; no three-family carry |
| Transfer contact | Four families appeared intermittently, but contact covered only `28.4%` of transfer steps and was lost |
| Maximum bottle lift | `0.00496 m` (`4.96 mm`; required `0.05 m`) |
| Release command | `13.600 s`; no stable grasp had been established |
| Final bottle position | `[0.18077, -0.15039, 0.83495] m`; target XY error `0.12913 m` |
| Target footprint / uprightness | Footprint failed; tilt `1.57079 rad` |
| Free-physics interval | `3.0 s`; not settled: linear speed `0.0331 m/s`, angular speed `0.9452 rad/s` |
| Maximum penetration | `0.00896 m` |
| Maximum per-step bottle translation / rotation | `0.00237 m` / `0.01862 rad` |
| Opposite-hand contact | None |
| Runtime weld/equality / bottle qpos writes during rollout | None / `0` |

The bottle began touching its source table at `0.004 s`. Target-table contact first occurred at `8.142 s`, after the bottle had slipped and fallen; it does not count as successful placement. The bottle remained a free MuJoCo body, with continuous per-step state evolution. No weld/equality carry, mocap attachment, hidden support, teleport, or left-hand assistance was used.

The corrected pregrasp is visually readable and the open approach is physically clear, but the dynamic grasp pose settles about `7.2 mm` below the requested palm height and about `2.1 mm` farther from the bottle along the approach axis. The grasp-hold contact evidence is still only the middle PIP. A small next discriminating experiment is to raise only the closed-grasp palm Z target by the measured `7 mm`, leaving X/Y, wrist rotation, finger commands, and station fixed; verify whether thumb and two other digit families contact during the hold before attempting to interpret lift.

## Remote closeout

Issue #46 remains open. GitHub reports no open PR with `codex/issue46-x2-single-hand-grasp` as its head. The last remote branch SHA is `f8b1aafbf67ccdc6276bde6f1021046e7b285e72`; the local evidence commits have not reached it. Push over SSH failed with a broken pipe, HTTPS push could not obtain a username, and the in-app GitHub page is signed out. No Issue comment or PR update was possible. Closeout is **BLOCKED** until authenticated GitHub write access is available; the local commits and external raw evidence are preserved.

## Evidence

Final raw run directory:

`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/canonical-scene/20261005-final-visual/`

It contains `result.json`, `runtime_identity.json`, `physics_contact_trace.jsonl`, `run.log`, `experiment_commands.txt`, `m0_x2_grasp.mp4`, overview/front/side/pregrasp/grasp/lift/release PNGs, and frame diagnostics. `before_previous_station.png` is copied unchanged from the earlier station attempt for comparison. Raw run files were not edited.

Reproduction from the RobotSim checkout:

```bash
ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/canonical-scene/20261005-final-visual ./scripts/run_issue46_x2_grasp.sh
```

The physical grasp, required lift, target footprint, and stable-placement gates failed. This is an evidence-bearing FAIL, not a PASS or maintainer visual-review-ready result.
