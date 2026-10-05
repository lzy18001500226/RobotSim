# RobotSim Issue #46: X2 Single-Hand Bottle Grasp

**Physical manipulation: FAIL. Static scene review: READY FOR MAINTAINER VISUAL REVIEW.** This update corrects only the static station and rendering. The right hand still has not established a stable multi-finger grasp, lifted the required 50 mm, or placed and settled the bottle at the target.

## Source and runtime

- Vendor: [AgibotTech/agibot_x2_urdf](https://github.com/AgibotTech/agibot_x2_urdf), exact commit `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Model: `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf`; SHA256 `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.
- License: Mulan PSL v2; pinned `LICENSE` SHA256 `d5777bea50dc2cd111845d8a9c51ab990d188ce343b843e5ad3e653b5fa8fdab`.
- Runtime: WSL2 Ubuntu 22.04, Python 3.10.12, MuJoCo 3.3.6, NumPy 1.26.4, `MUJOCO_GL=egl`.
- RobotSim harness commit tested: `b62d63174950cef9e1dd8a24da22348c5077b096`. The run identity marks the worktree dirty because four pre-existing untracked `debug_issue46_*.py` scratch files were present; they were not staged or changed. The tracked harness files were at the recorded commit.
- The vendor checkout remained clean. No vendor asset was copied into RobotSim or edited.

## Static visual baseline for maintainer review

The 2026-10-05 static scene review is under `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/static-scene/20261005-final-review/`. It includes the unchanged `before_overview.png`, canonical three-quarter overview, front view, side-alignment view, hand/material close-up, `static_scene.json`, `run.log`, and exact commands.

The X2 scene now calls `add_g1_canonical_table()` from `simulation/mujoco/canonical_manipulation_assets.py`. Its tabletop, four legs, and target site match the accepted G1 XML: table center `[0.300, -0.100] m`, top z `0.800 m`, tabletop half-extents `[0.200, 0.200] m`, 25 mm legs at x/y offsets `+/-0.175 m`, and brown RGBA `[0.6, 0.4, 0.2, 1]`. The G1 reference XML SHA256 is recorded in the JSON. G1 PR #45 is unchanged; migrating its scene to the shared helper remains future work.

The four-part X2 bottle definition is preserved as the reusable canonical X2 bottle in the same module: blue body, shoulder, neck, and dark cap; 70 mm diameter, 244.5 mm height, and 0.57 kg. It has not been replaced by the pure-cylinder benchmark object. Updating G1 to consume this shared bottle definition is future work.

The pinned vendor URDF's authored white body colors and dark detail colors remain intact. The importer drops name-only OmniHand materials, so the renderer maps the symbolic `silver`, `blue`, `brown`, `white`, `green`, and `orange` names to a documented color palette after model compilation; the URDF provides no numeric RGB values for those named materials, so those six shades are an approximation rather than source-exact RGB. This replaces the earlier uniform gray fallback. The original vendor meshes and black chest detail remain visible. Lighting uses the accepted G1 headlight values (ambient 0.3, diffuse 0.6, specular 0), matching haze and sky gradient, plus a downward directional key light. The checker texture washed out in this EGL view, so the floor is a neutral matte RGBA `[0.42, 0.45, 0.48, 1]`.

X2 starts directly at the station at `[0.38, 0.32, 0.68] m`, yaw `-pi/2`, facing world `-Y`. The G1 table center is directly ahead with an 80 mm lateral offset toward the right-arm workspace. The overview and side image show a forward-facing torso and a naturally bent right-arm reach; walking is not used.

The static hand pose copies the recorded `approach_preshape` arm and finger joint state from the prior run. The renderer calls `mj_forward` only: `mj_step` is not called, the bottle qpos is unchanged, and there is no hand/bottle or hand/table contact in the rendered state. This is a frozen scene pose, not new grasp evidence or a manipulation PASS.

Reproduction command:

```bash
MUJOCO_GL=egl ISSUE46_STATIC_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/static-scene/20261005-final-review /tmp/robotsim-issue46-x2-prototype-575cc6b9/.venv/bin/python scripts/research/issue46_x2_static_scene.py
```

## Earlier rollout scene and appearance (historical)

The physical trial below used the earlier source/target table arrangement: top height `0.80 m`, source center `[0.30, 0.00] m`, source half-extents `[0.20, 0.12] m`, target center `[0.30, -0.24] m`, target half-extents `[0.20, 0.10] m`, `0.02 m` gap, and `0.04 m` target-marker offset. The object was the current X2 bottle: `70 mm` body diameter, `244.5 mm` total height, `0.57 kg`; the previous G1 bottle was not used.

In that historical run, the converter preserved the 40 inline white or black/translucent URDF colors and vendor meshes, including the black chest logo, but reduced name-only OmniHand materials to its gray default. It then used 34 thin, collision-disabled floor-grid geoms and reduced ambient and diffuse while setting specular to `0.45`. Those settings produced the earlier gray/dark appearance and are superseded by the static visual baseline above.

The X2 fixed base is `[0.38, 0.32, 0.68] m`, yaw `-pi/2`, so its forward axis is world `-Y` and its torso faces the canonical table edge. Walking is not used. The table and bottle lie in front of the robot and within the right arm workspace. The palm frame comes from the vendor `R_palm_joint`; its pregrasp normal is approximately world `-Y`. Measured pregrasp position error is `8.70 mm`, rotation error `0.032 rad`; no preclosure bottle contact, right-hand/table contact, or opposite-hand contact occurred. Bottle translation before closure was `0.186 mm`.

That rollout still used a duplicated two-table layout. The static-scene update above consolidates the X2 table and bottle definitions in one reusable helper; G1 PR #45 remains untouched, so migrating its scene to consume the helper is still follow-up work.

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

The earlier closeout attempt was blocked by transient push/authentication failures. On 2026-10-05, the evidence branch was pushed successfully to `codex/issue46-x2-single-hand-grasp`; the final exact HEAD and closeout fields are recorded on Issue #46. The issue remains open and no PR was created. The result is **FAIL**: the evidence does not meet the physical-grasp, 50 mm lift, target-footprint, or stable-placement gates.

## Evidence

Final raw run directory:

`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/canonical-scene/20261005-final-visual/`

It contains `result.json`, `runtime_identity.json`, `physics_contact_trace.jsonl`, `run.log`, `experiment_commands.txt`, `m0_x2_grasp.mp4`, overview/front/side/pregrasp/grasp/lift/release PNGs, and frame diagnostics. `before_previous_station.png` is copied unchanged from the earlier station attempt for comparison. Raw run files were not edited.

Reproduction from the RobotSim checkout:

```bash
ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/canonical-scene/20261005-final-visual ./scripts/run_issue46_x2_grasp.sh
```

The physical grasp, required lift, target footprint, and stable-placement gates failed. This is an evidence-bearing FAIL, not a PASS or maintainer visual-review-ready result.
