# RobotSim Issue #46: X2 Single-Hand Bottle Grasp

**Outcome: FAIL. The corrected station and collision-free pregrasp pass, but the open-hand approach pushes the bottle away before closure. No physical grasp, 5 cm lift, transfer, or stable placement was achieved. This is not ready for X2 single-hand visual review.**

## Scope and provenance

This scratch prototype converts the official X2 Ultra OmniHand URDF into a MuJoCo model and uses a fixed base, ground-truth bottle initialization, offline damped-least-squares arm IK, and deterministic joint position servos. The bottle remains a free MuJoCo body throughout rollout. Only the right hand is commanded; no left-hand bottle contacts were observed.

- Source: `https://github.com/AgibotTech/agibot_x2_urdf`
- Pinned source commit: `575cc6b988f976c23550e0db85aa1e5475d3652d`
- URDF: `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf`
- URDF SHA256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`
- License: Mulan PSL v2; source `LICENSE` SHA256: `d5777bea50dc2cd111845d8a9c51ab990d188ce343b843e5ad3e653b5fa8fdab`
- Referenced mesh assets: 51; individual hashes are in the external `runtime_identity.json` files.
- No vendor asset was copied into RobotSim or modified in the pinned vendor checkout.

The earlier full-close run used RobotSim harness commit `7b9c80d9bc7754fd748aed5e9648462ebdfab8a5` and the original station. The current corrected-station run used WSL2 Linux `6.6.87.2-microsoft-standard-WSL2`, Python 3.10.12, MuJoCo 3.3.6, NumPy 1.26.4, `MUJOCO_GL=egl`, and RobotSim commit `17c1560f5c8efbd456c5d16ea2c46fedd29f701b`.

## Reproduction

From the RobotSim checkout:

```bash
./scripts/run_issue46_x2_grasp.sh
```

The launcher verifies/fetches the exact vendor pin, creates or uses an isolated Python 3.10 virtual environment under `/tmp`, installs pinned MuJoCo and supporting Python packages, and creates a new timestamped evidence directory. The command for the current run was:

```bash
ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/scene-frame-correction/20261005-051200-current-head ./scripts/run_issue46_x2_grasp.sh
```

The launcher accepts `ISSUE46_FINGER_CLOSE_FRACTION` in `(0, 1]` for controlled scratch trials. The default is `0.45`; `1.0` was also tested. The exact commands and runtime identities are preserved with each run.

## Corrected station and frame

The corrected fixed base is at `[0.443, 0.323, 0.68] m`, yaw `-pi/2`, with its forward axis aligned to world `-Y`. The source table and bottle are on that forward axis; the active shoulder is near the bottle X centerline, so the arm no longer reaches across its torso. The target table remains at the G1 M0 target location.

The scratch scene uses the same generated box-table collision geometry and bottle primitive specification as the G1 M0 builder, `simulation/mujoco/m0_pick_place.py`: source top center XY `[0.30, 0.00] m`, half-extents `[0.20, 0.12] m`; target body center XY `[0.30, -0.24] m`, half-extents `[0.20, 0.10] m`; table top `0.80 m`; target marker offset `0.04 m`. The bottle is the same four-cylinder collision model, 65 mm body diameter, 231 mm total height, and 0.50 kg total mass. These are procedural MuJoCo geoms in the scratch script; no reusable standalone shared table or bottle asset was present in the inspected G1 M0 builder, so no shared infrastructure was added.

The old station faced world `+X` while the canonical manipulation axis was world `-Y`, creating a lateral reach. The new base yaw and placement put the table directly in front of X2. No locomotion or standing controller was changed.

The official URDF `R_palm_joint` fixed transform was not missing or misconverted. The compiled palm site is attached to `right_wrist_roll_link` at the URDF origin with position error `0 m` and quaternion absolute-dot agreement `1.0`. The prior task convention instead treated the wrist-roll origin as the palm frame and added an undocumented 180-degree yaw. The corrected task frame maps palm-local `+Z` (palm normal and finger extension) to world `-Y`, local `+Y` (digit row and flexion axis) to world `+Z`, and local `+X` to world `+X`. Thus PIP/DIP flexion sweeps around the bottle's horizontal cross-section. The fix is in the scratch station and task-level end-effector convention, not the vendor URDF or converter.

The initial pregrasp target is `[0.30, 0.24, 0.878] m`; measured palm position error is `1.51e-15 m`, rotation error `0 rad`, and elbow angle `-2.142 rad`. At initialization there are no right-hand/bottle, opposite-hand/bottle, or robot/table contacts. This establishes a collision-free pregrasp before finger closure.

The scene now has neutral bright lighting, a visible non-colliding floor grid, and repeatable overview/close-up cameras. The floor grid has collision disabled. The external run includes old-station before frame, corrected overview, pregrasp and close-up, open-approach, grasp, lift, and release/settle images.

## Current corrected-station rollout

The current run is `FAIL`. First right-hand contact occurred at `0.640 s` during the open approach, on `R_middle_dip`, `R_ring_dip`, and `R_pinky_dip`. Those extended distal links push the bottle during approach; by `2.0 s` its center is `[0.30492, -0.11907, 0.83182] m` and it is tipped. At the finger-wrap/grasp gate there are no bottle contacts, so the commanded lift does not carry it.

| Measure | Current run |
| --- | ---: |
| Valid collision-free initial pregrasp | Pass |
| Active digit families at grasp | 0 |
| Maximum lift | 0.00518 m (required 0.05 m) |
| Final bottle position | `[0.18023, -0.12472, 0.83182] m` |
| Final XY error from target | 0.14147 m |
| Final footprint inside target table | Fail |
| Final upright tilt | 1.57053 rad |
| Maximum penetration | 0.00940 m |
| Maximum bottle translation / rotation per physics step | 0.00168 m / 0.01978 rad |
| Opposite-hand contact | None |
| Equality constraints / rollout bottle qpos writes | 0 / 0 |
| Free-physics phase | 3.0 s, but not settled (linear velocity norm about 0.0363 m/s; angular velocity norm about 1.13 rad/s) |

The bottle remained a free dynamic body throughout. First table contact was `0.004 s`; first target-table contact was `1.562 s` after the bottle had already been displaced. That contact is not evidence of a successful placement. No weld, equality carry, mocap attachment, opposite-hand assistance, or rollout qpos teleport was used.

The next experiment should change one manipulation variable: introduce a measured open-hand preshape before the same approach path, then compare first-contact time, bottle displacement, and digit-family contacts. Do not change station pose, wrist frame, friction, and gains together. The present evidence does not yet establish a physical grasp.

## Historical baseline: original station and frame attempts

The test bottle is a 70 mm diameter, 244.5 mm total-height, 0.57 kg free rigid body (estimated internal body capacity 0.597 L). The MuJoCo model uses timestep 0.002 s, `nq=70`, `nv=69`, `nu=63`, 65 bodies, 64 joints, 150 geoms, and 51 meshes.

Both attempts had zero equality constraints, no runtime weld, and zero bottle `qpos` assignments during the rollout. Maximum per-step bottle displacement remained below 0.41 mm, so the logged trajectory was advanced continuously by MuJoCo. The opposite hand had no contact. The largest measured penetration was 7.11 mm in the default run and 9.09 mm in the full-close run.

## Historical results before station/frame correction

| Gate or measure | Default closure 0.45 | Full closure 1.0 |
| --- | ---: | ---: |
| Active digit families at grasp | Thumb + index (2) | Thumb + index (2) |
| At least three digit families during lift and transfer | Fail | Fail |
| Lift from initial bottle height | 0.00343 m | 0.00411 m |
| Required lift | 0.05 m | 0.05 m |
| Final XY error from target | 0.11512 m | 0.10943 m |
| Bottle footprint inside target table | Pass | Pass |
| Opposite-hand contact | None | None |
| Runtime weld/equality | None | None |
| Bottle `qpos` writes during rollout | 0 | 0 |
| Maximum penetration | 0.00711 m | 0.00909 m |
| Free settling phase | 3.0 s, table-supported | 3.0 s, table-supported |

The higher closure setting increased peak lift by less than 1 mm but did not create additional digit-family contact or a carry. The bottle was pushed from its initial pose: default final position `[0.05854, -0.36881, 0.59650] m`, full-close final position `[0.06601, -0.37407, 0.59650] m`; target position is `[0.17, -0.34] m` in XY. Both final bottle footprints were on the table, but neither met the 5 cm target-position tolerance.

The recorded stage schedule is: approach `0.0-6.4 s`; thumb pre-grasp `6.4-7.4 s`; finger wrap `7.4-9.6 s`; grasp hold `9.6-10.6 s`; lift `10.6-12.6 s`; transfer `12.6-16.8 s`; lower `16.8-18.8 s`; open/release `18.8-19.8 s`; retreat `19.8-21.0 s`; free physics `21.0-24.0 s`. First right-hand contact occurred at 2.948 s and first table contact at 0.004 s. No opposite-hand contact occurred.

The 3 s free-physics phase left the bottle table-supported and clear of the hand, but residual velocity was not near zero (default linear/angular velocity approximately `[-0.00117, -0.00485, 0.00002] m/s` and `[0.0586, -0.0220, 0.0481] rad/s`). This does not establish a stable final placement.

## Evidence

External evidence directory:

`/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/`

The current corrected-station run is in `scene-frame-correction/20261005-051200-current-head/` and contains `result.json`, `runtime_identity.json`, `physics_contact_trace.jsonl` (12,000 physics steps), `m0_x2_grasp.mp4`, `run.log`, `experiment_commands.txt`, frame diagnostics, overview screenshots, and close-up pregrasp/approach/grasp/lift/release screenshots. `before_station_correction_frame0000.png` is frame 0 extracted from the prior root `m0_x2_grasp.mp4` (source MP4 SHA256 `82c025e11340eb18b0a658026e984abce2569fde9828087ac69d5e5720fc2685`); the derived PNG SHA256 is `69bcf75f52f4e6be8d61c198aa181d1f743f9b9fb925b7476006e642902e0016`. The original run evidence remains unchanged. Historical default/full-close artifacts remain at the evidence directory root and `attempts/committed-wrist180-close100/`.

The evidence demonstrates a functioning conversion, rendering, contact trace, free-object dynamics, and no-teleport/no-weld test harness. It does not demonstrate a credible grasp, the required lift, transfer, target placement, or stable settling. Further hand pose/contact-geometry and controller work is required before this can be considered an X2 manipulation implementation.
