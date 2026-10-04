# RobotSim Issue #46: X2 Single-Hand Bottle Grasp

**Outcome: FAIL. The physical single-hand grasp acceptance gates did not pass. This is not ready for X2 single-hand visual review.**

## Scope and provenance

This scratch prototype converts the official X2 Ultra OmniHand URDF into a MuJoCo model and uses a fixed base, ground-truth bottle initialization, offline damped-least-squares arm IK, and deterministic joint position servos. The bottle remains a free MuJoCo body throughout rollout. Only the right hand is commanded; no left-hand bottle contacts were observed.

- Source: `https://github.com/AgibotTech/agibot_x2_urdf`
- Pinned source commit: `575cc6b988f976c23550e0db85aa1e5475d3652d`
- URDF: `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf`
- URDF SHA256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`
- License: Mulan PSL v2; source `LICENSE` SHA256: `d5777bea50dc2cd111845d8a9c51ab990d188ce343b843e5ad3e653b5fa8fdab`
- Referenced mesh assets: 51; individual hashes are in the external `runtime_identity.json` files.
- No vendor asset was copied into RobotSim or modified in the pinned vendor checkout.

Runtime identity for the clean full-close run: WSL2 Linux `6.6.87.2-microsoft-standard-WSL2`, Python 3.10.12, MuJoCo 3.3.6, NumPy 1.26.4, and `MUJOCO_GL=egl`. The RobotSim harness commit for that run was `7b9c80d9bc7754fd748aed5e9648462ebdfab8a5`; its worktree was clean.

## Reproduction

From the RobotSim checkout:

```bash
./scripts/run_issue46_x2_grasp.sh
```

The launcher verifies/fetches the exact vendor pin, creates or uses an isolated Python 3.10 virtual environment under `/tmp`, installs pinned MuJoCo and supporting Python packages, and writes evidence to `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/`.

The launcher accepts `ISSUE46_FINGER_CLOSE_FRACTION` in `(0, 1]` for controlled scratch trials. The default is `0.45`; `1.0` was also tested. The exact commands and runtime identities are preserved with each run.

## Test object and physics constraints

The test bottle is a 70 mm diameter, 244.5 mm total-height, 0.57 kg free rigid body (estimated internal body capacity 0.597 L). The MuJoCo model uses timestep 0.002 s, `nq=70`, `nv=69`, `nu=63`, 65 bodies, 64 joints, 150 geoms, and 51 meshes.

Both attempts had zero equality constraints, no runtime weld, and zero bottle `qpos` assignments during the rollout. Maximum per-step bottle displacement remained below 0.41 mm, so the logged trajectory was advanced continuously by MuJoCo. The opposite hand had no contact. The largest measured penetration was 7.11 mm in the default run and 9.09 mm in the full-close run.

## Results

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

The directory root contains the complete default-closure run: `result.json`, `runtime_identity.json`, `physics_contact_trace.jsonl` (12,000 physics steps), `m0_x2_grasp.mp4` (600 decoded frames), `final.png`, `run.log`, and `experiment_commands.txt`. The clean full-closure comparison is under `attempts/committed-wrist180-close100/` with the same artifact set. A pre-commit parameter trial is kept separately under `attempts/wrist180-close100/` and is not the clean-head comparison.

The evidence demonstrates a functioning conversion, rendering, contact trace, free-object dynamics, and no-teleport/no-weld test harness. It does not demonstrate a credible grasp, the required lift, transfer, target placement, or stable settling. Further hand pose/contact-geometry and controller work is required before this can be considered an X2 manipulation implementation.
