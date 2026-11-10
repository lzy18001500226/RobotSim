# AgiBot Community Source Inventory

**Fetched:** 2026-11-03
**Purpose:** read-only source review for RobotSim X2 navigation and
manipulation research. This inventory is provenance, not an approval to run,
install, or integrate an upstream project.

## Collection contract

All public repositories returned by the authenticated GitHub organization query
for `AgiBot-Community` were fetched to the ignored local root
`reference/AgiBot-Community/<RepositoryName>/`. Each checkout is a clean,
detached commit. Fetching used `--no-tags --depth=1` and
`GIT_LFS_SKIP_SMUDGE=1`; no submodules were initialized and no upstream
scripts, installers, or binaries were executed. All eight snapshots contained
zero Gitlink entries at the pinned commit.

`NOASSERTION` below means GitHub's repository metadata did not declare an SPDX
license. It does **not** grant any right to redistribute or execute the
contents; any contained asset or binary needs its own license review before
use.

| Repository | Local immutable pin | License metadata and file evidence | Initial relevance and review status |
| --- | --- | --- | --- |
| [`.github`](https://github.com/AgiBot-Community/.github/tree/87f467dbe9ff402b336eeff51ae8934c819bb18b) | `87f467dbe9ff402b336eeff51ae8934c819bb18b` | `NOASSERTION`; no root license file | Community profile only; no RobotSim runtime role. Collected for completeness, not scheduled for code integration. |
| [`Agibot-X2-IK-upper-body-control`](https://github.com/AgiBot-Community/Agibot-X2-IK-upper-body-control/tree/444598038cf0e5b27e32e3e5295ab24503ecd43f) | `444598038cf0e5b27e32e3e5295ab24503ecd43f` | `NOASSERTION`; `THIRD_PARTY_NOTICES.txt` only | X2 MoveJ/MDI delivery with a robot-side wheel/runtime and x86 desktop client. Candidate read-only reference for the grasp/upper-body branch; it is not an MC/MuJoCo controller and has not been run. |
| [`agibot-X2-grasp-demo`](https://github.com/AgiBot-Community/agibot-X2-grasp-demo/tree/26d3d45bc466139cbf6a124439e23f3df6eb8d2c) | `26d3d45bc466139cbf6a124439e23f3df6eb8d2c` | `Apache-2.0`; `LICENSE` | X2 grasp/IK reference. Kept separate from the navigation runtime; no gripper mechanics or control code is adopted by this inventory. |
| [`agibot-x2-navigation`](https://github.com/AgiBot-Community/agibot-x2-navigation/tree/9ad1ad8af1b47b1d8fda42c3ca9c828bc5625629) | `9ad1ad8af1b47b1d8fda42c3ca9c828bc5625629` | `MIT`; `LICENSE` | X2 LiDAR/IMU correction, FAST-LIO, NDT-OMP, OctoMap, Nav2 and an AimDK velocity bridge. Already used as a **read-only interface/design reference**; no ROS package is a current RobotSim runtime dependency. |
| [`awesome-aima`](https://github.com/AgiBot-Community/awesome-aima/tree/3f7ae920ed811a541b89add145142055f3690d76) | `3f7ae920ed811a541b89add145142055f3690d76` | `NOASSERTION`; no root license file | Curated source index, not a controller. It is the starting point for selecting further official, version-matched sources; linked projects need independent provenance review. |
| [`raicom2026-agibot-D1`](https://github.com/AgiBot-Community/raicom2026-agibot-D1/tree/b33212377c3ee11498fcf2e4705ee9aa07f47780) | `b33212377c3ee11498fcf2e4705ee9aa07f47780` | `MIT`; `LICENSE` | D1 quadruped competition overlay. It may offer generic ROS task-orchestration patterns, but its model, SDK and Nav2 prerequisites are not X2-compatible. No adoption is planned. |
| [`raicom2026-agibot-X2`](https://github.com/AgiBot-Community/raicom2026-agibot-X2/tree/a97761820b398b6cab12b632ce9f313c394eeb7e) | `a97761820b398b6cab12b632ce9f313c394eeb7e` | `MIT`; `LICENSE` | Closest competition-system reference: coordinator, map/TF pose bridge, navigation-to-AimDK velocity bridge, perception and grasp task layers. Its static interface review is recorded in [RAICOM X2 reuse boundaries](raicom_x2_reuse_interfaces.md); it is not directly executed in the frozen x86 MuJoCo/MC runtime. |
| [`Unity-Agent-Playground`](https://github.com/AgiBot-Community/Unity-Agent-Playground/tree/b85dd74662ab1b5a682cce6422af52c90545e450) | `b85dd74662ab1b5a682cce6422af52c90545e450` | `Apache-2.0`; `LICENSE` | Unity X2 agent playground with WebSocket gateway, skills, voice agent and console. Candidate architecture reference for the future Unity frontend only; it neither validates the current MuJoCo scene nor replaces the vendor MC. |

## Review order and integration boundary

1. Compare `raicom2026-agibot-X2`'s TF/goal/`/cmd_vel`/AimDK bridge with the
   installed x86 MC message and state contract.
2. Compare `agibot-x2-navigation`'s LiDAR-localization and Nav2 assumptions
   against the RobotSim LiDAR frame, map and available runtime dependencies.
3. Use `awesome-aima` only to select additional official primary sources; do
   not bulk-import its linked projects.
4. Review the upper-body IK and grasp-demo sources only in the isolated grasp
   workstream, under the robot-model-integrity contract.
5. Treat the D1 and Unity sources as non-runtime references until a separate,
   version-matched adapter and validation plan are approved.

An upstream checkout being present locally means only that its source can be
inspected reproducibly. Production use requires a RobotSim-owned adapter,
version/interface compatibility review, license review for any redistributed
asset or binary, and separately recorded runtime validation.
