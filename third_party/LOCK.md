# Third-party Version Lock

Reviewed on 2026-10-03. These full commit IDs are the source of truth for the
Unitree source checkouts used by the first G1 baseline. The fetch script reads
the three Unitree rows below; it checks out detached at each exact commit and
refuses to change a checkout with local modifications or a different origin.

Run `./scripts/fetch_third_party.sh` to populate `third_party/`. For clean
validation or an alternate checkout location, pass a destination directory.
The script fetches Git source only. It does not download or commit vendor
binaries.

| Project | Repository | Commit | Commit date | License reference |
|---|---|---|---|---|
| unitree_sdk2 | https://github.com/unitreerobotics/unitree_sdk2.git | 63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36 | 2026-09-21 | [BSD 3-Clause `LICENSE`](https://github.com/unitreerobotics/unitree_sdk2/blob/63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36/LICENSE) |
| unitree_mujoco | https://github.com/unitreerobotics/unitree_mujoco.git | 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d | 2026-09-07 | [BSD 3-Clause `LICENSE`](https://github.com/unitreerobotics/unitree_mujoco/blob/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/LICENSE) |
| unitree_ros2 | https://github.com/unitreerobotics/unitree_ros2.git | 668d1ec5a05d1c38d3306bdca7d59f2ba3581a88 | 2026-07-02 | [BSD 3-Clause `LICENSE`](https://github.com/unitreerobotics/unitree_ros2/blob/668d1ec5a05d1c38d3306bdca7d59f2ba3581a88/LICENSE) |

## MuJoCo baseline and provenance

- Baseline: MuJoCo 3.3.6, matching the default `MUJOCO_VERSION` in
  [`docker/Dockerfile.dev`](../docker/Dockerfile.dev).
- Upstream source: [`google-deepmind/mujoco` tag `3.3.6`](https://github.com/google-deepmind/mujoco/tree/3.3.6), resolving to commit
  `eacad44a1a67afe520b263c9b15dab82f62a10aa` (2025-09-15).
- License reference: [Apache License 2.0 at that source commit](https://github.com/google-deepmind/mujoco/blob/eacad44a1a67afe520b263c9b15dab82f62a10aa/LICENSE).
- The development image downloads the official 3.3.6 Linux x86-64 release
  archive from
  `https://github.com/google-deepmind/mujoco/releases/download/3.3.6/mujoco-3.3.6-linux-x86_64.tar.gz`.
  It is an image build artifact, not a repository dependency checkout or a
  file to commit.

## G1 model provenance and asset handling

The pinned `unitree_mujoco` source contains the G1 MJCF entries
`unitree_robots/g1/g1_23dof.xml` and `unitree_robots/g1/g1_29dof.xml`, with
associated meshes under `unitree_robots/g1/meshes/`. Their source provenance
is the immutable [`unitree_mujoco` commit](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/g1).
The upstream repository has a root BSD 3-Clause `LICENSE`; the SDK repository
also carries separate third-party license files under `licenses/` and
`thirdparty/`. Retain upstream notices when using those sources.

The repository-level license notices do not establish separate redistribution
rights for every G1 model or mesh asset. Keep vendor files in the fetched,
ignored checkout; do not copy or redistribute them in RobotSim without
confirming permission for the specific assets.
