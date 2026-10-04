# RobotSim Issue #43 — M0 License and Provenance Audit

Audit date: 2026-10-04. Read-only source review of:

- [NVlabs/humanoidmimicgen](https://github.com/NVlabs/humanoidmimicgen/tree/d82844dcec242c82d6b82628ccc45933c3ad5cbd), commit d82844dcec242c82d6b82628ccc45933c3ad5cbd
- [lwm97/pickandplaceunitreeg1](https://github.com/lwm97/pickandplaceunitreeg1/tree/b07678b131885c55175e23d2342548263eaeb9b0), commit b07678b131885c55175e23d2342548263eaeb9b0

This records repository license/provenance evidence, not a legal opinion or guarantee. Categories: **A** copy supported by explicit license evidence; **B** keep as an external pinned checkout or download; **C** preserve attribution/license notices; **D** unclear/restricted pending resolution.

## Findings

### HMG code — A + C

The pinned repository root [LICENSE](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/LICENSE) is Apache-2.0 and [NOTICE](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/NOTICE) identifies NVIDIA copyright. NVIDIA-authored task/controller code can be copied under Apache-2.0 conditions.

HMG separately says humanoidmimicgen/locomanipulation contains RoboCasa-derived source under MIT, Copyright © 2024 The RoboCasa Team, and that mixed files carry stacked MIT/Apache headers. See [THIRD_PARTY_NOTICES.md](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/THIRD_PARTY_NOTICES.md) and LICENSES/ROBOCASA-MIT.txt. Copy only selected files and preserve their actual per-file SPDX/copyright headers; do not relicense all of locomanipulation as Apache.

**Can RobotSim copy HMG task/controller code?** Yes, on the source's stated terms: preserve Apache-2.0 and NVIDIA notices for NVIDIA code, and MIT/RoboCasa notices for derived files. HMG does not record an immutable RoboCasa source commit, so keep the HMG commit as RobotSim's source pin and avoid claiming stronger upstream file provenance than it records.

### HMG assets — C for bounded subsets; D for task G1 MJCF/hand bundle pending provenance clarification

HMG's [third-party inventory](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/THIRD_PARTY_NOTICES.md) labels humanoidmimicgen/locomanipulation/models/assets as RoboCasa assets under MIT (Copyright © 2024 The RoboCasa Team). LICENSES/ROBOCASA-MIT.txt is included. The repository also pins robosuite==1.5.1 and robosuite-models==1.0.0; the included LICENSES/ROBOSUITE-MIT.txt and LICENSES/ROBOSUITE-MODELS-MIT.txt cover those declared runtime components. Keep these packages externally installed at the selected versions rather than copying their trees.

There is an asset provenance overlap that the inventory does not explain: HMG's task G1 path contains g1_29dof_rev_1_0.xml, separate g1_threefinger_left_hand.xml and g1_threefinger_right_hand.xml, and 51 STL files. The 51 STLs are byte-identical to same-named files at Unitree unitree_ros commit f3772ce54c56ef2d34c6aee8100bc768896c7d19; HMG's own Unitree inventory identifies a separate URDF/mesh tree as Unitree BSD-3-Clause. The task MJCF is transformed rather than byte-identical to Unitree's similarly named hand model, and the two separated hand XML files have no specific source revision recorded. The blanket RoboCasa/MIT asset entry does not resolve the underlying Unitree provenance for this subset.

Therefore, **do not copy HMG's task G1 MJCF or hand XML/mesh bundle into RobotSim until HMG's per-file license/provenance mapping is confirmed**. Prefer RobotSim's already pinned G1 model checkout for M0. The separately inventoried HMG wbc/robot_model/model_data/g1 bundle is an URDF plus 50 meshes (not MJCF): HMG says its meshes are byte-identical to Unitree's pinned unitree_ros commit and includes LICENSES/UNITREE-BSD-3-CLAUSE.txt, Copyright © 2016–2022 Unitree Robotics. That bundle has explicit BSD evidence (**C**) if actually needed; preserve the BSD license and the URDF's stacked Unitree BSD / NVIDIA Apache notices.

HMG also records README images under CC BY-SA 4.0. Do not reuse its teaser/task-card media unless preserving CC BY-SA attribution and share-alike conditions.

### HMG WBC weights — C, redistribution expressly conditioned by NVIDIA Open Model License

HMG does not store the optional weights. [download_wbc_policies.py](https://github.com/NVlabs/humanoidmimicgen/blob/d82844dcec242c82d6b82628ccc45933c3ad5cbd/humanoidmimicgen/download_wbc_policies.py) fetches from NVlabs/GR00T-WholeBodyControl commit 4141c34280abb67c82e115342a8720f4a83d750d and checks SHA-256:

- GR00T-WholeBodyControl-Balance.onnx → f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18
- GR00T-WholeBodyControl-Walk.onnx → 7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407

The exact upstream commit contains Git LFS pointers with those same artifact hashes. Its [LICENSE](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/4141c34280abb67c82e115342a8720f4a83d750d/LICENSE) says source code is Apache-2.0 but model weights are under the NVIDIA Open Model License. HMG also includes LICENSES/NVIDIA-OPEN-MODEL-LICENSE.txt and states the required attribution: **“Licensed by NVIDIA Corporation under the NVIDIA Open Model License.”**

**Can RobotSim redistribute the WBC weights?** The repository license evidence permits reproduction/distribution subject to the N-OML: include a copy of that agreement and the attribution, and comply with its Trustworthy AI terms and other conditions. They must not be treated as Apache-licensed source. Prefer fetching into an ignored external cache by immutable URL and verifying the published SHA-256; do not commit them by default.

### HMG datasets — D

No HDF5, ONNX, checkpoint, or training dataset artifact is tracked at the audited HMG commit. The README points to public Hugging Face repositories:

- linkenv/humanoidmimicgen-g1-source-demo-replay (human source-demo HDF5)
- linkenv/humanoidmimicgen-g1-nine-task-action-replay
- linkenv/humanoidmimicgen-g1-benchmark (about 9K generated demonstrations)

The README documents download commands, but the audited source tree does not record immutable dataset revisions/content hashes or dataset-specific license files/terms. HMG's root Apache license does not establish rights for separately hosted datasets. The Hugging Face endpoint was inaccessible under this audit environment's restricted network policy, so dataset cards/terms could not be independently checked.

**Can RobotSim redistribute HMG demo HDF5/data?** Not established. Do not vendor, mirror, or redistribute any of these datasets until each dataset's license, provenance, and exact revision are recorded. Local external download/use should also wait for the dataset-specific terms to be checked.

### LWM code — A + C

The selected LWM commit's root [LICENSE](https://github.com/lwm97/pickandplaceunitreeg1/blob/b07678b131885c55175e23d2342548263eaeb9b0/LICENSE) is MIT, Copyright © 2026 lwm97. The self-contained CV/IK/tactile controller logic (notably fastwam_policy.py and authored portions of lightwam_policy.py) can be copied with that copyright and permission notice.

The optional legacy Light-WAM loader imports an external lightwam package/tree and accepts external checkpoint/statistics paths. That package and those artifacts are not present or pinned in this repository; the LWM MIT file does not establish their license. Do not copy/use those external components until their own origin and license are established.

### LWM assets — C; source pin should be strengthened before vendoring

LWM's [NOTICE](https://github.com/lwm97/pickandplaceunitreeg1/blob/b07678b131885c55175e23d2342548263eaeb9b0/NOTICE) explicitly says root assets/ and g1_robot.xml are derived from Unitree G1 and redistributed under BSD-3-Clause. It points to third_party/unitree_g1/LICENSE and README; the license carries Unitree's copyright and standard source/binary conditions. This is separate from the root MIT code license.

The 49 STL files in the LWM checkout match the current MuJoCo Menagerie G1 asset files byte-for-byte; the upstream asset content dates back to Menagerie commit b5ff2e8666c2fa36fddd0961e5bd992726c1efc1. LWM's own README/NOTICE refer to Menagerie main and Unitree ROS master, not an immutable upstream revision; its g1_robot.xml is a modified derivative, not byte-identical to Menagerie's XML. The LWM repository itself is pinned by the task and contains the relevant BSD license evidence, but record an immutable upstream source revision before importing these assets into RobotSim.

**Can RobotSim copy LWM G1/hand assets?** The LWM package explicitly states BSD-3-Clause for these files, so repository evidence supports conditional copying if the Unitree copyright, conditions, disclaimer, NOTICE, and BSD license are preserved. Do not use the LWM root MIT license as the asset license. Prefer an external checkout at the exact LWM commit until the upstream asset pin/provenance is added.

## Notices to preserve if reuse is approved

- HMG Apache code: root LICENSE, root NOTICE, and relevant per-file SPDX headers.
- HMG RoboCasa-derived code/assets: LICENSES/ROBOCASA-MIT.txt, RoboCasa copyright notices, and stacked per-file headers; HMG THIRD_PARTY_NOTICES.md.
- HMG RoboSuite packages/models, if redistributed rather than installed externally: corresponding LICENSES/ROBOSUITE-MIT.txt and LICENSES/ROBOSUITE-MODELS-MIT.txt.
- Unitree G1 assets: HMG LICENSES/UNITREE-BSD-3-CLAUSE.txt or LWM third_party/unitree_g1/LICENSE, plus source NOTICE/provenance.
- HMG WBC model weights: exact NVIDIA Open Model License text and the required “Licensed by NVIDIA Corporation under the NVIDIA Open Model License” attribution.
- LWM controller source: LWM root LICENSE (MIT).
- HMG copied README media, if any: CC BY-SA 4.0 attribution/license and share-alike notices.

## Recommended M0 reuse boundary

1. Selectively reuse HMG task/controller source under its Apache/MIT file-level terms; record the HMG commit as the immutable source.
2. Keep HMG's pinned RoboSuite packages externally installed; include their license notices if a distribution bundles them.
3. Use RobotSim's existing pinned G1 model as the M0 source of truth. Do not copy HMG's overlapping task G1 MJCF/hand asset tree until its dual RoboCasa/Unitree provenance is resolved.
4. If HMG's WBC is needed, download the two ONNX files only into an ignored external cache from the pinned GR00T commit, verify exact SHA-256, and distribute only with N-OML terms and attribution.
5. Do not include HMG HDF5 or training data in RobotSim. Resolve the HF dataset card terms and pin exact dataset revisions before any download intended for redistribution.
6. LWM controller logic is a small MIT-reusable alternative. Keep its G1 model/hand assets in an exact external checkout unless needed; if imported, preserve the separate BSD notice/license and record an immutable upstream asset revision.
7. Do not fetch or vendor the optional LWM Light-WAM package/checkpoints/statistics until their separate provenance/license is documented.

RobotSim's existing [third_party/LOCK.md](https://github.com/lzy18001500226/RobotSim/blob/main/third_party/LOCK.md) already cautions that a repository-level license may not settle rights for every G1 model/mesh asset and says not to copy/redistribute vendor assets absent asset-specific confirmation. This audit supports the above conditional reuse boundaries and identifies the HMG task MJCF/hand overlap and HF data terms as unresolved.