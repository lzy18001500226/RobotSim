# Issue #46: Dual Native OmniHand X2 Platform

## Result

**Morphology and source-to-compiled parity: PASS. Physical single-hand grasp: FAIL at the source mimic gate.** The complete official X2 body, both five-finger OmniHands, their source visual meshes, collision geometry, articulated joints, inertials, and mimic relations compile into the same MuJoCo model. The source-faithful contact probe stops before a grasp hold or lift; this work does not establish physical grasp capability.

The earlier fixed-palm results remain intact in [the loaded mimic gate](issue46_loaded_mimic_gate.md) and [the contact-loaded checkpoint](issue46_contact_loaded_grasp_checkpoint.md). This report does not replace or reinterpret them.

## Provenance

- RobotSim branch: `codex/x2-omnihand-experimental-20261007`.
- Starting RobotSim HEAD: `4424a33d793e0436b480e37869510ece63a4bc95`.
- Official source: `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Official model: `X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf`.
- Source URDF SHA-256: `344c188605f307474456749525259ad8ca2e7b356d9ef3c50ad887f228045259`.
- Runtime: WSL2 x86-64, Python `3.10.12`, MuJoCo `3.3.6`, timestep `0.002 s`.
- The pinned vendor checkout was clean. No upstream vendor file was modified.

## Compiled Morphology

The generated diagnostic URDF sets `discardvisual=false`, preserves mesh paths, and disables static fusion. It is derived under the external evidence directory; the vendor checkout remains unchanged. The generated MJCF includes all original collision geoms separately from non-colliding visual mesh geoms and all 12 source mimic equalities.

| Quantity | Source / compiled |
| --- | ---: |
| Links / bodies | 86 source links / 89 total MuJoCo bodies |
| Joints | 85 source joints: 63 revolute, 22 fixed |
| Robot hinge coordinates | `nq=63`, `nv=63` |
| Total model, including free canonical bottle | `nq=70`, `nv=69`, `nu=51`, `neq=12` |
| Visuals | 86 / 86 source-to-compiled records; 76 mesh visual geoms; 51 unique mesh assets |
| Collisions | 58 / 58 source-to-compiled records |
| Robot mass | `45.1822 kg` from compiled inertials |
| Mimics | 12 / 12 source relations retained |

Each hand has the official palm and 22 finger links. Visual mesh counts per hand are thumb `5`, index `3`, middle `3`, ring `3`, pinky `3`; left and right inventories match their respective official source subtrees. The machine-readable inventory compares link visual/collision shapes and transforms, joint parent/child/type/origin/axis/range, inertial mass/tensor, and mimic driver/multiplier/offset. It reports `status=PASS` with zero validation errors.

The official URDF defines no actuator transmissions. Therefore the diagnostic platform's 51 bounded joint-position PD actuators (`kp=18`, `kv=2.4`) are explicitly **SIMULATION_ONLY**; they are not claimed to represent verified hardware drives. The base is fixed to world exactly as in this official URDF, so these visuals are not a free-standing or balance result.

## Visual Evidence

The saved front/back/side, individual hand, simultaneous both-hand OPEN/CLOSE, collision-only, visual/collision overlay, overhead, table/bottle, and 135-frame dual-hand MP4 are in:

`C:\Users\HP\Desktop\Robot\reviews\issue-x2-dual-omnihand-20261010\morphology-visual-11\`

The MP4 is a render-only kinematic animation: joint coordinates are assigned for frames and `mj_forward()` updates transforms; it does not call `mj_step()` and is not actuator or grasp validation. It demonstrates the official visual meshes moving with their compiled finger and arm kinematic transforms. The table and bottle in this morphology scene are translated together by `[0, +0.70, -0.80] m` only to align the canonical tabletop with the fixed-root X2 preview; their geometry and mass are unchanged. This is a visualization convenience, not a task state.

The bottle/hand frame in `kinematic-bottle-preview-06/right_hand_kinematic_bottle_preview.png` is generated from a separate `MjData` qpos copy and uses no physics steps. To pose the mimic followers for this illustration, the preview-only copy assigns their qpos from the source mimic relations; this is not an active-rollout write. Its identity file explicitly marks `physics_stepped=false` and `contact_or_grasp_evidence=false`. It is only a pose illustration.

## Physical Contact Gate

The bounded probe reuses the existing full-body, source-mimic contact setup. It does not create a bottle weld, mocap attachment, hidden support, or upward force. The bottle remained a free body with its canonical mass and geometry; active-rollout bottle qpos writes were `0`. The opposite/left hand made no bottle contact.

The first right-hand/bottle contact was `R_middle_dip` at global step `678` (`1.356 s`). It occurred during `approach_preshape` for 47 of 475 steps (`9.895%`); no thumb/opposing-finger grasp formed. The first failure was at step `725` (`1.450 s`): `R_middle_dip_joint` source mimic error `0.0033166746 rad`, exceeding the unchanged `0.003 rad` gate. The run stopped there. No `grasp_hold` steps occurred, estimated mean upward support was `0 N` (bottle weight `5.5917 N`), and no lift milestone was attempted. The maximum measured bottle penetration was `0.1908 mm`, and maximum per-step translation was `0.03924 mm`.

The preceding `open_hold` lasted 250 steps but had maximum hand target tracking error `0.14112 rad`; it is not accepted as a stable OPEN gate. This physical probe therefore fails before source-valid loaded closure. The `1 mm`, `5 mm`, `30 mm`, and `50 mm` lift milestones are **DEFERRED**, not failed lift tests.

No further gain search was performed. The next distinct mechanism to investigate, if authorized, is a reduced thumb-plus-opposing-digit actuator/transmission model that preserves the official joint topology and source mimic map, with contact disabled until its loaded mimic residual passes. Any transmission abstraction remains SIMULATION_ONLY unless vendor hardware semantics are independently established. Do not repeat the current full-hand gain adjustment as the next step.

## Reproduction

Run in the existing WSL environment. The morphology output directory must be new or empty:

```bash
cd /home/lzy18001500226/robotsim-x2-omnihand-experimental-20261007
source /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/activate
MUJOCO_GL=egl AGIBOT_X2_VENDOR_ROOT=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf ISSUE46_DUAL_OMNIHAND_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-x2-dual-omnihand-20261010/morphology-visual-11 python scripts/research/issue46_dual_omnihand_platform.py
```

The kinematic-only bottle illustration can be regenerated without stepping physics:

```bash
cd /home/lzy18001500226/robotsim-x2-omnihand-experimental-20261007
source /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/activate
MUJOCO_GL=egl AGIBOT_X2_VENDOR_ROOT=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf python scripts/research/issue46_dual_omnihand_grasp_probe.py --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-x2-dual-omnihand-20261010/kinematic-bottle-preview-06 --kinematic-preview-only
```

The physical run already captured as `physical-probe-02` used this exact command (also saved in its `reproduction_command.txt`):

```bash
MUJOCO_GL=egl AGIBOT_X2_VENDOR_ROOT=/home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf ISSUE46_EVIDENCE_DIR=/mnt/c/Users/HP/Desktop/Robot/reviews/issue-x2-dual-omnihand-20261010/physical-probe-02 ISSUE46_MIMIC_EQ_SOLREF_SCALE=4.0 ISSUE46_MIMIC_DRIVER_GAIN_SCALE=2.0 ISSUE46_APPROACH_FINGER_DURATION_STEPS=1000 /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python /home/lzy18001500226/robotsim-x2-omnihand-experimental-20261007/scripts/research/issue46_dual_omnihand_grasp_probe.py --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-x2-dual-omnihand-20261010/physical-probe-02
```

## Validation and Evidence Hashes

`python -m py_compile scripts/research/issue46_dual_omnihand_platform.py scripts/research/issue46_dual_omnihand_grasp_probe.py tests/test_issue46_dual_omnihand_platform.py` passed. `python -m unittest tests.test_issue46_dual_omnihand_platform -v` passed all 5 tests, including expected rejection of missing visual mesh, missing finger joint, swapped left/right geometry, and a dropped mimic relation. `git diff --check` passed before report finalization.

Selected external artifact SHA-256 values:

- Source-to-compiled inventory: `c2ccab7ca174f8cc6b8f3961384954ce28e3d1603b783bbde0dbe8a8080db63e`.
- Generated augmented URDF: `e89eeb664e804574f2f6ea65b0bbcaf43d77b2f68b27f22118e0e0127a9691ef`.
- Generated MJCF: `575766118e942d69323efcdc9c6cfe82f242850a1f0d80385f980dc0eb027dc9`.
- Dual-hand OPEN/CLOSE MP4: `bca85bd7102d8f12a2017989c86ed14c3798fc976fca9e4c38af8b252e249ab4`.
- Kinematic-only bottle preview: `2803d3f0d6274f7b6054f7113ff1fd92c2a330828ff240a9c4e6c850c23742f0`.
- Kinematic-only preview identity: `7c292e2903fd1d85e62f2b34a65e7dffd000aa7fa05d8f817b267a155f860316`.
- Physical contact trace: `6ba5917abbe873426bada5336f44ead561277605f3d611162251e78ddc0a28d7`.
- Physical result JSON: `8a92b2d13ad059aba9ef729549d578eb3c76b1302fdaa71ae8ec47025892c626`.
- Physical probe MP4: `e7b400b6823d74da043ed5ae8d535350264563d900c4ce7d0b58245d04ea05fa`.
- Physical run script hash recorded at runtime: `7e2c833c160bb1dc92b237cd4e4577d3c71ce0f70ccf67e286091f4f1aa5a889`.
- Current dual-hand platform harness: `febff7353887bdf5cb4dde46c15907b5220106265aba34a6becbe2a2d9b3103c`.
- Current preview-capable probe script: `00cbd26f7c4eced1feb651ecbf183317638c5a42e43ce9c1544296c2833432b3`.

Evidence remains external to the repository under `C:\Users\HP\Desktop\Robot\reviews\issue-x2-dual-omnihand-20261010\`. The committed artifact is the reproducible harness, regression tests, and this report; generated caches, upstream copies, screenshots, videos, and raw traces are not added to Git.

## Gate Summary

| Gate | Result |
| --- | --- |
| Complete official dual-hand X2 compiles | PASS |
| Both native five-finger visual families present | PASS |
| Source visual/collision/joint/inertial/mimic parity | PASS |
| Both hands stay attached under render-only arm/finger kinematic motion | PASS, kinematic only |
| Official actuator semantics | DEFERRED; source URDF contains no transmissions |
| Stable dynamically actuated OPEN | FAIL; `0.14112 rad` target tracking error |
| Source-valid loaded mimic through approach | FAIL; `R_middle_dip_joint` first exceeds `0.003 rad` |
| Persistent bilateral opposition / physical grasp | DEFERRED; no thumb and no hold gate |
| Bottle load transfer and 1/5/30/50 mm lift | DEFERRED; no lift attempted |
| Left-hand physical control / simultaneous manipulation | DEFERRED |
