# Issue #46 Native OmniPicker Grasp Follow-up (2026-10-12)

## Result

**Source-faithful static endpoint: FAIL for both bounded candidates. Physical contact/lift: NOT RUN.** This is a bounded result for the tested configurations, not proof that all native OmniPicker grasps are impossible. Both candidates failed zero-step collision/endpoint gates, so no physics was stepped and no full-arm IK was attempted.

The first unresolved gate is a collision-valid open/pregrasp configuration. In both layouts, the compiled right wrist-roll collision geom intersects the canonical bottle before a valid endpoint can be used. Both layouts also report the same compiled wrist-yaw/wrist-pitch self-contact at reset. No collision filters, source limits, meshes, bottle properties, or physical parameters were changed.

## Provenance and method

- RobotSim baseline at experiment start: `549f2062fe9aff399101526bfae606964eb54585`; branch `research/issue46-omnipicker-grasp-frame-20261012`.
- Official X2 source: `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- OmniPicker source URDF SHA-256: `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- Canonical bottle helper SHA-256: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`; canonical body diameter 70 mm, total height 244.5 mm, mass 0.570 kg.
- MuJoCo: 3.3.6. Source joint ranges, axes, mimic relation, collision geometry and bottle physics were retained. Candidate closure was sampled at 101 static states. `physics_steps=0` in both results.
- Candidate runner SHA-256 at execution: `e82d4e20d23893ee865af6664416ca3cc580195268f793351d1e8813e83e5633`. The small follow-up code edit changes the generated report wording only; it does not change geometry or candidate calculation. Source mesh and compiled hull inventories are preserved per candidate.

The two newly authorized, geometry-derived alternatives were a neck cross-section at local z offset 0 mm and an upper shoulder ellipsoid cross-section at local z offset +15 mm. Both used azimuth 0 degrees and zero finger tilt. The shoulder width is analytically derived from the compiled ellipsoid cross-section; the bottle asset itself was not edited.

## Measurements

| Candidate | Effective target width | OPEN jaw gap | Endpoint jaw gap | Narrow / wide pad signed distance | Min wrist-roll to bottle | Reset yaw/pitch self-distance | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Neck, z=0 mm | 36.000 mm | 103.091 mm | 36.082 mm | -0.075 / +0.069 mm | -25.175 mm | -39.217 mm | FAIL |
| Shoulder, local z=+15 mm | 57.176 mm | 103.091 mm | 57.176 mm | -5.196 / -4.960 mm | -32.829 mm | -39.217 mm | FAIL |

The endpoint tolerance for each intended pad witness is ±0.020 mm. The neck candidate narrowly misses that gate on both sides; it still has 25.175 mm wrist-roll penetration at the open state and throughout the sampled closure. The shoulder candidate has approximately 5 mm pad penetration and 32.829 mm wrist-roll penetration. Neither is a collision-valid endpoint. Source limits pass and endpoint mimic residual is 0 rad in both candidates.

## Wrist collision evidence

The pinned OmniPicker URDF assigns `right_wrist_roll_extend_link.stl` to visual geometry and `right_wrist_roll_link.stl` to collision geometry. Their mesh extents differ substantially (visual: 83.700 x 45.151 x 78.367 mm; collision: 99.633 x 71.043 x 204.628 mm). This is evidence of different mesh envelopes, not proof that either one matches the physical housing.

The pinned `X2-Ultra_simple_collision.urdf` has 39 links and contains no `R_omnipicker_base_link`, so it is not a complete OmniPicker comparison assembly. Where it defines `right_wrist_roll_link`, both visual and collision refer to `right_wrist_roll_link.stl`; the collision file SHA-256 is `f6ed0a2a01407c6f314b6b2c04e5df8914f60963f3b546622f6734f03d487c72`, identical to the OmniPicker wrist collision mesh. It therefore does not provide a smaller alternative wrist collision representation. No authoritative physical CAD or housing measurement was available to justify replacing the collision envelope.

The repeated reset contact is between `right_wrist_yaw_link/geom_2` and `right_wrist_pitch_link/geom_4`, signed distance -39.217 mm. This is a compiled collision-model observation; it is not independently verified as real hardware interference.

## Evidence and reproduction

Raw outputs are preserved under:

- Neck candidate: `/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-native-omnipicker-grasp-recovery/attempt-24bdfd13-73e8-4e86-aa4a-d2a8035c95a2-neck-a0-t0/`
- Shoulder candidate: `/mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-native-omnipicker-grasp-recovery/attempt-24bdfd13-73e8-4e86-aa4a-d2a8035c95a2-shoulder-a0-z15/`

Each directory contains `result.json` (including `runtime_identity`), `experiment_commands.txt`, the 101-row `closure_geometry_trace.jsonl`, endpoint/source/compiled geometry inventories, and five PNG views including `source_visual_collision_overlay_contact_witness.png`, `source_visual_collision_overlay_front.png`, `source_visual_collision_overlay_side.png`, and `compiled_collision_hull_only.png`.

Reproduction commands (run from the research worktree; the selected geom's local-z offset is passed through `--grasp-height-offset-mm`):

```bash
MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -B scripts/research/issue46_native_omnipicker_grasp_frame_audit.py \
  --x2-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf \
  --canonical-helper /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robosuite-bottle-20261011/robotsim_reference/canonical_manipulation_assets.py \
  --output /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-native-omnipicker-grasp-recovery/attempt-24bdfd13-73e8-4e86-aa4a-d2a8035c95a2-neck-a0-t0 \
  --azimuth-deg 0 --finger-tilt-deg 0 --grasp-height-offset-mm 0 --target-bottle-geom bottle_neck

MUJOCO_GL=egl /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -B scripts/research/issue46_native_omnipicker_grasp_frame_audit.py \
  --x2-root /home/lzy18001500226/.cache/robotsim/research/issue46-x2-robotiq-m0-20261009/sources/agibot_x2_urdf \
  --canonical-helper /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/robosuite-bottle-20261011/robotsim_reference/canonical_manipulation_assets.py \
  --output /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-native-omnipicker-grasp-recovery/attempt-24bdfd13-73e8-4e86-aa4a-d2a8035c95a2-shoulder-a0-z15 \
  --azimuth-deg 0 --finger-tilt-deg 0 --grasp-height-offset-mm 15 --target-bottle-geom bottle_shoulder
```

## Decision and next blocker

Do not proceed to contact physics, full-X2 IK, or lift using these candidates. The specific blocker is the pinned wrist collision envelope intersecting the canonical bottle at the proposed native hand poses, compounded by an active yaw/pitch collision in the fixture reset pose. The available simple-collision URDF does not resolve the wrist envelope discrepancy. Physical-envelope verification remains unavailable from these assets alone.

The next useful evidence is authoritative vendor CAD or measured housing dimensions/assembly transforms for the wrist-roll and OmniPicker mount. Without it, a less obstructive replacement would be an unsupported collision-model change. This report does not claim global infeasibility or hardware acceptance.
