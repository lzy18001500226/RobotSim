# Issue #46 X2 OmniPicker Physical Envelope Decision

## Result

**PHYSICAL ENVELOPE UNVERIFIED.** The pinned vendor package contains URDF, STL, MJCF, preview images, and an optional simplified-collision URDF, but no STEP/IGES/CAD assembly or dimensioned drawing for the right wrist-roll housing or the OmniPicker loop links. A file inventory found no STEP/STP, IGES/IGS, SolidWorks, DWG/DXF, or local part drawing. Official documentation exposes whole-robot dimensions and an OmniPicker overall-size drawing, but neither identifies the physical envelope of these individual parts or their datums. No new grasp, corridor, controller, or collision-model experiment was run.

The previously measured collision remains valid evidence about the model. It is not evidence that the same surface obstructs the physical robot.

## Pinned Sources

- Vendor: [AgibotTech/agibot_x2_urdf at `575cc6b988f976c23550e0db85aa1e5475d3652d`](https://github.com/AgibotTech/agibot_x2_urdf/tree/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0), Mulan PSL v2.
- Active end-effector model: [`X2-Ultra_omnipicker.urdf`, wrist-roll declaration](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf#L1035-L1063).
- Right narrow and wide loop declarations: [narrow loop](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf#L1454-L1476), [wide loop](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf#L1580-L1602).
- Vendor README identifies the `X2-Ultra_simple_collision.urdf` as a simplified-collision model; it does not document part-level physical-envelope accuracy ([README](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/README.md#L7-L26)).
- Official AimDK sources: [X2 robot specifications](https://x2-aimdk.agibot.com/en/dev/about_agibot_X2/robot_specifications.html) and the [OmniPicker user-manual download](https://x2-aimdk.agibot.com/zh-cn/v0.8.1/_downloads/9179a383c97b0f37b1f0fbf51ed2cb35/omni_picker_introduction_for_x2.pdf). The public specs give overall X2 dimensions (1310 x 460 x 210 mm) and arm span, not wrist/loop dimensions. The search-indexed manual figure reports OmniPicker overall external-view callouts of 125 mm, 88.1 mm, and 60.5 mm across views, with 143 mm height. The PDF endpoint returned 404 during this audit, so those callouts are leads only; no part-level dimensions or datum mapping could be verified.
- MuJoCo documents that mesh collisions use convex hulls; non-convex shapes require a union of convex geoms or another supported representation ([collision documentation](https://mujoco.readthedocs.io/en/stable/computation.html)). This describes the simulator's collision representation, not the vendor's physical housing.

## Collision And Visual Comparison

| Part | Pinned URDF assignment | Finding | Physical interpretation |
|---|---|---|---|
| Right wrist-roll | Visual: `right_wrist_roll_extend_link.stl`; collision: `right_wrist_roll_link.stl`; both origins are zero and scales default to one. | The source intentionally or accidentally selects different meshes. SHA-256: visual `94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de`; collision `f6ed0a2a01407c6f314b6b2c04e5df8914f60963f3b546622f6734f03d487c72`. Prior mesh audit measured extents of 83.700 x 45.151 x 78.367 mm versus 99.633 x 71.043 x 204.628 mm, respectively; collision-mesh convex-hull volume was 3.3611x the visual hull, and 69.0% of collision vertices were outside the visual convex hull. | A possible asset or variant inconsistency is established. Intentional coverage of an unrendered housing, a conservative envelope, or stale/misaligned asset is not established. The separate standard `X2-Ultra.urdf` uses `right_wrist_roll_link.stl` for both visual and collision, which confirms this mesh is used as a vendor visual in another model variant, but does not prove the OmniPicker variant's physical wrist shape. |
| Right narrow loop | Visual and collision both use `narrow_loop_Link.stl`, at the same origin and scale. SHA-256: `63b895ff646b799c303c41c924ea4c77cf00f9c5d357e2fb4f6761c649237b17`. | No URDF-level visual/collision asset mismatch. | The loop STL has a physical-looking frame shape, but the source does not say whether the whole mesh is an accurate exterior or safety envelope. MuJoCo convexification can fill the loop's concavity and produce collision where the rendered frame is open. |
| Right wide loop | Visual and collision both use `wide_loop_Link.stl`, at the same origin and scale. SHA-256: `31aa42a993b3364e16efa657700ffe621eeaa5994d415bbe1b77f0687419b2b8`. | No URDF-level visual/collision asset mismatch. | Same uncertainty as the narrow loop. |

The collision/visual wrist difference was already tested as a historical diagnostic: replacing the wrist collision mesh with the vendor visual mesh still left a minimum `-35.489 mm` non-gripping robot/bottle intersection elsewhere in the full gripper. This did not identify the physically correct envelope and is not a grasp result. The original source collision mesh at the prior contact frame extended to `-34.792 mm` against the bottle body while visual-mesh samples remained `+23.423 mm` clear. These values are model-frame measurements from the prior A/B/C evidence, not hardware metrology. See [the existing outcome report](issue46_outcome_driven_grasp_corridor_20261009.md) and the preserved projection at `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\outcome-grasp-corridor-20261009\collision-counterfactuals\wrist_collision_vs_visual_projection.png`.

## Evidence Classification

- **Verified physical obstruction:** none. All prior obstructions are from the source mesh plus compiled simulation collision geometry; there is no corresponding physical-hardware measurement.
- **Verified conservative safety envelope:** none. The vendor names a separate simplified-collision model, but does not describe these OmniPicker parts as safety envelopes or quantify their conservatism.
- **Possible asset inconsistency:** yes for the OmniPicker wrist-roll visual/collision mesh selection. No mismatch is shown for the loop assets at URDF level.
- **Known simulator representation effect:** yes. MuJoCo collision meshes are convexified, so a non-convex loop mesh does not retain its open interior as a single collision geom. This may contribute to modeled interference; it does not prove the vendor collision mesh is physically wrong.
- **Unknown:** the actual external wrist-roll housing and loop surfaces, whether the larger wrist mesh intentionally includes hidden structure, and the physical clearances to the canonical bottle.

## Engineering Options

### A. Preserve the original geometry

Keep the pinned collision model unchanged. Request a dimensioned drawing, CAD assembly, or measurements of the right wrist-roll housing and narrow/wide loops in their URDF datums. This is the recommended current path. No physical claim or additional grasp attempt follows from the existing simulation-only measurements.

### B. Separate `SIMULATION_ONLY` diagnostic collision model

This option requires explicit maintainer approval before implementation. Keep the vendor checkout and source-faithful model untouched. In a separately named diagnostic copy, preserve the kinematic tree, transforms, masses/inertias, source limits, visual meshes, jaw collision surfaces, bottle, and physics settings. Replace collision surfaces only for the right `right_wrist_roll_link`, `R_hand_narrow_loop_Link`, and `R_hand_wide_loop_Link`, using surfaces derived from supplied physical CAD or measured housing boundaries. Do not remove a surface merely because it is absent from a vendor visual mesh.

The existing visual-mesh wrist substitution is an available diagnostic comparison, not a physical proxy and not a candidate for promotion; it already failed to clear the full model. The loop collision assets match their visuals at source, so no specific loop surface removal is supported now. Once a physical reference exists, validate it by aligned CAD-to-mesh surface deviation and isolated collision checks; retain intended jaw contacts and report every changed surface. Any later grasp test would require separate authorization.

## Single Maintainer Decision

Choose the next evidence gate: keep Option A and defer X2 grasp work until a physical envelope reference is supplied (recommended), or supply that reference and authorize Option B as a separately labeled diagnostic model. No collision-model change is requested or implemented here.

## Publication State

- Research branch: `research/issue46-outcome-grasp-corridor-20261009`; this report is an Issue-only research update and must not be represented as content of PR #58.
- Baseline HEAD for the prior investigations: `2850f11501919f892ba5e07ddabc8dc27a6575bc`.
- PR #58 remains OPEN and Draft at its independently verified head `fea64b09c8b159c4597fdc58de197b801136e144`; Issue #46 remains OPEN.
- The repository closeout interface accepts `pr_number: null` and targets the Issue identified by `task_id`; the research branch and exact pushed head are verified without inventing PR metadata. This report is for Issue #46 only and is not content of PR #58.
