# Issue #46 X2 OmniPicker Isolated Collision Diagnostic

**Outcome: C — no defensible replacement collision representation can be established without vendor CAD or hardware measurement.** The diagnostic remains bounded to the pinned source assets and already completed collision counterfactuals. No new collision model, pose search, physical grasp, or lift was run.

## Scope and immutable control

- Vendor: `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d` (Mulan PSL v2).
- Model: `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- Comparison files at the same commit: `X2-Ultra.urdf` SHA-256 `5973626d3639968daf48c688609d032d6dc73490ef63834c172481a112e33c60`; `X2-Ultra_simple_collision.urdf` SHA-256 `f1de13da59ad9484c8940c8574705c75c0b0bd445b7954ae222639f33086f452`.
- Original source-faithful collision model is the immutable control. No vendor file, production collision geometry, accepted physics, or canonical bottle property was changed.
- The canonical diagnostic object remains the multipart 70 mm, 244.5 mm, 0.570 kg bottle with its accepted friction and gravity settings.

## Verified source facts

| Component | Pinned source assignment | Evidence |
| --- | --- | --- |
| OmniPicker right wrist roll | Visual uses `right_wrist_roll_extend_link.stl`; collision uses `right_wrist_roll_link.stl`. Both origins are zero and mesh scale is the default 1. | The [pinned OmniPicker URDF](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf#L1035-L1063) specifies different mesh files. SHA-256: visual `94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de`; collision `f6ed0a2a01407c6f314b6b2c04e5df8914f60963f3b546622f6734f03d487c72`. |
| Standard X2 wrist roll | Visual and collision both use `right_wrist_roll_link.stl`. | The [pinned standard URDF](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra.urdf#L1035-L1055) uses the collision mesh as its visual too. This establishes reuse across model files, not physical compatibility with the OmniPicker assembly. |
| Simple-collision wrist roll | Visual and collision both use `right_wrist_roll_link.stl`; this URDF has no OmniPicker loop/tool assembly. | The [pinned simple-collision URDF](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra_simple_collision.urdf#L1202-L1217) and [vendor README](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/README.md) identify a simplified-collision model. No part-level accuracy or OmniPicker compatibility is documented. |
| Narrow and wide loops | Each loop uses the same named STL for visual and collision, at zero origin and default scale 1. | [Narrow loop declarations](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf#L1454-L1476) and [wide loop declarations](https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf#L1580-L1602). Mesh SHA-256: narrow `63b895ff646b799c303c41c924ea4c77cf00f9c5d357e2fb4f6761c649237b17`; wide `31aa42a993b3364e16efa657700ffe621eeaa5994d415bbe1b77f0687419b2b8`. |

The pinned wrist collision mesh is much larger than its paired visual mesh: AABBs are `99.633 x 71.043 x 204.628 mm` and `83.700 x 45.151 x 78.367 mm`, respectively; their convex-hull volume ratio is `3.3611`, and `69.0%` of collision vertices lie outside the visual convex hull. In the previously measured contact frame, the collision mesh sampled `34.792 mm` inside the bottle-body cylinder while the visual mesh was `23.423 mm` clear. These are source-mesh/simulator measurements, not hardware-envelope measurements.

### Verified simulator effect

MuJoCo 3.3.6 represents a mesh geom collision shape by its convex hull. A single hull therefore fills concavities and loop openings; the [MuJoCo 3.3.6 mesh reference](https://mujoco.readthedocs.io/en/3.3.6/XMLreference.html#asset-mesh) and [collision documentation](https://mujoco.readthedocs.io/en/3.3.6/computation/index.html#collision-detection) describe this simulator behavior. The source narrow-loop STL is watertight and has a convex-hull/source signed-volume ratio of `2.28876`; `4,606 / 5,155` vertices lie strictly inside its hull. The wide-loop STL is non-manifold: exact-float edge counting finds 4 boundary edges and 648 edges with incidence 4, so its signed volume is not a reliable baseline; `3,515 / 4,026` vertices lie strictly inside its hull. These figures come from the new `loop_hull_metrics.json` parser and replace an earlier rough topology count in the working notes. They establish a material hull-fill effect in the model, not that the loop collision surfaces are physically inaccurate.

The new projection in `loop-geometry/loop_source_vs_convex_hull_projections.png` draws the original pinned loop STL facets against the corresponding single-hull facets in three orthographic views. It is a geometric visualization of the simulator representation, not a substitute collision model or hardware validation.

## Counterfactuals reused

No new collision representation was generated. The following prior diagnostics are reused and preserved in the external evidence packet:

| Diagnostic | Existing result | Decision |
| --- | --- | --- |
| Wrist collision replaced with its vendor visual mesh | Full model still had `-35.489 mm` non-gripping interference. | Does not establish that the visual mesh is the correct physical collision envelope. |
| Six convex pieces per wrist/loop mesh, binned from source triangles | Sum-of-piece-hull / single-hull volume ratios: narrow loop `1.609`, wide loop `1.600`, wrist `0.992`. Minimum non-jaw distances remained `-37.143 mm` at the original pose and `-44.035 mm` at the centered pose. | Rejected: loop pieces add substantial volume relative to the original single hull and the grasp corridor did not improve. |
| 81-piece diagnostic | Minimum non-gripping distance worsened to `-56.153 mm`. | Rejected; no further decomposition or pose search is justified by this result. |

The six-piece decomposition preserved an algorithmic approximation of source triangles, but its inflated loop hull volume and failed clearance do not support it as a physically grounded model. The historical run records aggregate piece-hull volume, not the boolean union volume, and did not save a sampled two-sided exterior-surface deviation metric. Its source/candidate scene views and closure profile are included only as historical visual evidence; this report does not claim a quantitative exterior-surface error for that decomposition. The original source-vs-single-hull loop overlay and wrist visual/collision projection are direct geometry projections. Raw source/counterfactual JSON and closure profiles are copied without modification.

## Hardware interpretation: facts and unknowns

**Verified:** the OmniPicker URDF intentionally or accidentally assigns different wrist visual and collision files. The standard and simple-collision URDFs reuse `right_wrist_roll_link.stl` as both visual and collision. The two OmniPicker loop files match at the URDF visual/collision level. MuJoCo's convex-hull conversion fills concave mesh interiors.

**Not verified:** whether the wrist visual/collision difference is a deliberate hidden housing/extension, a conservative collision envelope, a stale asset, or an assembly/variant mismatch; the physical outer surface of either loop; and the assembled wrist/loop clearance to the canonical bottle. The pinned vendor package contains no dimensioned wrist/loop drawing or STEP/IGES assembly. The simple-collision model cannot resolve this because it supplies no OmniPicker loop assembly and no part-level accuracy statement.

## Static and physical gates

- The source-faithful control did not pass the existing tested OPEN configurations: active non-gripping bottle contacts were present before closure. The two tested configurations are bounded failures, not a global incompatibility finding.
- The visual-only substitution and six-piece counterfactual also failed static clearance. No tested representation demonstrated collision-free OPEN, approach, opposing intended contact, and a clear 30 mm lift corridor together.
- Physical grasp/lift was **not run**. No X2 contact-force trace or X2 physical MP4 is claimed. No object weld, attachment, teleport, or rollout qpos write was used in this task.

## Decision and next evidence gate

**C — no defensible collision representation can be established without vendor CAD or hardware measurement.** Preserve the original source collision model as the control. Do not promote the visual-only or decomposed models.

The single maintainer decision needed before any collision-model work is to provide or authorize a dimensioned physical-envelope reference for the right wrist-roll and narrow/wide loop links, registered to the URDF link frames. After that reference exists, a separately named `SIMULATION_ONLY` model could compare source collision against CAD-derived outer surfaces, report surface deviation and altered regions, then rerun static gates. Any physical grasp remains a separate approval gate.

## Provenance and publication

- Research branch: `research/issue46-omnipicker-collision-diagnostic-20261009`.
- Evidence directory: `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-collision-diagnostic-20261009\`.
- This report is an Issue #46 research update only. PR #58 remains Draft and is not represented as containing this report. Issue #46 remains open.
- Historical raw runs and exact commands are indexed in the evidence directory. No prior artifacts were modified.
