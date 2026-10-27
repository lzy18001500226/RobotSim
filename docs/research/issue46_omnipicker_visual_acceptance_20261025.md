# Issue #46 OmniPicker Visual Acceptance

**Decision:** `HUMAN_VISUAL_REVIEW=PASS` for the exact static Option D scene and artifact set listed below. This is a visual/static acceptance record, not a physics qualification.

## Frozen identities

- Canonical gripper: `X2-OmniPicker-Sim-Functional-v0.1`, classified `SIMULATION_ONLY_UNVERIFIED_EXPERIMENTAL_GEOMETRY_SELECTED_COUPLING`.
- Canonical package source: PR #92 HEAD `5e2ed1dcbfac2e357fd19191733b377034ea09f5`; portable `model.xml` SHA-256 `5e33d30e212ee72271efc3f0a790ebfb46cd31e775c578a10f1a2fb1f33c93e7`.
- Vendor source: `https://github.com/AgibotTech/agibot_x2_urdf.git`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`; official X2 OmniPicker URDF SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- Reviewed assembly: Option D, `SIMULATION_ONLY_DIAGNOSTIC`; scene SHA-256 `a137a6efc793a16df808e23d764594eba4f3449664fbc5ceca554a210742b10a`.
- Review viewer: PR #94, source HEAD `3dc22a5bbcc5b9ac7e6ca1419d2df159f5e7f7f4`.
- Maintainer visual approval: 2026-10-25, covering the exact scene SHA above.

The machine-readable companion manifest records hashes for the model metadata, source and collision assets, the Option D scene inputs, and the accepted screenshot/report set. Local screenshots and generated scene files remain under the ignored `reviews/` tree; they are review evidence, not clean-checkout runtime dependencies.

## Accepted review scope

| Check | Result |
|---|---|
| Full-X2 visual assembly and wrist mounting | `PASS` |
| Left and right wrist visual geometry | `PASS` |
| Left and right gripper static poses at 2%, 50%, and 100% | `PASS_STATIC_TRANSFORM_PARITY` |
| Collision-overlay visibility | `PASS` (displayed compiled collision geoms; this does not assert collision safety) |
| `HUMAN_VISUAL_REVIEW` | `PASS` |
| Physics stepping in reviewed scene | `0` steps; `mj_forward` only |

The three reviewed static pose samples show five-section aperture measurements of 12.851 mm (2%), 61.416 mm (50%), and 90.992 mm (100%). The viewer does not expose the 0% hard-stop endpoint. These are static kinematic visualization values, not grasp clearances or dynamic performance.

## Qualification boundaries

- `FULL_X2_DYNAMIC_PHYSICS=NOT_QUALIFIED`.
- `BOTTLE_GRASP=NOT_RUN / BLOCKED`.
- `HEAD_TORSO_COLLISION=BLOCKED`: the pre-existing `head_pitch_link`–`torso_link` contact remains in 99/99 static samples; deepest reported penetration is 7.112 mm. Option D does not resolve or mask it.
- Option D uses the official extended right-wrist STL as a diagnostic collision-mesh substitution. It is not the official URDF right-wrist collision assignment; this scene is not a production collision model.
- `MANUFACTURER_EQUIVALENT_GRIPPER_TRANSMISSION=UNVERIFIED`; v0.1 remains an experimental functional coupling.
- The prior 17,500-step result applies to the original PR #90 MJCF hash `32353b81b9893fb049da55962b72a65bbae865189e1d404579502140515d6c45`, not the portable v0.1 MJCF hash above. Portable-model dynamics were not rerun.

The accepted source review capture is `reviews/issue-46/dual-omnipicker-human-review-20261025-20261025T062606Z/`. The exact Option D scene and its static audit are under `reviews/issue-46/official-wrist-variant-static-audit-20261024T162630Z/verified/`. These local evidence paths may be absent from a fresh Git clone; the companion manifest preserves their SHA-256 identities.

## Future change contract

Treat the v0.1 gripper package and this exact Option D static scene as immutable review baselines. Head/torso collision repair, collision refinements, bottle contact, and full-X2 dynamics belong in separately identified derived configurations. Any change to the reviewed gripper or wrist geometry requires a new model identity and visual acceptance review.
