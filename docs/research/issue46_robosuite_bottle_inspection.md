# Issue #46: robosuite BottleObject Inspection

## Scope and result

Inspected the pinned robosuite `BottleObject` as an isolated MuJoCo object and compared it against the existing RobotSim four-geom bottle. The scene contains only the ground, G1 table, and bottles; no robot, grasp, arm IK, controller, or finger tuning was run.

**Result:** both objects fell onto the table and remained table-supported and upright for the 10 s run. The strict low-motion settle check did not pass for either object. This is an inspection result, not evidence that either bottle is ready for manipulation.

## Provenance

- Repository: [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite)
- Pinned commit: `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`
- Object XML: `robosuite/models/assets/objects/bottle.xml`
- Mesh: `robosuite/models/assets/objects/meshes/bottle.stl`
- Texture: `robosuite/models/assets/textures/glass.png`
- MIT license and the separate MuJoCo Apache-2.0 notice are preserved in the external evidence copy.
- Source asset SHA-256 values and the exact clean-checkout verification are in the external `result.json` and `sha256sums.txt`.

The runner reproduces the relevant `BottleObject` XML transformation directly and does not import the robosuite package. The pinned `PickPlace` implementation was inspected and constructs Milk, Bread, Cereal, and Can objects; it does not instantiate `BottleObject`.

## Measurements

Runtime: Python 3.10.12, MuJoCo 3.3.6, timestep 0.002 s, Euler integrator, Newton solver, 100 solver iterations, gravity `(0, 0, -9.81) m/s^2`.

| Property | robosuite BottleObject | RobotSim canonical bottle |
|---|---:|---:|
| Mesh/bounds | 55.105 x 58.550 x 160.000 mm | 70 x 70 x 244.5 mm |
| Compiled mass | 0.028031821 kg | 0.570000040 kg |
| Principal inertia (kg m^2) | `[5.02712e-5, 5.02712e-5, 8.55547e-6]` | `[0.00202403, 0.00202403, 0.000329865]` |
| Collision | One active mesh; 132 render triangles and 14 compiled convex-hull polygons | Four primitive collision geoms: body, shoulder, neck, cap |
| Friction / `condim` | `[0.95, 0.3, 0.1]` / 3 | `[1.4, 0.02, 0.001]` / 4 |

The raw upstream object compiles to 0.028031811 kg. `BottleObject` retains the source visual and collision geoms and duplicates the collision geom as a non-contact visual with mass `1e-8`; the original visual geom's density remains active in body mass. The source mesh is therefore not a 570 g bottle. Its original `glass` material and `glass.png` texture were preserved. Only the review renderer's skybox was hidden.

Both bottles were dropped from 50 mm above the same G1 table (0.400 x 0.400 m top, top height 0.800 m), with their existing object friction/contact properties unchanged.

- Upstream first table contact: 0.102 s; one final table contact; final normal support force 0.275 N; maximum horizontal drift after contact 0.235 mm; maximum upright error 0.232 deg; maximum transient penetration 3.021 mm.
- Canonical first table contact: 0.102 s; one final table contact; final normal support force 5.646 N; maximum horizontal drift 0.189 mm; maximum upright error 0.078 deg; maximum transient penetration 7.148 mm.
- At 10 s, upstream linear/angular speeds were 0.00609 m/s and 0.10435 rad/s. Canonical speeds were 0.00418 m/s and 0.03796 rad/s.
- The measured low-motion check required, for the final 0.5 s, linear speed <= 0.010 m/s, angular speed <= 0.050 rad/s, and upright error <= 5 deg. Both failed because angular motion exceeded the threshold. No solver or contact tuning was performed.

## Recommendation

**Retain both as comparison fixtures.** The upstream BottleObject is useful for preserving robosuite provenance and mesh appearance, but is about 20.3 times lighter and smaller than the RobotSim canonical bottle. Do not silently replace the canonical 570 g target. No adapted upstream variant was created.

## Evidence and validation

External review bundle: `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\robosuite-bottle-20261011\`

It contains full/front/side/collision views, the labeled side-by-side comparison, both settling MP4s, pose/contact CSVs, generated MJCF, `result.json`, exact reproduction instructions, source copies with license, and a SHA-256 manifest. Both MP4s decoded fully at 30 fps (295 frames each); every manifest entry verified.

Validation performed:

- `python3 -m py_compile scripts/research/issue46_robosuite_bottle_inspection.py` — PASS.
- `git diff --check` — PASS.
- Standalone and paired MuJoCo gravity/contact run — table support/upright PASS; strict low-motion settle FAIL.
- MP4 decode and SHA-256 manifest verification — PASS.

See the external `REPORT.md`, `REPRODUCE.md`, and `result.json` for the exact command, per-geom values, full traces, asset hashes, and final state.

## Research tool disposition

`scripts/research/issue46_robosuite_bottle_inspection.py` — **KEEP-AS-TOOL**, limited to offline asset inspection and evidence generation. Production/runtime code must not depend on it. Generated media and upstream assets remain external to the repository.
