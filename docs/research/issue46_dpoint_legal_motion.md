# Issue #46 — D-Site Mapping and Legal-Range Static Preflight

**D-site mapping:** `VERIFIED`
**Legal-range static kinematics:** `PASS`
**Collision preflight:** `BLOCKED`
**MuJoCo physical motion:** `NOT RUN`
**Model class:** `UNVERIFIED_EXPERIMENTAL`

This follow-up applies only the D-site re-expression and CAD-to-URDF coordinate sign conversion approved for the isolated candidate. It does not modify the official URDF/STLs, active model, root limits/axes, mimic relation, topology, actuator settings, collision geoms, or collision filtering. The old E-phase fitting workaround was not used.

## Source and output identity

- Official source: `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`.
- Official X2 OmniPicker URDF SHA-256: `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- Source candidate MJCF SHA-256: `8f2f39d2a8fc3e8fbc9ec729a11b16a10b462fe3a525f4db9bd92745d6cb1b6f`.
- Corrected isolated candidate MJCF SHA-256: `e935c38d2869760de2ec1da6a689639a90ba143922aae0b48f8dbb7a98be1ba2`.
- Governed run directory: `/home/lzy18001500226/projects/RobotSim/reviews/issue-46/dpoint-legal-motion-20261021T113855Z/`.
- Its `artifact_paths.json` was validated with the shared path-governance helper. It contains the corrected MJCF copy, exact four-site diff, static JSON/CSV, and a non-physical kinematic telemetry preview. Original assets remain outside the run directory and were only read.

## Static result

The corrected candidate was loaded with MuJoCo 3.3.6. Eleven existing source samples per branch were evaluated from candidate q=0 through narrow q=-0.10 rad and mirrored wide q=+0.10 rad. Each sample used actual `MjModel`/`MjData`, assigned the mapped static `qpos`, and called `mj_forward`; this is static kinematic evidence, not a physics rollout.

| Check | Result |
|---|---:|
| Maximum D-site gap from `mjData.site_xpos` | `0.000308 mm` |
| Maximum `q_root_rel + q_B - q_E` residual from `qpos - qpos0` | `2.78e-16 rad` |
| Maximum official narrow/wide mimic residual | `0 rad` |
| Official root limits for all 11 samples on both branches | PASS |
| Full candidate q=-0.32 endpoint | Not run; outside unchanged official root limits |

At q=0 the corrected jaw and loop sites coincide within numerical precision. At q=-0.10 the largest D gap is `0.000308 mm`. No joint axis, limit, root phase, or mechanism relation was changed to achieve this.

## Collision evidence

For every sample, the audit compared MuJoCo's convex-hull distance with BVH-pruned triangle intersection checks on the original official collision STL triangles. The imported triangle counts, unique vertices, and vertex frames match the pinned source STL assets; maximum source-to-compiled-vertex error is below `2e-9 m`. The exact source mesh hashes and per-sample results are in `static_preflight.json` and `legal_interval_collision.csv`.

| Pair | MuJoCo hull penetration at q=0 / maximum in interval | Source STL triangle result | Assessment |
|---|---:|---|---|
| `narrow1_Link` × `narrow_loop_Link` | `6.111 / 7.466 mm` | No surface crossing; both source meshes are closed and containment test is clear | Confirmed convex-hull proxy artifact |
| `wide1_Link` × `wide_loop_Link` | `6.134 / 7.493 mm` | No surface crossing; the wide loop STL is nonmanifold, so containment is unverified | Hull artifact likely; source-solid status not fully established |
| `narrow3_Link` × `narrow_loop_Link` | `4.418 / 4.447 mm` | Source triangle surfaces intersect at all 11 samples | Real collision-mesh surface intersection; intended-vs-unintended status unresolved |
| `wide3_Link` × `wide_loop_Link` | `4.414 / 4.452 mm` | Source triangle surfaces intersect at all 11 samples | Real collision-mesh surface intersection; intended-vs-unintended status unresolved |

At q=0, the first narrow distal/loop intersecting triangle pair is `327 × 1326`; its intersection is around `[9.84, -13.72, 107.16] mm`, approximately `11.0 mm` from the corrected D center. The wide pair is `7132 × 2236`, around `[18.92, 13.3–13.6, 106.8–107.1] mm`, approximately `19.6 mm` from D. These are not intersections localized at the D pin. The relevant source collision STLs contain open/nonmanifold edges (narrow3: 1,422; wide3: 3,416; wide loop: 652), so a closed-solid penetration volume or designed-fit interpretation cannot be asserted from these surface records alone.

The root/loop pairs also show that convex hulls fill substantial concavities: source mesh volume is only about 44% of hull volume for narrow1/loop and wide1/loop. The narrow root/loop pair is proven separated as source solids while MuJoCo reports millimetres of hull penetration. The wide loop's nonmanifold topology prevents the same complete containment conclusion.

## Decision and next gate

The D-coordinate correction and legal-range source-sampled constraints pass. Physical MuJoCo motion is blocked before `mj_step`: the current collision model produces gross hull contacts, and both distal/loop source triangle surfaces intersect at the mapped source pose and throughout the sampled legal interval. Running the actuator against those contacts would mix real mesh intersections with false hull forces; no contacts were disabled and no collision mesh was replaced.

The smallest reviewable next decision is:

1. For the narrow/wide root-to-loop hull artifacts, approve a collision-only, source-derived convex decomposition of the existing official collision STL files. Keep the official visual meshes and original STL files unchanged, retain all collision pairs, and report decomposition error and per-sample contacts before any motion test.
2. For `narrow3_Link`/`wide3_Link` against their loop links, obtain manufacturer confirmation that the source STL surface intersections away from D are intended mating geometry or an authoritative corrected collision representation. A convex decomposition cannot resolve a true source-surface crossing.

No derived collision representation has been created. No `mj_step`, physical motion, bottle test, or full travel test ran. Candidate q=-0.10 is a short legal probe, not a complete open/close cycle or evidence of grasp readiness.

The local [kinematic telemetry preview](</home/lzy18001500226/projects/RobotSim/reviews/issue-46/dpoint-legal-motion-20261021T113855Z/kinematic_preview.html>) is explicitly non-physical. It exposes both branches' sampled root/passive coordinates and D gaps; it is not a 3D simulation.

Reproduction uses the pinned MuJoCo 3.3.6 environment and a fresh path-governed run directory:

```bash
cd /home/lzy18001500226/robotsim-issue46-dpoint-evidence-20261021
PYTHONPATH="$PWD/scripts:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages" \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python scripts/research/issue46_dpoint_legal_motion.py --self-test
RUN_DIR="$(PYTHONPATH="$PWD/scripts:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages" \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python -c 'from agent.path_governance import create_unique_output_directory; print(create_unique_output_directory("/home/lzy18001500226/projects/RobotSim/reviews/issue-46", prefix="dpoint-legal-motion", environment="wsl").wsl_path)')"
PYTHONPATH="$PWD/scripts:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages" \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python scripts/research/issue46_dpoint_legal_motion.py --run-dir "$RUN_DIR"
# Only if static_preflight.json reports dynamics_ready=true:
PYTHONPATH="$PWD/scripts:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages" \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python scripts/research/issue46_dpoint_legal_motion.py --run-dir "$RUN_DIR" --simulate
```

The current evidence command returned `BLOCKED_BEFORE_MOTION`; `--simulate` refused before any `mj_step` call. A new unique run directory is required for each reproduction.
