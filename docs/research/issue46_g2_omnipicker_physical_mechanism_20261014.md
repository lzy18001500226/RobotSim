# G2 OmniPicker MuJoCo Physical Mechanism Gate

## Result

**BLOCKED — static loop closure was not established across the official root-joint range.** The generated stand-alone MuJoCo model compiles and preserves the pinned G2 link/joint tree, source limits, inertias, original visual DAE assets, and original convex collision STL assets. A 101-sample constrained static scan found a 24.178 mm best-fit endpoint residual at full open. The closed-end residual is only 0.099 mm; that small value is reported as unresolved numerical/geometry discrepancy and is not by itself treated as proof of mechanical infeasibility. No `mj_step()` calls were made, no dynamic or contact forces were measured, and no actuator-driven viewer is claimed.

The G2 candidate is **not validated for physical opening/closing**. The result is limited to the literal spherical-loop interpretation described below. The pinned Xacro supplies custom `loop_joint` tags but no implementation or tolerance for those tags was found in the pinned repository, so the interpretation cannot be called an authoritative AgiBot simulator contract.

## Provenance and Reproduction

| Item | Identity |
|---|---|
| Upstream | [AgibotTech/genie_sim_robot_model](https://github.com/AgibotTech/genie_sim_robot_model) |
| Commit | `6d5b2159806ae0e64b432b638e9ee288d8bddc8f` |
| License | Apache-2.0 (`LICENSE` in pinned checkout) |
| Xacro | `robots/genie/g2/xacro/gripper.omnipicker.urdf.xacro` |
| Xacro SHA-256 | `b20d8691b03a256ad3481380b71fd1e1eaa11495673f4bf7fd7377f67238d183` |
| Runtime | Python 3.10.12, MuJoCo 3.3.6, NumPy 2.2.6, SciPy 1.15.3 |
| Experiment attempt | `e2956566-cb0c-405e-8bc3-b11f631e677b` |

Reproduce the static gate and verify that the requested dynamics stage is suppressed:

```bash
cd /home/lzy18001500226/robotsim-issue46-g2-omnipicker-physics-20261014
env MUJOCO_GL=egl /home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python \
  scripts/research/issue46_g2_omnipicker_physical.py \
  --out /home/lzy18001500226/projects/RobotSim/reviews/issue-46/g2-omnipicker-physical-20261014-e2956566-cb0c-405e-8bc3-b11f631e677b/validation02 \
  --samples 101 --dynamics
```

The command exits successfully after recording `BLOCKED_STATIC_CLOSURE_NOT_ESTABLISHED` and `mj_step_calls: 0`. The exit code indicates the evidence run completed, not that the mechanism passed.

## Source-to-MuJoCo Mapping

| Source item | Source definition | MuJoCo representation / status |
|---|---|---|
| Active inner root | `inner_joint1`, axis `(0,0,1)`, range `[-π/4,0] rad`, effort/velocity `10` | Hinge with same range and axis; the only position actuator. `drive stiffness=1000`, `damping=10` retained; effort capped at source 10 N·m. |
| Outer root | `outer_joint1`, mimic `-1 × inner_joint1`, range `[0,π/4] rad` | One joint equality `outer=-inner`; no second actuator. |
| Six passive hinges | Inner mimic ratios `0.1, 0.25, -0.7`; outer `0.1, -0.25, 0.7`; all source ranges `[-2,2] rad` | Preserved as unactuated hinge DOFs. Their source mimic values are measured as a kinematic reference, not imposed as additional equalities; passive coordinates are selected only by static closure solve and would be left to constraints during any permitted rollout. |
| Outer spherical loop | Base local point `(0,+12,+105) mm` to `outer_link2` local origin | MuJoCo site-to-site `equality/connect`: three translational coincidence equations; relative orientation remains free. |
| Inner spherical loop | Base local point `(0,-12,+105) mm` to `inner_link2` local origin | Same mapping, separately named. |
| Geometry | Original Xacro visual DAE and `convex/*.STL` collision proxies | DAE is converted for rendering with its COLLADA scene matrix/unit/up-axis recorded; original source convex STL files are used for collision. Vendor files remain untouched. |
| Root mount | `G2.gripper.urdf.xacro`: parent `right_arm_end_link`, offset `(0,0,+5.1) mm`, RPY `(0,0,π/2)` | Stand-alone fixture fixes base at world identity; robot wrist mount is recorded but not part of this fixture. |

Compiled model dimensions: `nq=8`, `nv=8`, `njnt=8`, `neq=3`, `nu=1`, `ngeom=21`. The three equalities are one root mimic relation and two three-dimensional point connections. This leaves one nominal mechanism DOF. No passive-joint actuator, equality mimic, weld, collision exclusion, or rollout qpos write was added.

## Static Results

The scan fixes the source root pair at `outer=-inner` and independently solves the six passive coordinates within their source `[-2,2] rad` ranges to minimize the two declared spherical-loop endpoint errors. Optimizer `success` means only that its termination criterion was met; it does not mean the loop closed.

| Root `inner_joint1` | Source-mimic loop residual (inner / outer) | Best-fit loop residual (inner / outer) | Interpretation |
|---:|---:|---:|---|
| `0 rad` (closed) | `0.572 / 0.572 mm` | `0.099 / 0.099 mm` | Small residual remains; no vendor tolerance is published, so it is not alone called a mechanical failure. |
| `-π/8 rad` (half travel) | `18.396 / 20.303 mm` | `11.414 / 11.414 mm` | Centimeter-scale residual after passive-coordinate optimization. |
| `-π/4 rad` (full open) | `35.239 / 39.392 mm` | `24.178 / 24.178 mm` | Static closure not established at full opening. |

At full open, best-fit residual vectors in the base/world fixture frame are:

| Loop | `end - base` residual (mm) | Best-fit passive angles |
|---|---|---|
| Inner | `(-0.133, -17.634, -16.541)` | joint3 `-0.8875`, joint4 `+0.8555`, joint0 `0 rad` |
| Outer | `(+0.133, +17.634, -16.541)` | joint3 `+0.8875`, joint4 `-0.8555`, joint0 `0 rad` |

The passive angles remain within their source limits; the full-open residual is not explained by a passive limit being reached. The maximum absolute difference from the six source mimic targets in the best-fit scan is `1.1501 rad`. This is evidence that source mimic expansion and the literal loop-point constraints do not describe the same motion over this range; it does not establish how the custom Xacro tag is interpreted by an external AgiBot engine.

Across 101 evenly spaced root samples, 88 had best-fit residual at or above 1 mm; the first such sample is `q=-0.10210 rad`, residual `1.144 mm`. The 1 mm band is only a descriptive scale marker, not a vendor acceptance threshold. The maximum remains `24.178 mm` at full open. An existing eight-seed comparison at representative root positions converged to the same best-fit values; it is a bounded local robustness observation, not proof of a global no-solution theorem.

## Collision and Dynamics Status

`mj_forward` reports overlapping pairs among the source convex collision proxies in the best-fit static samples. The most severe minima over the scan include:

| Convex collision proxy pair | Minimum signed distance |
|---|---:|
| base / outer link1 | `-29.969 mm` |
| base / inner link1 | `-29.628 mm` |
| base / inner link2 | `-25.256 mm` |
| base / outer link2 | `-25.024 mm` |
| inner link2 / outer link2 | `-15.655 mm` |
| outer link1 / outer link2 | `-13.010 mm` |
| inner link1 / inner link2 | `-12.294 mm` |

These are static convex-proxy overlap/contact candidates, not measured contact forces or proof of physical interference in the real assembly. The source provides no collision-pair exclusions in this macro; none were added here. Their severity is a second reason not to proceed into unconstrained dynamic interpretation, but it is not used to infer contact force.

Because full-range static closure was not established, the dynamic gate was not entered: `mj_step_calls=0`, no dynamic cycles, no measured passive motion, no solver/contact forces, and no actuator-driven interactive viewer. Static `closed.png`, `open.png`, and `open_closed.png` show the best-fit geometry states only and must not be read as dynamics evidence.

## Engineering Decision

The current MuJoCo representation is a defensible literal mapping of the two Xacro spherical connections to point-coincidence constraints, but it does **not** establish a closed mechanism across the source root range. The pinned Xacro does not define an implementation for its custom `loop_joint` tags beyond the tags, link names, local points, orientation fields, and the comment “Closed-loop kinematic constraints”; no loop implementation was found elsewhere in the pinned upstream checkout. Thus the software-side blocker is **unresolved source semantics / kinematic compatibility under the literal spherical-joint mapping**, plus unresolved source convex-proxy overlaps—not a demonstrated MuJoCo actuator or solver failure.

Do not copy this G2 mechanism into X2 as validated physics. G2 remains a useful separate structural reference: its extra passive hinges and loop tags are absent from the pinned X2 OmniPicker representation, and the G2 wrist mount `(0,0,+5.1) mm, RPY (0,0,π/2)` differs from X2's mount. No X2 model was changed.

The next discriminating evidence is the authoritative G2 loop-joint interpretation (the generator/runtime that consumes these custom tags, or vendor mechanical joint/anchor definition) and a validated collision-pair policy for the supplied convex proxies. Until then, physical G2 opening/closing and X2 reuse are not established.

## Evidence Files

The original first attempt is retained unchanged in the experiment root. The corrected comparison is under `validation02/`:

- `result.json`, `static_summary.json`, `static_scan.json`, `static_scan.csv`
- `g2_omnipicker_constrained.xml` and mesh manifest embedded in result JSON
- `closed.png`, `open.png`, `open_closed.png`
- no dynamics trace, because the static gate prevented physics
