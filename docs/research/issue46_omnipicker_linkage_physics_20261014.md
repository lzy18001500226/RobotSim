# Issue #46: X2 OmniPicker Linkage Constraint Feasibility

## Verdict

**BLOCKED — the current fitted-pin SIMULATION_ONLY candidate does not establish a source-grounded static loop closure.** The exact rigid-pin constraints leave a nearly constant 0.08764 mm narrow-side and 0.12008 mm wide-side residual across 101 driver positions. The residual is comparable to the mesh-feature fit uncertainty, and the official source provides no pin datums or clearance tolerance. This does not establish that the physical X2 mechanism is infeasible. Since the static geometry gate is unresolved, no `mj_step()` dynamics or bottle test was run.

## Source and model boundary

The X2 source is `AgibotTech/agibot_x2_urdf` revision `575cc6b988f976c23550e0db85aa1e5475d3652d`, file `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf` (SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`). It defines two root revolute joints: `right_claw_joint` in `[-1, 0] rad`, and `R_hand_wide1_joint` mimicking it at multiplier `-1`. The second-stage and loop links are fixed in this URDF. Its fixed-joint axis/limit tags do not define moving passive-joint axes or valid passive limits.

The separate G2 reference is `AgibotTech/genie_sim_robot_model` revision `6d5b2159806ae0e64b432b638e9ee288d8bddc8f`, Xacro `robots/genie/g2/xacro/gripper.omnipicker.urdf.xacro` (SHA-256 `b20d8691b03a256ad3481380b71fd1e1eaa11495673f4bf7fd7377f67238d183`). G2 defines six revolute follower joints with mimic coefficients and two custom spherical `loop_joint` tags. Those tags are not standard URDF joints, and the G2 wrist mount differs from X2. G2 is therefore a topology reference only; its linkage parameters were not transferred as validated X2 mechanics. The prior comparison is in `issue46-g2-omnipicker-compatibility-20261013/REPORT.md`.

## Candidate constraint model

The isolated candidate retains the X2 candidate collision meshes and root geometry. It adds six passive revolute hinges at axes inferred from mesh bore features, a source-ratio root joint equality `q_wide = -q_claw`, and two site-to-site `connect` equalities at distal pin centers fitted from the STL. Only the narrow root has a bounded, simulation-derived position servo (`kp=0.04`, `kv=0.002`, force cap `0.02 Nm`); there are no passive actuators. The six passive hinges have no source-backed limits because their corresponding source joints are fixed. This is an experimental model definition, not a reconstruction of verified hardware pivots.

MuJoCo supports equality constraints for loop closure; `connect` constrains two sites to a common point and `joint` equality can express a polynomial coordinate relation. These are solver constraints in the simulated model, not evidence that the source URDF supplies the omitted mechanism. [MuJoCo 3.3.6 XML Reference](https://mujoco.readthedocs.io/en/3.3.6/XMLreference.html)

## Static results

MuJoCo 3.3.6 compiled the isolated model (`nq=8`, `nv=8`, `neq=3`, `nu=1`, `ngeom=18`). A 101-sample `mj_forward()` scan covered the full source root range `[0, -1] rad`. The source mimic relation was exact in the scan (maximum residual `0 rad`). Passive angles were solved offline for each sample; no dynamic stepping occurred.

| Driver q (rad) | Narrow pin residual (mm) | Wide pin residual (mm) | Maximum absolute passive angle (rad) |
|---:|---:|---:|---:|
| 0.0 | 0.087642 | 0.120083 | 0.0646 |
| -0.5 | 0.087642 | 0.120083 | 2.3896 |
| -1.0 | 0.087642 | 0.120083 | 1.3492 |

The q=0 residual vectors are approximately narrow `( +0.087634, -0.001152, 0.0000004 ) mm` and wide `( -0.120073, +0.001579, -0.0000005 ) mm`. Finite-difference passive-angle Jacobians have rank 2: their smallest singular values are approximately `8.9e-13` and `5.3e-13 m/rad` at q=0. The residual direction lies outside the locally reachable displacement subspace, so changing the three fitted hinge angles cannot exactly close the current pin-site model. The residual is almost unchanged at q=-0.5 and -1.

The maximum residual is the same scale as the previously measured STL bore-fit RMS (roughly 0.05–0.12 mm). Those fits are not CAD datums, and no pin clearance or manufacturing tolerance is provided. Therefore the result rejects the *current exact fitted-site model as an established exact closure*, but cannot distinguish fit error from physical pin clearance or establish hardware infeasibility. The 0.01 mm code threshold is only a strict numerical exact-closure diagnostic, not a vendor tolerance or product requirement.

The candidate's compiled collision hulls also produce contacts/overlap through the scan: 597 contact-pair occurrences across eight distinct pairs; worst proxy penetration is 14.201 mm between narrow2 and narrow loop at q=-0.5. These are MuJoCo mesh collision results for the candidate hulls. They are not triangle-surface measurements and do not prove source hardware interference.

## Dynamics and grasp status

- Actual physics steps: **0**. The static gate is unresolved, so the dynamic branch was not entered.
- Open/close without manually initialized passive qpos: **not tested**.
- Passive state, dynamic loop residual, solver stability, and dynamic collision behavior: **not tested**.
- Bottle grasping: **not run**, as requested.
- The code contains a gated dynamic path, but its presence is not a validation result.

## Reproduction and evidence

Run the static 101-point scan from the isolated worktree:

```bash
cd /home/lzy18001500226/robotsim-issue46-omnipicker-linkage-physics-20261014
/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python \
  scripts/research/issue46_omnipicker_linkage_constraint_validation.py \
  --source-xml /mnt/c/Users/HP/Desktop/RobotSim/reviews/issue46-g2-omnipicker-compatibility-20261013/x2_g2_omnipicker_review.xml \
  --source-urdf /home/lzy18001500226/.cache/robotsim/upstreams/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf \
  --g2-xacro /home/lzy18001500226/.cache/robotsim/upstreams/genie_sim_robot_model/robots/genie/g2/xacro/gripper.omnipicker.urdf.xacro \
  --attempt-id df89d796-4673-445f-b003-d65f6dc689e5 \
  --out /home/lzy18001500226/projects/RobotSim/reviews/issue-46/omnipicker-linkage-physics-20261014-df89d796-4673-445f-b003-d65f6dc689e5 \
  --samples 101
```

Evidence files: `result.json`, `static_scan.json`, `static_scan.csv`, `constrained_candidate.xml`. The candidate model SHA-256 is `7157e89eaf030f1f987fc4f7b723b3fee1e7e5bbc0c541d326121e1e0cc0f35a`; result JSON SHA-256 is `89b39eaee910e002f6d615539330e367617b95698099411c3a26e49ad34a2f4e`.

## Next evidence needed

Do not tune equality stiffness or invent passive limits from the current fit. The next discriminating input is a source-grounded pivot/clearance definition: vendor CAD/drawing or independently measured pivot axes, pin centers, and pin/bore clearance for the X2 assembly. With those, rebuild the isolated candidate and repeat only the static full-range closure check. Run constrained dynamics only after the geometry can be shown to close within documented clearance and without unsupported passive limits.
