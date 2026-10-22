# Issue 46: official URDF / reviewed CAD overlay delivery

This is an isolated `UNVERIFIED_EXPERIMENTAL_MODEL`, not the accepted X2 runtime.
It completes the missing source chain: pinned official full URDF → derived full
URDF → parse its right gripper subtree → generated MJCF → real physics.
Vendor files, existing CAD/viewer artifacts and other worktrees are unchanged.

## Source and exact changes

- `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`,
  `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`, SHA-256
  `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- Six right-hand joints become continuous experimental hinges:
  `R_hand_{narrow,wide}{2,3}_joint` and `R_hand_{narrow,wide}_loop_joint`.
  Their official origins/axes are retained and compared with CAD evidence.
  Fixed-joint position placeholders are removed; effort/velocity metadata is
  explicitly simulator-only. Left hand, wrists, root axes/ranges/mimic are intact.
- The visually accepted measured mesh-local registrations are applied to both
  visual/collision origins: narrow3 `+4.213496°`, narrow_loop `+11.940984°`,
  wide3 `−4.213431°`, wide_loop `−11.940920°`. No STL vertices/scale are changed.
- Two meshless right D reference-frame positions use the reviewed PR #84
  coordinate re-expression. Loop-side D anchors remain in the MJCF overlay.
- All original inertial values are retained. The geometry registrations have not
  been validated against real hardware mass distribution.
- Mesh paths are rebased to immutable local source files. Full and isolated
  descriptions are deliberately separate: only the right hand is stepped here.
  The original vendor URDF has zero root effort/velocity metadata; actuation is
  supplied by the simulator overlay, not silently claimed as vendor calibration.

The generated `official_to_derived.diff` preserves unchanged upstream text;
`result.json` enumerates the deltas and hashes. No model/output binaries enter Git.

## Controller and collision assumptions

One normalized opening command computes root + six follower targets using the
**existing** Candidate B theta2–theta4 source-phase relation. This is an
experimental relationship, not a transmission established by STEP. The previous
endpoint's linear angle extrapolation is replaced by analytic evaluation of that
same relation. No new closure branch, gear ratio or tendon equality was invented.
Root mapping stays `q_URDF = -0.13412598906686654 - q_CAD`.

Seven position servos use the prior experimental `kp=0.5`, `kv=0.05`, ±0.5 Nm
caps; the opposite root remains an unactuated official mimic follower. Servos
are not CAD measurements. The two existing D connects, four inherited internal
body-pair exclusions, and hashed source-STL CoACD approximations are unchanged.
External object contact remains enabled. These collision approximations and
exclusions are not manufacturer-validated internal mechanics.

## Observed validation on 2026-10-22

Local outputs below are relative to the canonical WSL `reviews/issue-46/`.
They remain ignored local evidence, not durable public download links.

| Check | Result and evidence |
| --- | --- |
| Official-source integrity / scoped edits | PASS; pinned inputs, preserved inertias/mesh identity, left hand and root contract |
| Independent URDF FK vs actual MuJoCo FK | PASS at 21 poses; position error ≤1.66e-17 m, rotation-matrix error ≤6.67e-16 |
| Prior sampled range, q_CAD [−0.10,+0.20] | 6,000 `mj_step`, 6 s, gravity on; max D gap 0.00840 mm, penetration 0.18854 mm, speed 0.54269 rad/s. `official-urdf-cad-pipeline-20261022T065759Z/` |
| Continuous source-start → open → root-limit end | 6,000 steps, gravity on; D gap 0.04460 mm, penetration 0.54014 mm, speed 0.58186 rad/s; target error 0.05627 rad. `official-urdf-cad-pipeline-20261022T070211Z/` |
| Root-limit precision | Commands obey official ranges; observed soft-limit excursion up to **0.000887 rad** in the logged trace. The run passes its explicit 0.001 rad numerical stop tolerance; this is **not zero limit violation** |
| Abrupt closed-limit initialization | FAIL at step 9, wide root −0.001102 rad. Preserved in `official-urdf-cad-pipeline-20261022T065135Z/`. The later test starts from the mapped STEP source instead; it does not erase this failure |
| Full physical jaw closure | NOT VERIFIED: at root zero, minimum jaw **vertex** distance remains 5.62745 mm, vs 50.15438 mm at the tested open end. This is not exact gripping-surface aperture, manufacturer minimum travel, or a verified hardware constraint |
| Canonical bottle fixture | PREFLIGHT_BLOCKED: reusing the old neck-grasp fixture produces 9.08933 mm initial MuJoCo proxy penetration between `bottle_shoulder` and `wide3_Link_coacd_00` (also 3.55630 mm against piece 01). No valid bottle lift result |

The first bottle harness version detected the bad initial placement after **one**
step, recorded in `urdf-derived-canonical-bottle-20261022T070644Z/`. That was too
late for an initial-contact gate. The corrected harness checks before stepping:
`urdf-derived-canonical-bottle-20261022T070906Z/` has **zero** steps. Both records
are preserved. The incorrect fixture was not retuned and no collision was hidden.
The probe's first compile also exposed missing bottle inertia under the inherited
`inertiafromgeom=false`; `auto` now derives only absent fixture inertia, with an
explicit invariant that all gripper masses/inertias/COM frames remain identical.

All physical runs set joint qpos only for initialization. There are no per-step
qpos writes, object attachments, mocap or hidden support welds. Safe motion is
useful evidence for the generated model; it is not bottle pickup acceptance.
Two early setup-only output directories (renderer frame size / fixture inertia)
remain preserved as incomplete diagnostics rather than completed runs.

## Reproduce / review

Use the existing WSL native MuJoCo **3.3.6**, NumPy, SciPy and Pillow environment.
No install, vendor extraction or canonical-model update is required.

```bash
cd /home/lzy18001500226/robotsim-issue46-official-urdf-articulation
PY=/home/lzy18001500226/.venvs/robotsim-mujoco336/bin/python
$PY -m unittest discover -s scripts/research -p test_issue46_urdf_pipeline.py -v
MUJOCO_GL=egl $PY scripts/research/issue46_urdf_pipeline.py --motion source-window
MUJOCO_GL=egl $PY scripts/research/issue46_urdf_pipeline.py --motion root-endpoint
# Exact tested model, native MuJoCo window + local scalar slider, gravity on:
MUJOCO_GL=glfw $PY scripts/research/issue46_urdf_review.py \
  --model-run /home/lzy18001500226/projects/RobotSim/reviews/issue-46/official-urdf-cad-pipeline-20261022T070211Z
# Then open http://127.0.0.1:8781/ . 0 means root-limit end, NOT fully touching jaws.
# Optional existing bottle-fixture preflight (currently stops before mj_step):
MUJOCO_GL=egl $PY scripts/research/issue46_urdf_bottle_probe.py \
  --model-run /home/lzy18001500226/projects/RobotSim/reviews/issue-46/official-urdf-cad-pipeline-20261022T070211Z
```

Each generation allocates a unique governed directory, emits the full derived
URDF, MJCF, exact diff, hashes, static checks, trace, PNG/GIF physics evidence and
verified artifact manifest. Native review starts at the mapped STEP pose, keeps
the same collision physics, and freezes at the recorded safety thresholds.

Next engineering work is a collision-free bottle pregrasp frame and measured
bilateral load transfer with this generated model. Do not infer zero jaw gap or
change official limits/geometry to force it. Full X2 arm integration, manufacturer
travel calibration, hardware, bottle lift and production promotion remain open.
Temporary implementation disposition: KEEP-AS-TOOL for this bounded research
pipeline pending independent review; no production dependency is introduced.
