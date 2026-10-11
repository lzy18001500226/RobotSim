# Issue #46: OmniPicker Fingertip Collision Proxy

**Result: `PROXY_LATERAL_ESCAPE`**
**Model:** `SIMULATION_ONLY_COLLISION_PROXY`; this is not an official-model or hardware acceptance result.

## Finding

The source-derived proxy passed the static entry/approach checks and maintained bilateral contact through the one-second hold. During controlled preload, the free 0.570 kg cylinder reached the existing 5 mm relative-slip stop before developing load-bearing contact. The run stopped at `PRELOAD_EXCESSIVE_SLIP`; no lift was attempted, and measured lift remained 0 mm.

The proxy therefore did not establish that the original STL fingertip collision shape was the primary cause of grasp failure. It modestly reduced the measured mean lateral resultant relative to the saved original-model run, but contact force directions remained poorly opposed and the available normal force remained insufficient. The next useful diagnosis is grasp/contact orientation and force alignment, not another collision-proxy or parameter search.

## Comparison

| Measurement | A: original source collision | B: source-facet proxy |
|---|---:|---:|
| Evidence | Existing run reused; no rerun | `run-03-validated` |
| Static corridor | Existing validated reference | PASS; no initial robot contact; 4.498 mm minimum non-pad bottle clearance; 75.489 mm gripper/table clearance |
| One-second bilateral hold | PASS in saved baseline | 100% bilateral contact fraction |
| Mean pad normal force | Narrow 0.433 N; wide 0.391 N | Narrow 0.986 N; wide 0.987 N during preload |
| Mean normal-direction angle | 124.023 deg | 134.022 deg (124.538-139.510 deg) |
| Mean lateral resultant on bottle | 1.133 N | 0.588 N; peak 1.165 N |
| Maximum relative slip | 5.003 mm | 5.003 mm; existing stop reached |
| Lift | 0 mm | 0 mm; lift stages not attempted |

For B, mean hand vertical support was -0.003 N while mean table support remained 5.594 N. The mean ideal friction-capacity upper bound from measured normal forces was 2.761 N, below the 5.033 N minimum support gate. Final narrow/wide normal forces reached 1.434/1.349 N, still below the 2.496 N per-pad target. The measured world force-direction angle averaged 146.133 deg, with a wide 99.099-178.348 deg range. The geometry change did not produce reliable opposing load-bearing forces.

Maximum jaw tracking error was 0.01423 rad, below the existing 0.020 rad diagnostic guard. Peak applied driver/follower torques were 0.1410/0.1393 Nm against unchanged approximately 0.225 Nm caps; there was no effort saturation. Maximum pad penetration was 1.023 mm against the unchanged 2 mm gate. There were no source-limit violations, NaNs, bottle qpos writes, bottle welds/equalities, mocap carry, artificial bottle force, hidden support, new collision exclusions, mass/inertia changes, or joint-range changes.

Maximum mimic diagnostic residual was 0.005645 rad. It was recorded, not used as the physical grasp acceptance gate for this proxy experiment.

## Proxy Definition and Limits

The pinned source is `AgiBotTech/agibot_x2_urdf` at `575cc6b988f976c23550e0db85aa1e5475d3652d`; the source URDF SHA-256 is `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`. Only the collision geometries on `R_hand_narrow3_Link` and `R_hand_wide3_Link` were replaced. Their visual meshes, link inertias, joint tree, limits, mimic mapping, actuators and effort caps were preserved. The 11 historical simulation-only exclusions remained documented; this experiment added none.

Each proxy is a convex triangular prism whose outer face is the exact source-STL contact triangle. The narrow and wide source triangles were indices 9087 and 27419, with face areas 0.086143 and 0.321803 mm^2. Seven barycentric interior samples bounded inward extrusion; a 10% depth reserve was applied, yielding thicknesses 0.402273 and 0.450000 mm. Selected outer-face vertices match their source face with 0 measured error. No outward offset, enlarged jaw opening, or extended grasp reach was introduced.

Important limitation: replacing a terminal link's full collision mesh with one local prism also removes that link's other collision coverage in this diagnostic model. Other collision bodies were unchanged. The result is therefore a narrowly scoped contact-shape counterfactual, not a validated replacement collision model. Static and physical results are conditional on this limitation.

## Runtime and Reproduction

- MuJoCo 3.3.6; timestep 0.002 s; integrator 0; solver 2 with 100 iterations.
- Native MuJoCo library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Bottle and table, gravity, contact/friction parameters, controller profile, and actuator caps were held at the existing validated values.
- Exact execution command is recorded in `run-03-validated/result.json`. The executed runner snapshot is preserved as `run-03-validated/executed_runner_snapshot.py`, SHA-256 `c082fe1263fa92fcdd3b26a6c886adb5a34b7e7623b2714c2b4998ce6f09d4cd`.
- The current tracked runner SHA-256 is `a8c06fec93a304d8d06fc2c3fd8434b5b6daacb9f72c30c3f122e49cf5632242`; compared with the executed snapshot, its only changes are a vertex-order-independent geometry metric and corrected experiment failure classification/result reporting. It was not the exact file hash used for the physical run. No physics rerun was performed for those reporting corrections.

```bash
PYTHONPATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib/python3.10/site-packages:/tmp/robotsim-issue46-omnipicker-p0-pin:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages \
LD_LIBRARY_PATH=/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib:/tmp/robotsim-issue46-omnipicker-p0-pin/cmeel.prefix/lib64 \
MUJOCO_GL=egl python3 scripts/research/issue46_omnipicker_collision_proxy.py \
  --output /mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-p0-contact-proxy-20261011/run-04-reproduction
```

The command recorded in `result.json` is the exact invocation used for run 03; the command above selects a fresh output path to avoid overwriting it. The raw physics trace was not modified by the post-run summary tool.

## Evidence

External evidence directory: `C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-p0-contact-proxy-20261011\`

- Geometry overlay: `run-03-validated/proxy_geometry_comparison.png`
- Static scene: `run-03-validated/B2_open_overview.png`, `B2_open_front.png`, `B2_open_side.png`
- Contact and stop frames: `C_first_contact.png`, `C_closed_grasp.png`, `C_hold.png`, `review_display_C_preload_stop.png`, `review_display_C_preload_stop_closeup.png`
- Physical clip: `run-03-validated/C_preload_feedback.mp4`
- Static clearance trace: `B2_CLEARANCE_TRACE.jsonl`; result: `B2_STATIC_GEOMETRY_GATE.json`
- Raw physical trace/result: `C_PRELOAD_TRACE.jsonl`, `C_RESULT.json`; identity and hashes: `runtime_identity.json`, `analysis_summary.json`
- The review-display PNGs apply brightness/contrast adjustments only; the original rendered PNGs are preserved unchanged.

The isolated hand test did not meet the 50 mm lift milestone. Full X2 arm integration was not run. This proxy result must not be described as source-faithful collision acceptance or hardware validation.
