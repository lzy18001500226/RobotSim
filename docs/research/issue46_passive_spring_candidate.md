# Issue #46 — B-Axis Spring Candidate Result

**Result: `EXPERIMENTAL_MUJOCO_FAIL`**\
**Model class: `UNVERIFIED_EXPERIMENTAL`**\
**Bottle contact / grasp / lift: `NOT RUN`**

This follow-up removes the prior Candidate B fixed-tendon equations from a new isolated model and tests whether the STEP-located torsion springs at the two B axes can produce useful one-root-motor motion. The old Candidate B model and its evidence are preserved as a separate hypothesis; this run does not validate that law.

## Source-supported model content

- Official X2 v1.4.0 source: `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`; original URDF/STL hashes are checked by the runner. The original STEP SHA-256 is `5d61f8d7e94fa3c789ddd497e4574aa2a508e24aed447d841585dfdbb8e8c20b`.
- Existing URDF mesh, inertia, root axes/limits, and opposite-root mimic are retained. Six measured downstream joints remain experimental hinges. The two D-point connects remain.
- The source occurrence tree locates torsion-spring product `220000054701` coaxially at narrow B `(-58.990804, 209.775100) mm` and wide B `(23.486015, 225.818328) mm`; see `reviews/issue-46/omnipicker-rk-candidate-20261020T020622Z/step_occurrence_transforms.csv` and `candidate_joint_connectivity.csv`.
- STEP does not identify spring-end ownership, handedness, preload, torque curve, or damping, and does not encode a C-D/palm gear/slot motion law. Applying a torsional spring between outer and middle links at B is therefore explicitly experimental.
- The candidate has one right-root actuator, zero follower actuators, zero passive qpos writes after initialization, two D closures, the official root mimic, and **no Candidate B tendon/equality**. At the registered STEP pose the equality Jacobian leaves 3 system DOFs: the root plus two passive coordinates.

## Sensitivity and physical run

The 9 short `mj_step` sensitivity probes varied B-spring stiffness `0.01/0.05/0.2 Nm/rad` and reference angle `-0.05/0/+0.05 rad`. With the experimental root servo `kp=2`, `kv=0.1`, force cap `±0.5 Nm`, each 0.5 s close-side probe remained within the official root range and had D gap at most `0.0142 mm`; measured fixed-ray section clearance changed from `30.373 mm` at the source pose to `11.945–13.273 mm`. Passive angles varied materially across the spring settings, so the sweep does not identify a vendor calibration.

The final bounded opening/closing attempt used MuJoCo 3.3.6, 1 ms timestep, gravity on, and one root actuator. It executed **3,900 steps / 3.900 s** and stopped at `GRIPPING_SURFACE_SECTION_NOT_MEASURABLE`, before the requested root command `-0.99 rad` (actual root `-0.853938 rad`, command coordinate `86.35%`). The inner-face ray at the fixed measurement section stopped intersecting the jaw mesh. The sampled fixed-ray width rose as high as `101.37 mm` before becoming undefined; that is not a certified jaw aperture or gripper diameter. The sampled aperture was not monotonic in either commanded phase.

At the stop, passive angles were narrow B/C/loop `-0.0978 / +0.6874 / -1.1693 rad` and wide B/C/loop `+0.1532 / -1.8365 / +1.5206 rad`. This is the first concrete physical symptom: the C/loop/palm coordinates are not following a stable opening branch under the B-axis spring hypothesis. Max D gap was `0.043945 mm`, root mimic error `0.001265 rad`, collision-proxy penetration `0.2311 mm`, and max joint speed `16.575 rad/s`. Eight original-STL triangle snapshots found no new pair relative to the registered source-pose baseline; this does not certify the proxy penetration as harmless. Existing exclusions and source geometry were not changed.

The exact sealed run is `reviews/issue-46/passive-spring-single-motor-20261022T114341Z/`. A startup-only attempt (`113328Z`) and a renderer-formatting interruption (`114148Z`) are preserved separately and are not acceptance runs. The latter failed after the jaw-section query returned no intersection; the reader was corrected and the complete failure evidence was captured in `114341Z`.

## Review and reproduction

The interactive review page is in `reviews/issue-46/passive-spring-review-20261022T114906Z/review.html`; it scrubs the **recorded physics video** and synchronized trace. It does not send commands or run physics. The full MuJoCo render is `passive_spring_open_close.mp4` in the same directory.

Reproduce the sealed physics and review export in fresh unique governed directories with:

```bash
cd /home/lzy18001500226/robotsim-issue46-official-urdf-articulation/scripts/research
MUJOCO_GL=egl PYTHONPATH=.:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python issue46_passive_spring_candidate.py
MUJOCO_GL=egl PYTHONPATH=.:/home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/lib/python3.10/site-packages \
  /home/lzy18001500226/.cache/robotsim/issue46-vt-20261007/bin/python issue46_passive_spring_candidate.py \
  --stage review-export --source-run /home/lzy18001500226/projects/RobotSim/reviews/issue-46/passive-spring-single-motor-20261022T114341Z
```

The offline review server used for the delivered page is `http://127.0.0.1:8783/review.html`.

## Engineering conclusion

The B-axis spring location is source-grounded, but a spring force does not define the missing C-D/palm relation. The existing geometry plus D closures leaves two passive branch freedoms after root mimic and root actuation. A one-motor opening percentage can be the external API without being an internal angle lookup, but the model still needs the physical passive response: verified C-D gear/slot/cam engagement and phase, or the actual spring attachment/preload/torque law plus any relevant stop/contact behavior. This candidate does not supply that information and **does not achieve a valid full open/close cycle**.

The least speculative next model change is therefore not a different spring constant: it is to identify the C-D/palm coupling or the real torsion-spring endpoints and calibration in source CAD/manufacturer assembly data. Until then, retain this candidate as a review-only research tool (`KEEP-AS-TOOL` for Issue #46) and do not promote it to the canonical X2 runtime. No official URDF/STL/STEP, grasp/contact setup, or production code was changed.
