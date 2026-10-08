# Issue #46: OmniPicker Grasp Topology Redesign

## Verdict

**NO-GO FOR TESTED CONFIGURATIONS**: none of the three bounded grasp configurations passed canonical 70 mm static feasibility. The best witness-derived wrist relief still has 37.143 mm of wrist/loop overlap, and its two jaw distances remain +6.030 mm and -7.632 mm. The physical rollout was correctly not started.

This result does not establish global infeasibility. It identifies an obstruction in the tested isolated poses and calls for a model/design decision before physical validation.

## Scope and identity

- Branch: `research/issue46-omnipicker-collision-audit-20261008`
- Starting HEAD: `2ebf748bf421898dfac9b3a080665837cde27341`
- Vendor: `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`
- Model: `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`
- URDF SHA-256: `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`
- MuJoCo Python/native: `3.3.6`
- Native library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`
- Python: `3.10.12`; platform: WSL2 Ubuntu 22.04; `MUJOCO_GL=egl`
- Accepted paired-jaw controller SHA-256: `9ed3523b397d8cc009d437e445aeea8e027451dfd4846ead06cbb2a43629b28f`
- Source collision meshes, source limits, controller parameters, bottle, table, contact, and friction parameters were unchanged.

## Collision and kinematic audit

The corrected isolated extraction is rooted at `right_elbow_link` and preserves the complete chain through `right_wrist_yaw_joint` into the wrist and OmniPicker. The source yaw/pitch pair has a measured signed overlap of `-39.217 mm`, but the source-valid pair filtering produces no active contact for that pair. Source collision-model correctness therefore passes for the audited pair.

The prior wrist-yaw-root extraction is not used. The canonical bottle remains 70 mm diameter and 0.570 kg. Aperture measurements reject inconsistent distance-query witnesses instead of passing them into interpolation or optimizer inputs. The run's witness-integrity record contains the rejected samples.

## Candidate results

Distances are signed: positive is separation and negative is overlap. Jaw values are ordered narrow, wide.

| Configuration | Minimum closed wrist/loop clearance | Jaw distances | Radial normal dot | Result |
| --- | ---: | ---: | ---: | --- |
| `distal_lower_body` | -43.497 mm | +12.443 / -12.348 mm | -0.9334 | Fail |
| `central_body_reference` | -43.222 mm | +12.415 / -12.376 mm | -0.9333 | Fail |
| `witness_derived_wrist_relief_tilt` | -37.143 mm | +6.030 / -7.632 mm | -0.9977 | Fail |

Each candidate also fails at least open-bottle clearance, approach corridor, two-jaw contact, closed non-jaw clearance, the static 30 mm corridor, and perturbation margin. No candidate qualifies for dynamic simulation.

## Geometry diagnosis

At the central reference pose, source-STL surface samples overlap the bottle cylinder for the narrow loop (`-9.314 mm`), wide loop (`-34.907 mm`), and wrist-roll link (`-33.478 mm`). Their MuJoCo convex-hull signed distances are respectively `-14.931 mm`, `-37.748 mm`, and `-42.515 mm`. This indicates that the hull increases the measured overlap, but is not its sole cause at this pose. The source-STL check samples mesh vertices, triangle centroids, and edge midpoints; it is diagnostic and can miss intersections between samples.

The source wrist-pitch and wrist-yaw samples clear the cylinder, while their compiled hulls also clear it. The most restrictive geometry is therefore localized to the wrist-roll and OmniPicker loop structures in the tested grasp region. The witness-derived tilt improves jaw normal opposition, but does not create collision-free wrist/loop clearance or simultaneous jaw contact.

The diameter map is diagnostic only. Reducing the diagnostic cylinder to 50 mm still gives about `-32.570 mm` critical clearance and jaw distances `+22.415 / -2.376 mm`; it does not provide a valid two-jaw grasp. No altered diameter was used for acceptance.

## Validation boundary

- Source collision-model correctness: **PASS** for the audited source-filtered wrist pair and corrected chain extraction.
- Canonical 70 mm static feasibility: **FAIL** for all three tested configurations.
- Actual bilateral physical grasp: **NOT RUN**.
- Measured load transfer: **NOT RUN**.
- Actual 30 mm bottle lift: **NOT RUN**.
- Full X2 right-arm integration: **NOT RUN**.

The detailed candidate poses, raw optimizer evaluations, clearance maps, witness checks, diagnostic source-mesh comparison, images, runtime identity, and commands are preserved under:

`C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\omnipicker-grasp-topology-redesign\run-20261008-3\`

The earlier run directories are retained unchanged. Run `run-20261008-2` is a provisional geometry pass that exposed a plane-distance query issue; `run-20261008-3` contains the corrected support-plane calculation and the final three configurations. The first launcher-only attempt is also retained and is not experiment data.

## Reproduction

See [`issue46_omnipicker_grasp_topology_redesign_reproduce.md`](issue46_omnipicker_grasp_topology_redesign_reproduce.md). The runner's final SHA-256 is recorded in `runtime_identity` inside `run-20261008-3/result.json`.
