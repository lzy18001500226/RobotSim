---
name: robotsim-mujoco-model-diagnostics
description: Diagnose RobotSim MuJoCo model identity, mesh and joint frames, actuator mapping, constraints or contact discrepancies from existing evidence and static loading. Use when a robot looks correct but compiled geometry, contacts or motion differ; not for transmission redesign or automatic dynamics trials.
---

# MuJoCo model and contact diagnostics

## Identify the model before explaining behavior

Read the [model-integrity contract](../../../docs/engineering/model_integrity.md), [MuJoCo backend guide](../../../simulation/mujoco/README.md) and affected package's README/manifest. Inspect the actual entrypoint, includes, source hashes, model selection, units and MuJoCo version. Use the [X2 loader and registry contract](../../../simulation/mujoco/models/agibot_x2/README.md) for OmniPicker; do not substitute a historical experiment when loading fails.

Preserve vendor sources and frozen models. Read the smallest relevant saved trace and locate the first failing frame before proposing another run. Confirm that a measurement failure is not being mistaken for a mechanical limit.

## Compare in a common frame

1. Track source STL coordinates through mesh registration, geom-local transform, body/joint transform and world transform. Separate visual and enabled collision geoms. MuJoCo mesh compilation may recenter/rotate mesh data; include the compiled mesh/geom transforms before comparing vertices.
2. Check pivot position, axis direction, parent ownership, zero/reference pose and units. Compare complete matched joint states; a root command alone does not determine every passive coordinate.
3. For closure, use actual model/data site transforms and report residuals in units. A small pin gap does not prove a unique or physically correct passive transmission.
4. For contact, identify both geom/body names, masks, exclusions and proxy types. Compare original triangle surfaces with MuJoCo's enabled collision representation; distinguish intended mating overlap, proxy contact and real source-surface interference.

Do not infer a rubber pad's boundary from a link name or a tip landmark. If source meshes lack material labels, mark selected surface triangles and their IDs as a geometric hypothesis for review. A circle fit or landmark distance is not by itself a collision-safe jaw aperture.

## Keep static and physical evidence separate

Static loading and `mj_forward` can verify transforms, initialized constraints and contact candidates; they do not integrate a grasp. Offline poses may assign qpos only when explicitly labeled kinematic. An authorized physical trial uses actuator commands and `mj_step`, with no runtime follower/object qpos replay. This Skill never grants permission for such a trial.

For an authorized force diagnosis, distinguish `ctrl`, joint state, actuator force/saturation, bias/passive forces and constraint/contact forces. Use saved `mj_contactForce` telemetry where available; a static proxy touch does not prove load transfer or lift. Do not weaken a failed gate or disable contacts to produce a pass.

## Return the smallest reviewable result

Report model/source identities, affected frames/geoms, first discrepancy, measured units, command and evidence path, supported correction and remaining uncertainty. Use [validation planning](../robotsim-validation-planning/SKILL.md) and the [evidence-path rules](../../../docs/setup/evidence-path-governance.md). Do not promote visual acceptance, static equivalence or an old model's dynamics result to qualification of another model.
