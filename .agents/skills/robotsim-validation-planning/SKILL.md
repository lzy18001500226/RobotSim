---
name: "robotsim-validation-planning"
description: "Use when planning or reviewing validation for a RobotSim change; select required evidence levels, gates, and honest result statuses."
---

# RobotSim validation planning

Use this Skill when planning or reviewing validation for a RobotSim change. Inspect the requested behavior and actual diff, including affected runtime boundaries; do not choose levels from filenames alone.

## Select evidence

Use the [validation contract](../../../docs/engineering/validation.md) as the source of minimum levels and add any level whose boundary the change touches. Read the [runtime contract](../../../docs/engineering/runtime.md) for runtime, timing, frame, or backend questions. Use the [environment checklist](../../../docs/checklists/environment.md) and [reproducibility checklist](../../../docs/checklists/reproducibility.md) when setup or repeatability affects the claim. Reuse these sources instead of copying their procedures.

Report each applicable level separately with the command, environment/version, result, and evidence. Use these statuses:

- `PASS`: the check ran and passed in the named environment.
- `FAIL`: it ran and failed; retain the useful failure evidence.
- `NOT RUN`: it was not attempted.
- `DEFERRED`: it is required but cannot currently run because an environment, capability, permission, dependency, or human gate is unavailable. Name the blocker and exact unlock action. Do not use this for a check that was merely skipped.
- `MANUAL PASS`: a person performed a check that cannot credibly be automated; name the setup and observed result.

Lower-level results never establish higher-level behavior. A build or unit test does not establish simulation or ROS integration; headless MuJoCo does not establish Unity rendering; simulation success does not establish hardware behavior or L5.

## Stop at gates

- For an accepted architecture boundary, identify the affected ADR, prepare a scoped proposal if requested, and stop for human review. Do not settle unresolved architecture in a validation plan.
- For Unity visual behavior, require the applicable L4 check. If local GUI/GPU validation is unavailable, report L4 `DEFERRED` with the exact local action; use `MANUAL PASS` only after a person observes the result.
- For hardware or HIL, require applicable lower-level evidence and the named-target safety/human gate for L5. Do not actuate hardware.

## Sanity cases

These examples check that the selected evidence matches the touched boundary; the validation contract remains authoritative.

| Change or result | Required interpretation |
| --- | --- |
| Python repository tooling | L0 + L1 |
| MuJoCo model, physics, or runtime | L0 + L1 + L2 |
| ROS integration change | L0 + L1 + L3; add L2 when the change also affects simulation behavior |
| Unity change without local GUI/GPU access | L4 is `DEFERRED`; name the local validation action |
| Simulation succeeds | Report only the simulation evidence; it is not L5 |
