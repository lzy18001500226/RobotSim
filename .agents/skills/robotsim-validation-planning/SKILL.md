---
name: "robotsim-validation-planning"
description: "Use when planning or reviewing validation for a RobotSim change; select required evidence levels, gates, and honest result statuses."
---

# RobotSim validation planning

Use this Skill when planning or reviewing validation for a RobotSim change. Inspect the requested behavior and actual diff, including affected runtime boundaries; do not choose levels from filenames alone.

## Select evidence

Use the [validation contract](../../../docs/engineering/validation.md) as the source of minimum levels and add any level whose boundary the change touches. Read the [runtime contract](../../../docs/engineering/runtime.md) for runtime, timing, frame, or backend questions. Use the [environment checklist](../../../docs/checklists/environment.md) and [reproducibility checklist](../../../docs/checklists/reproducibility.md) when setup or repeatability affects the claim. Reuse these sources instead of copying their procedures.

Select the minimum validation levels from the contract, adding any level whose boundary the change touches. Report each applicable level with its exact command, environment/version, result, and evidence, using the contract's status definitions.

## Stop at gates

- For an accepted architecture boundary, identify the affected ADR, prepare a scoped proposal if requested, and stop for human review. Do not settle unresolved architecture in a validation plan.
- For Unity visual behavior, require the applicable L4 check. If local GUI/GPU validation is unavailable, report L4 `DEFERRED` with the exact local action; use `MANUAL PASS` only after a person observes the result.
- For hardware or HIL, require applicable lower-level evidence and the named-target safety/human gate for L5. Do not actuate hardware.
