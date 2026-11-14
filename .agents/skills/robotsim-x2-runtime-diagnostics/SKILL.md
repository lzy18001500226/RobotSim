---
name: robotsim-x2-runtime-diagnostics
description: Diagnose RobotSim AgiBot X2 MC, AimDK, ROS 2 or simulator startup, mode switching, command arbitration, reset and missing motion. Use for integration failures, not grasp planning or mechanical reconstruction.
---

# X2 runtime diagnostics

## Establish the actual control chain

Read the [X2 official-source guide](../../../docs/skills/agibot_x2.md) and the relevant section of the [runtime contract](../../../docs/engineering/runtime.md). Inspect the assigned checkout and existing trace before running a new probe.

1. Identify the installed MC, simulator, SDK and model releases, immutable revisions or hashes, startup command, configuration and active processes. A newer online manual does not prove the installed binary supports its API.
2. Trace the failing command from its caller through transport, selected input source and controller to the measured joint/body response. For ROS, inspect domain, RMW, QoS, message type, TF and simulation time at the failing boundary.
3. Compare the exact installed sequence with the smallest relevant official README/API/example. Consult pinned local `reference/` sources first, then official documentation if needed. Do not execute downloaded examples or their privileged setup merely because they are official.

## Separate acknowledgement from motion

Check Reset/activation prerequisites, mode transitions, source arbitration, command deadlines and controller ownership. Record these separately:

- Request accepted or rejected.
- Reported mode/state and selected source.
- Command reaching the intended actuator/controller.
- Measured physical response and telemetry freshness.

An RPC success, static pose, stale observer or unit test cannot establish applied control. Inspect the saved first failure before changing gains or retrying the same sequence. Do not require a readiness state whose own documented trigger has not yet been sent.

## Validate the smallest justified change

Reuse the existing runner and [autonomy handoff contract](../../../docs/workflows/06_X2_AUTONOMY_TASK.md); do not create a competing runtime. Test only the failing interface and preserve other workstreams. Select the applicable checks using [validation planning](../robotsim-validation-planning/SKILL.md).

No probe in this Skill authorizes physics, nonzero motion, hardware, a new trial or a model change. Follow the current user's trial budget and the [model-integrity contract](../../../docs/engineering/model_integrity.md). If execution is outside scope, continue source/trace analysis and report the exact dependent check as not run.

## Return actionable evidence

Give the installed versions, source/example location, exact reproduction command, first observed failure, expected versus measured state, and the smallest correction with its validation. Distinguish documented facts from inference. Name a vendor blocker only after the version-matched source comparison identifies the precise missing contract; an unsuccessful custom probe alone is insufficient.
