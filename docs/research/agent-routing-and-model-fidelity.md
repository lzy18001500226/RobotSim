# Agent Routing and Model Fidelity Review

## Routing audit

The App exposed Local04 with the WSL RobotSim checkout as its working directory. Local01 was a separate thread rooted under the Windows `Desktop/Robot` folder. Both threads exposed the same project ID, so that value is not a unique worker or thread identity. The project catalog did not return a matching project record with a source folder.

The misrouted robot task arrived in Local04 as a direct user message. No queue `run_id`, `attempt_id`, worker ID, or dispatched task packet was attached to that request. The read-only queue status command found no state database at its configured default path. There is no evidence that the Issue #49 dispatcher assigned that robot task to Local04. The observed cause is manual task placement in the wrong chat; repository code cannot prevent arbitrary task text from being pasted into another manually selected App thread.

The queue can enforce ownership only for queue-launched work. Its durable packet binding checks the Issue/run/attempt/worker tuple plus branch, workspace, saved session, and allowed write roots before executor handoff and publication. A session ID already recorded for another Issue/run is rejected. This does not map `Local01` or `Local04` labels to App thread IDs and does not make a shared App project ID an owner key.

## Task closeout behavior

The existing `robotsim.task-closeout.v1` fields distinguish implementation from experiment, research, review, and audit work. No schema extension is needed:

- A completed implementation with a PR is posted to that PR and requires its description to close the originating Issue.
- A completed review with a PR is posted to the PR without requiring a closing keyword.
- Experiment, research, and audit outcomes are posted to their originating Issue, including when a PR is supplied as evidence.
- Blocked, deferred, failed, and cancelled outcomes are posted to the Issue and never require a PR body reference.
- Whenever `pr_number` is present, the notifier still verifies the canonical RobotSim PR, branch, and exact head SHA before writing.

This keeps implementation acceptance distinct from experiment completion and lets a blocked experiment record persist without changing or relying on an unrelated PR description.

## Model fidelity and morphology proposal

Keep this as a review checklist, not an automatic disposition or pass/fail quality score. A future model-import or end-effector review should capture:

1. **Provenance:** upstream repository and pinned revision, asset/license note, source file hashes, conversion tool revision, and local transforms.
2. **Tool topology:** one explicitly named accepted tool at each intended wrist; detect a replacement beside an undeleted stock hand. Record the wrist mount separately from the hand subtree.
3. **Morphology:** left/right link and joint names, parent chains, ordering, axes, ranges, and mimic relations. Compare sides and record every intentional asymmetry rather than silently dropping a side or relation.
4. **Visual and collision representation:** report visual meshes/materials separately from contact-enabled collision geoms. Primitive collision proxies may be valid; they do not count as a visually faithful mesh representation.
5. **Command versus motion:** retain commanded targets and measured joint/body motion in the same trace. Accepted command input is not evidence that the robot moved or completed a task.
6. **Evidence:** fixed camera views for both sides and the end effector, plus a compact manifest of model/source hashes and test outputs. Keep large renders and raw traces outside Git.

The two tests in `tests/test_model_fidelity_poc.py` are fixture-only proofs of concept. They demonstrate duplicate-tool detection and separation of mesh-backed visual evidence from primitive collision geometry. They do not inspect, certify, or change any real RobotSim robot model.
