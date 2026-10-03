# Repository Agent Instructions

## Task ownership and authority

- The current user's request defines the task, scope, and acceptance criteria. Repository documents guide the work but do not create or authorize additional tasks.
- The primary agent owns the task from inspection through implementation, validation, and Git/PR closeout. Normal development uses one primary agent thread.
- Subagents are optional and limited to independent, bounded research or review. They do not own the task or edit shared project files; the primary agent reviews and validates their input.
- Follow higher-priority instructions and the current user's directions when they conflict with repository guidance.

## Minimal startup and reading

1. Read this file and the repository `README.md`.
2. Use [`docs/README.md`](docs/README.md) to locate relevant project documents.
3. Read only the source, workflow, checklist, skill note, or decision record needed for the current task. Do not load the whole documentation tree by default.

Project plans and workflows provide context or procedures; they are not task queues or authorization to expand the current request.

For implementation placement and RobotSim runtime/validation rules, follow [`docs/engineering/README.md`](docs/engineering/README.md). For multi-step work that needs persistent continuation, follow [`docs/workflows/goal_driven_development.md`](docs/workflows/goal_driven_development.md); Goals are thread-scoped and do not authorize actions beyond the user's request.

## Architecture boundaries

- MuJoCo is the source of truth for robot physics and simulation ground truth.
- Unity provides high-fidelity scenes, rendering, external sensor simulation, and visualization. It must not become a second robot-physics authority.
- ROS 2 is the system integration layer for sensor and robot interfaces and higher-level capabilities; sensor simulators must expose ROS 2 compatible interfaces.
- Unitree G1 is the first robot backend; AgiBot X2 is the second. Keep vendor-specific SDK logic inside `robots/unitree_g1/` or `robots/agibot_x2/` adapters. Do not edit upstream vendor robot files in place.
- Keep simulation ground truth separate from estimated state. Algorithms must be runnable with a simulated backend before real deployment, and robot backends must preserve common interfaces.
- Treat Dockerfiles and Compose files as the source of truth for the reproducible headless/runtime baseline. Unity, WSLg, and GPU integration remain host-side capabilities documented separately; the container baseline does not reproduce them. Treat rosdep/apt as the ROS dependency source and pinned commits in `third_party/LOCK.md` as the third-party version source.

Read the relevant record in `docs/adr/` before changing an established architecture boundary. Do not change these boundaries as incidental task cleanup.

## Workspace and change boundaries

- Inspect the current branch, working-tree status, and relevant diff before editing. Preserve unrelated user changes.
- Work on a task branch; do not commit task changes directly to `main`.
- Keep edits within the paths needed for the current request. Never use broad staging, reset, clean, or force-push operations to remove or hide unrelated work.
- Do not rewrite unrelated technical documentation or broaden scope without direction from the current user.

## Execution and validation

- Inspect the smallest relevant source and documentation set before editing. Follow the applicable setup, workflow, skill, and checklist documents.
- Make the smallest change that satisfies the task, then run focused checks relevant to the changed paths.
- Report which checks ran and their results. Distinguish static checks from environment, simulation, or hardware validation; do not claim checks that were not run.
- If a required check or Git operation is blocked, preserve the work and report the specific blocker and remaining validation gap.

## Documentation responsibilities

- Put durable architecture decisions in `docs/adr/`, setup procedures in `docs/setup/`, repeatable task procedures in `docs/workflows/`, focused technical guidance in `docs/skills/`, and verification lists in `docs/checklists/`.
- Update the document that owns a changed rule or procedure. Link to existing guidance instead of duplicating it, and do not add progress documents just to narrate routine work.

## Git and PR closeout

- At task closeout, follow the [Codex closeout workflow](docs/setup/codex.md#closeout). Local Codex's trusted project `Stop` hook invokes `scripts/agent/notify_task.py` once per turn; Codex Cloud must invoke it explicitly once before its final answer because project hooks are not assumed to run there. Use the actual terminal outcome. Missing notification configuration or delivery failure is best-effort and does not change the task result.
- Routine, low-risk tasks have standing authorization to self-review, commit, open a PR, wait for required checks and any configured independent review, and merge once those gates pass. Do not ask for an additional per-merge confirmation.
- Stop at a reviewable PR for architecture decisions, destructive or high-risk work, dependency/security changes, unresolved GUI/hardware validation, and explicitly review-gated tasks.
- Never bypass branch protection or force merge. Explicit current-user instructions take precedence.
- For repository changes, review the scoped diff, run relevant checks including `git diff --check`, and stage only task-owned paths.
- Verify the pushed branch and PR. Do not report closeout as complete if commit, push, or PR creation failed.
