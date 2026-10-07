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

Keep research experiments under `scripts/research/`; production/runtime code must not depend on them. Apply the [runtime prototype lifecycle](docs/engineering/runtime.md#prototype-lifecycle) and [test/evidence retention rules](docs/engineering/validation.md#test-retention-and-generated-evidence), and record a disposition for temporary milestone code before closing that milestone.

## Architecture boundaries

The [engineering contracts](docs/engineering/README.md) and relevant records in [`docs/adr/`](docs/adr/) own architecture, runtime, and evidence details. Read affected records before changing an established boundary; do not treat a proposed ADR as accepted.

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

- At every terminal task result, follow the [Codex closeout workflow](docs/setup/codex.md#durable-task-closeout). Review and commit every intended task-owned change; preserve unrelated work and leave `git status --porcelain` empty except exact, reason-documented unrelated paths. Push the task branch and run `python3 scripts/agent/closeout_gate.py` to verify the local branch/HEAD and exact remote push-ref SHA. A dirty tree, missing push, SHA mismatch, missing GitHub reference, or failed GitHub update means the final status is `CLOSEOUT BLOCKED`, never `DONE`.
- Emit exactly one structured `robotsim.task-closeout.v1` event after that check: Local Codex places its marked envelope in the trusted `Stop` hook's final message; Codex Cloud explicitly calls `scripts/agent/notify_task.py task-closeout` once. The canonical Issue/PR record must include the task result (`completed`=PASS, `failed`=FAIL, `blocked`/`deferred`/`cancelled`=BLOCKED), branch, exact local and remote HEAD SHAs, PR/Issue URL, reproduction command, validation result, durable evidence/output reference, and limitations/blockers. If the local gate fails, the notifier must not persist `completed`; it attempts an Issue-level BLOCKED record instead.
- Create one attempt ID at task start, reuse it and the exact payload for retries, and use a new ID for a genuine rerun. GitHub is canonical; AgentMail is secondary and cannot turn a failed GitHub write into success. The notifier may use `GH_TOKEN`/`GITHUB_TOKEN` or an already-authenticated `gh auth token` path; never print or store credentials. Missing GitHub write authentication means `CLOSEOUT BLOCKED`.
- Routine, low-risk tasks have standing authorization to self-review, commit, open a PR, wait for required checks and any configured independent review, and merge once those gates pass. Do not ask for an additional per-merge confirmation.
- Stop at a reviewable PR for architecture decisions, destructive or high-risk work, dependency/security changes, unresolved GUI/hardware validation, and explicitly review-gated tasks.
- Never bypass branch protection or force merge. Explicit current-user instructions take precedence.
- For repository changes, review the scoped diff, run relevant checks including `git diff --check`, and stage only task-owned paths.
- Verify the pushed branch and PR. Do not report closeout as complete if commit, push, or PR creation failed.
