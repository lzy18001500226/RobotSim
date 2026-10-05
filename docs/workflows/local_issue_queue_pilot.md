# Local Issue Queue Pilot

This document describes the bounded, single-host pilot authorized by Issue #49. It is a foreground
local command, not a daemon or distributed scheduler. GitHub Issues own readiness, dependencies,
human gates, and completion; GitHub PRs own branch, CI, and review state. SQLite stores only local
writer leases, retry fingerprints, private review feedback, and an exact pending closeout payload so
the same local queue can recover after a restart.

## One-time setup

Use a Linux or WSL2 shell with a clean RobotSim checkout and `git`, `gh`, Python 3, and Codex CLI
available. Process-restart recovery currently reads Linux `/proc`, so native Windows and macOS
dispatch are unsupported. Authenticate `gh` through the normal local GitHub identity without
copying credentials into repository files.
Create these labels in the repository if they do not exist:

- `agent:ready`, `agent:retry`, `agent:running`, `agent:blocked`, `agent:review`,
  `human:retry-approved`
- `human:visual`, `human:repro`, `human:architecture`, `human:security`, `human:hardware`
- `human:visual-approved`, `human:repro-approved`, `human:architecture-approved`,
  `human:security-approved`, `human:hardware-approved`

Mark an eligible Issue `agent:ready`. Dependencies use explicit body lines such as
`Depends on: #12, #18`; every referenced Issue must be closed. Add `human:architecture`,
`human:security`, or `human:hardware` before dispatch when that approval is required before work
starts. The matching `human:<gate>-approved` label releases that pre-execution stop; the queue
consumes those approvals when the writer starts so they cannot also approve its resulting code.

## Run one bounded cycle

Run from the RobotSim repository root:

```bash
python3 scripts/agent/task_queue.py --list-ready
python3 scripts/agent/task_queue.py --plan-issue 123
python3 scripts/agent/task_queue.py --dispatch --reconcile
```

The pilot allows at most two local writable Codex slots; `--slots 1` reduces the default bound. The queue
sorts ready Issues by number, checks dependencies and running labels, and atomically claims local
SQLite slots. It creates one stable `issue/<number>-task` branch and worktree per Issue under
`~/.cache/robotsim/task-queue/worktrees`. It refreshes `origin/main` and renders the Issue with
`AGENTS.md` and the Goal workflow read from that fetched revision. A task branch cannot replace the
guidance used to start its next attempt.

Each writer is one `codex exec` process in the issue worktree, with Codex's `workspace-write`
sandbox. No approval-bypass option is used. A separate Codex review uses `read-only`, holds no
writer slot, and runs only when the local worktree is clean and exactly matches the PR head. The
queue records result summaries and failure fingerprints, not agent transcripts. Review feedback is
kept in the user-only SQLite state file and passed back to the same Issue's next attempt.

The command waits for its selected batch. Run another `--dispatch --reconcile` cycle after the
existing Auto-PR workflow has created/updated PRs or after new checks/reviews arrive. `--reconcile`
alone also refreshes `origin/main` and updates stored PR state. `--plan-issue` prints a decision and
workspace plan without creating a worktree or dispatching Codex.

## Retry and human gates

The same root-cause fingerprint on a second attempt becomes `agent:blocked`. The queue also caps
an issue at three total primary attempts between maintainer releases, so changing the reported
root cause cannot create an endless retry loop. A repeated failure or exhausted attempt budget
triggers an explicit RobotSim task-closeout notification. Failed closeout delivery retains the
original event payload and retries it on the next queue cycle; the command reports notification
failure with a nonzero exit instead of silently treating it as delivered.
Review comments and failed check names are routed to the next primary attempt. A repeated identical
review/CI root cause also blocks. Inspect the branch and Issue before removing `agent:blocked`,
marking it ready again, and adding `human:retry-approved`. The pilot consumes that label at dispatch.

After CI and both GitHub and read-only review approve the PR, the queue creates a local and Issue
comment review packet containing the branch, head SHA, check results, and required human gates. A
Unity scene/prefab/material/shader change stops at `human:visual`; a requested reproduction stops
at `human:repro`; architecture, security, and hardware/safety changes carry their matching gates.
Add the matching `human:<gate>-approved` label after that review. The queue may then report
`merge_eligible`; it never merges. `done` is derived from the canonical GitHub Issue becoming
closed; there is no separate `agent:done` label. GitHub branch protection and the repository's
standing merge policy still apply. There is no Codex Stop-hook dependency: actionable blocked and
human-review transitions explicitly call `scripts/agent/notify_task.py task-closeout`.

The trusted Auto-PR workflow now adds `Closes #<number>` only when the `issue/<number>-...` branch
match is unambiguous. A dependent Issue becomes eligible only after its dependency is actually
closed in GitHub. PRs created before that Auto-PR change is active may need their originating Issue
closed manually after merge. No queue code merges or closes an Issue on its own.

## Limits

The SQLite leases and two-slot bound apply to one local state directory on one host; this is not a
cross-host lock service. Do not run multiple independent queue coordinators against the same Issues
at once. GitHub's `agent:running` label is an additional visible stop, not an atomic distributed
lease. Use ephemeral worker IDs generated per attempt; do not assign permanent Local01/02/03/04
identities.

This Cloud implementation was validated with repository-side tests and static checks only. It does
not establish local WSL/Codex/MuJoCo dispatch E2E behavior. The maintainer must validate that flow
on the intended local machine before relying on the pilot for real tasks.
