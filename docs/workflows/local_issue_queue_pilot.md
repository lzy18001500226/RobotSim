# Local Issue Queue Pilot

This document describes the bounded, single-host pilot authorized by Issue #49. The local WSL2
watcher provides unattended continuation; it is not a distributed scheduler. GitHub Issues own
readiness, dependencies, human gates, and completion; GitHub PRs own branch, CI, and review state.
SQLite stores local writer leases, stable run/worker identity, Codex session IDs, attempt history,
retry fingerprints, private review feedback, exact branch SHAs, and pending closeout payloads.

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
python3 scripts/agent/task_queue.py --show-state 123
python3 scripts/agent/task_queue.py --status-report
python3 scripts/agent/notify_task.py config-preflight
```

For unattended continuation through Auto-PR and CI updates, use one watcher for the shared state
directory:

```bash
python3 scripts/agent/task_queue.py \
  --watch --slots 2 --poll-seconds 20 --watch-issue 123 \
  --stop-at-review-checkpoint
```

Remove `--watch-issue 123` to process all ready Issues. Send SIGINT or SIGTERM for a clean shutdown;
the watcher stops taking new work and waits for its current bounded batch to finish. A state-directory
file lock prevents a second watcher from creating another local writer pool. Use the same `--state`
path after a restart so saved leases, Codex sessions, attempts, and SHA/checkpoint history are
recovered. A retry resumes the persisted deterministic worktree even if the configured worktree root
changes, preserving uncommitted task changes;
result files use each attempt ID so an interrupted run cannot collide with its retry. A hard
`agent:blocked` state cannot be retried until a maintainer removes the block and explicitly adds
`human:retry-approved`.

The pilot allows at most two local writable Codex slots; `--slots 1` reduces the default bound. The queue
sorts ready Issues by number, checks dependencies and running labels, and atomically claims local
SQLite slots. It creates one stable `issue/<number>-task` branch and worktree per Issue under
`~/.cache/robotsim/task-queue/worktrees`. It refreshes `origin/main` and renders the Issue with
`AGENTS.md` and the Goal workflow read from that fetched revision. A task branch cannot replace the
guidance used to start its next attempt.

Each writer is a `codex exec` process in a standalone issue checkout, with Codex's `workspace-write`
sandbox. Its Git metadata stays under that task directory, so the writer needs no extra writable
roots for shared refs or objects. No approval-bypass option is used. A clean legacy linked checkout
is migrated at the same path and branch, with ignored files copied; a dirty checkout or ambiguous
migration state fails closed for inspection. Checkout setup carries only allowlisted local author
and signing settings and rejects remote URLs with inline credentials; authentication must come from
the normal credential helper or SSH agent. A separate Codex review uses `read-only`, holds no writer
slot, and runs only when the local checkout is clean and exactly matches the PR head. A retry keeps
the same `run_id`, `worker_id`, issue branch, and Codex session while creating a new
`attempt_id`; Codex resumes that session with the new CI/review feedback. The queue records a
sanitized append-only event history and result summaries, not transcripts. Private review feedback
stays in the user-only SQLite state file.

The one-shot command waits for its selected batch. `--watch` automatically repeats recovery,
reconciliation, and dispatch so a pushed issue branch can flow through the trusted default-branch
Auto-PR workflow and required CI without a maintainer pasting a prompt into a chat. Before marking an
execution complete, the watcher requires a clean issue worktree and verifies the same full SHA on
`origin`. It records discovered PR number, head SHA, CI state, review state, and human gates, then
stops at `awaiting_review`/human-review states when `--stop-at-review-checkpoint` is set. The queue
never merges. `--reconcile` alone refreshes `origin/main` and updates stored PR state. `--plan-issue`
prints a decision and workspace plan without creating a worktree or dispatching Codex.
`--status-report` prints ready/running/blocked/CI/human-gated task summaries, each task's worktree
and current/remote SHA, retry fingerprint, last successful checkpoint, notification state, and a
reproduction command. It omits prompts and review feedback. Closeout validation runs from the exact
task worktree and checks the deterministic Issue branch, local HEAD, remote HEAD, and any associated
PR head before publishing the Issue record.

## Retry and human gates

The same root-cause fingerprint on a second attempt becomes `agent:blocked`. The queue also caps
an issue at three total primary attempts between maintainer releases, so changing the reported
root cause cannot create an endless retry loop. A repeated failure or exhausted attempt budget
triggers an explicit RobotSim task-closeout notification. Failed closeout delivery retains the
original event payload and retries it for at most three attempts across queue cycles. The durable
notification state distinguishes sent, skipped, failed, blocked, and exhausted outcomes; a skipped
AgentMail delivery does not erase the persisted GitHub closeout. The command reports retryable
notification failure with a nonzero exit instead of silently treating it as delivered.
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

The SQLite leases, runner lock, and two-slot bound apply to one local state directory on one host;
they are not a cross-host lock service. Do not run multiple independent queue coordinators against
the same Issues. GitHub's `agent:running` label is an additional visible stop, not an atomic
distributed lease. The queue's stable worker ID identifies one Issue run only and is not a permanent
Local01/02/03/04 identity.

This pilot requires a local WSL2/Linux host with the documented Python, Git, `gh`, and Codex CLI
environment. CI validates queue behavior but does not establish a successful remote GitHub
Auto-PR/CI path; record the exact Issue, Codex run, PR, CI, and human-review checkpoint from the
local E2E separately.
