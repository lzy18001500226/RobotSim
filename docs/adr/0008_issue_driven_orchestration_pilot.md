# ADR-0008: Bounded GitHub-Issue-driven orchestration pilot

Status: Accepted

Supersedes: none

Related: [ADR-0006](0006_goal_mode_first.md), [Issue #19](https://github.com/lzy18001500226/RobotSim/issues/19), [Issue #49](https://github.com/lzy18001500226/RobotSim/issues/49)

## Context

ADR-0006 chose native Goal mode first and said to reconsider a repository queue only after repeated task-switching and continuation friction appeared. That condition is now met. The RobotSim issue trail records repeated prompt copying and worker routing, while the maintainer observation recorded on [Issue #49](https://github.com/lzy18001500226/RobotSim/issues/49#issuecomment-6033815256) describes manual conditional queue construction, continuation after bounded failures, recovery after a network interruption, and task state spread across threads, branches, evidence packets, Issues, and PRs.

This evidence is qualitative, not an instrumented frequency or time-saved study. The current repository governance and Auto-PR path are now verified, including the live success/failure E2E recorded on [Issue #32](https://github.com/lzy18001500226/RobotSim/issues/32#issuecomment-6033813033). A small pilot can test whether durable issue state reduces repeated hand-routing without introducing a new general-purpose platform.

## Decision

Propose one bounded, repository-specific orchestration pilot with these constraints:

- GitHub Issues remain the control plane and canonical task record.
- Only explicitly ready issues with resolved dependencies may be dispatched.
- Each issue has one writable primary worker and one deterministic issue workspace/worktree. Independent review is read-only.
- Initial active-work concurrency is at most two issues.
- Dispatch respects declared dependencies and does not start blocked or downstream work early.
- Retries are bounded. If the same root cause recurs twice, mark the issue blocked and require human direction instead of looping or changing ownership silently.
- The durable record contains the current stage, branch/head, last completed checkpoint, validation result, evidence packet location, blocker/retry state, and next action. GitHub plus the evidence packet must be sufficient to recover after thread or network loss.
- Continuation after a bounded FAIL or interruption resumes the same primary worker's durable task state; it must not create a second writable owner for that issue.
- Keep independent review read-only. Preserve human gates for architecture decisions, visual validation, credentials/security, hardware/safety, and destructive work.
- Continue to use required CI, repository rules, and review. The pilot cannot bypass them or merge past a human gate.

This pilot is not a general distributed scheduler, does not keep eight workers permanently busy, and does not make AgentMail or Codex App Stop-hook delivery a correctness dependency. GitHub is canonical; any notification remains secondary to durable state.

## Alternatives

1. Continue manual issue triage and native Goal mode only. This preserves the ADR-0006 baseline but leaves the repeated routing and recovery work manual.
2. Build a general multi-worker scheduler now. This exceeds the observed need and creates unnecessary queue, persistence, and ownership machinery before a narrow pilot has been measured.
3. Adopt the bounded issue-driven pilot. This directly tests the reported friction while retaining a small concurrency limit, one writer per issue, and existing human/CI gates.

## Pilot validation and exit criteria

Implement only after this proposal is accepted. Use one disposable, non-critical Issue to verify readiness/dependency evaluation, deterministic workspace creation, one-owner enforcement, bounded retry/blocked behavior, same-worker recovery from a durable checkpoint, PR/CI closeout, and sufficient evidence for read-only review. Do not use a task requiring hardware, credentials, destructive operations, or unresolved architecture decisions as the first pilot.

Record a baseline and pilot counts for manual routing/continuation touches, interruptions and recoveries, retries, duplicate-owner incidents, queue wait, and time to review. Keep the pilot only if it reduces maintainer routing or recovery work without weakening governance or losing task state. Otherwise return to the manual queue and update this ADR with the result.

## Consequences

- Issue readiness, dependencies, and checkpoint fields become operational inputs, so the implementation must validate malformed or incomplete records as blocked rather than guessing.
- Durable recovery state must be updated at meaningful stage boundaries and linked to immutable branch/head and evidence identities.
- A worker restart is a continuation of the same issue owner, not permission to create another writer.
- Human gates remain explicit terminal states; automation may prepare the review packet but cannot accept it.
- This ADR does not authorize implementation. A separate bounded implementation PR and a measured disposable pilot are required.

## Evidence

- [Issue #19](https://github.com/lzy18001500226/RobotSim/issues/19) records repeated manual prompt-copy/worker-routing overhead and the requirement to resume the same primary worker after CI/review feedback.
- [Issue #49](https://github.com/lzy18001500226/RobotSim/issues/49) defines the current manual readiness/dependency flow, bounded retries, same-worker continuation, and durable human review packet.
- The [maintainer observation recorded on Issue #49](https://github.com/lzy18001500226/RobotSim/issues/49#issuecomment-6033815256) records the additional Local02/Local03 routing, conditional queue construction, bounded-FAIL continuation, network-interruption recovery, and distributed-state observations. These remain qualitative; the pilot must establish counts and timing.
- [Issue #32](https://github.com/lzy18001500226/RobotSim/issues/32#issuecomment-6033813033) records the current governance settings and live Auto-PR success/failure E2E needed by the pilot's existing PR/CI path.
