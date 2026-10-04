# Goal-driven development

Use the native Codex Goal capability when a task needs persistent continuation across work periods. A RobotSim Goal is treated as a **thread-scoped completion contract**: it keeps one active task and its criteria attached to that Codex thread. It is not a repository-wide queue, issue tracker, background daemon, or substitute for a clear request. Keep one primary task owner, as required by [AGENTS.md](../../AGENTS.md).

This is RobotSim's operating convention layered on the [official Codex documentation](https://developers.openai.com/codex/) and [AGENTS.md guidance](https://developers.openai.com/codex/guides/agents-md). The open Codex configuration source documents `max_goal_token_budget` as the maximum token budget for a Goal and the default for new Goals ([source at the reviewed revision](https://github.com/openai/codex/blob/2739e828581c3deee59a270cd740675dc278ed14/codex-rs/config/src/config_toml.rs)); a budget bounds work, it does not establish success. That source defines the budget setting, not a stable UI lifecycle contract. Follow the documented [sandbox and approval boundaries](https://developers.openai.com/codex/security); non-interactive mode is for bounded automation ([official guidance](https://developers.openai.com/codex/noninteractive)) and does not grant new approval. Client controls and labels can change, so use the current Codex UI's Goal controls rather than relying on an undocumented command.

## Prompt or Goal

Use a normal prompt for a one-off edit, explanation, short review, or small deterministic task with an obvious stopping point. Use a Goal for architecture exploration, uncertain debugging, performance/benchmark work, multi-step integration, dependency migration, evidence-backed research, or implementation that must continue across sessions and has explicit completion criteria.

Create a Goal only when the desired end state and evidence can be stated. A vague instruction such as “improve the simulator” is not a useful Goal; first bound its scope or turn it into a research Goal that must produce a decision artifact.

## Reusable Goal template

```text
Desired end state
- Describe the observable repository or runtime state that means the task is done.

Verification evidence
- Name required checks, target environments, and expected artifacts/results.
- Identify which checks are PASS, MANUAL PASS, or allowed to be DEFERRED.

Constraints and invariants
- Name architecture boundaries, files/areas in scope, safety limits, and forbidden actions.

Allowed tools and environment
- State the repository/branch and available local, Cloud, simulator, or hardware targets.
- Never put credentials in the Goal; use approved platform or secret-manager injection.

Next-action policy
- After each result, choose the next safe action that advances the end state.
- Investigate recoverable compile/test/research failures and try a supported alternative.
- Keep one primary task owner; do not build an orchestration framework to manage the Goal.

Blocker behavior
- Before declaring BLOCKED_HUMAN, finish independent in-scope work and preserve evidence.
- State the exact external decision, permission, credential, hardware access, or manual check
  that unlocks the remaining work. Do not infer approval from silence.

Budget and stop behavior
- Treat the Goal budget as a work limit. If it is exhausted first, report the partial state,
  next action, and remaining evidence; do not report completion.
- Stop when all completion evidence passes, a genuine human gate blocks all useful work,
  the user cancels/changes scope, or the budget ends. Summarize the evidence and status.
```

## Continuation, blockers, and evidence

Continue from the Goal and the repository's durable state; inspect the current branch, tree, diffs, and existing artifacts before acting. Save useful handoff state in the right maintained artifact (code, tests, docs, issue/PR) rather than relying on conversation memory or a new progress diary. When a hypothesis fails, retain its useful evidence, choose the next safe hypothesis, and continue if scope and budget allow.

Completion is evidence-based. Apply the minimum levels and status definitions in [the validation contract](../engineering/validation.md), and report each result with its command and evidence. Finish other in-scope work before stopping on a blocker; name the exact unlock action for any required deferred check.

The Goal's thread scope matters: it helps continue one task, but does not automatically discover or switch to other independent repository tasks. Use manually curated GitHub Issues if a durable backlog becomes useful. Do not build a supervisor or queue merely to imitate an issue tracker.

## Pause, resume, clear, and review gates

Where the current Codex client exposes pause, resume, or clear/end controls, RobotSim applies these meanings. Verify the exact control and label in the current client; product mechanics may change:

- **Pause:** suspend further autonomous continuation. The current thread, repository edits, and evidence remain available. Do not describe the task as complete just because it is paused.
- **Resume:** continue the same Goal in its existing thread from the current repository state and the last evidence. Re-check the tree before resuming work.
- **Clear/end:** retire the active Goal contract in that thread. This does not roll back files, commits, or PRs and is not a success signal. Record any unfinished work before clearing if it needs a handoff.

Continue through ordinary test failures and safe alternatives. Stop for architecture acceptance, destructive operations, new credentials/authentication, irreversible dependency migrations, unsafe hardware actuation, or unresolved visual/manual assertions that affect correctness. Routine low-risk merge authorization and required checks/review follow [AGENTS.md](../../AGENTS.md); Goal continuation does not waive these meaningful gates or expand task scope.

For repository PR closeout, review the scoped diff and its evidence, then follow [the Codex closeout workflow](../setup/codex.md#closeout), including its explicit `ready_for_review`, `completed`, or `blocked` notifier call. A Goal status and a PR notification are separate records: use the notification that matches the actual task outcome and report deferred checks honestly.

## Long-running-agent evidence and future decision

Anthropic's [effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) and [building effective agents](https://www.anthropic.com/engineering/building-effective-agents) are comparative engineering material. RobotSim adopts the useful themes of durable artifacts, incremental progress, evaluator feedback, and human judgment at meaningful checkpoints. These examples do not justify copying a multi-agent or multi-agent supervisor design; RobotSim's default remains one primary owner.

See [ADR-0006](../adr/0006_goal_mode_first.md) for the Proposed choice to start with native Goals, add a manually maintained issue backlog only when useful, and reconsider a local supervisor only after observed task-switching failures provide evidence.
