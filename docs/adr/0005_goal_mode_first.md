# ADR-0005: Start with native Goal mode for long-running work

Status: Proposed

## Context

RobotSim is at early platform bring-up and does not yet have a repository-wide autonomous task queue. Long-running work needs a durable completion contract and evidence across sessions, but adding a runner, database, watchdog, or multi-agent framework now would create an independent system before there is evidence that the repository needs one.

Codex provides native Goal behavior scoped to a task thread and supports a configurable maximum Goal token budget. That is a natural first tool for one task's continuation, but it does not provide a repository-wide queue or automatic switching among independent work. Anthropic's long-running-agent material supports durable task artifacts, incremental progress, evaluator feedback, and meaningful human checkpoints; it is comparative evidence, not a requirement to adopt its multi-agent harness.

## Options

1. Native Codex Goal mode only.
2. Goal mode plus a manually curated GitHub Issue backlog.
3. A custom local supervisor or runner that selects and resumes tasks.

## Proposed decision

Start with **native Goal mode** for one bounded, multi-step task and one primary task owner. Use GitHub Issues manually when RobotSim needs a durable prioritized backlog shared across people or sessions. Do not implement a custom supervisor now.

Reconsider a supervisor only after real usage demonstrates repeated, concrete failures in switching among independent tasks after a Goal is blocked or complete, and after a simpler manually curated backlog has proved insufficient. Any new proposal must present those observations, safety boundaries, and measured benefits.

## Consequences

- Goals carry desired outcome, verification evidence, constraints, allowed environment, next-action policy, blocker behavior, and budget/stop behavior.
- Goal budget exhaustion is a partial stop, not completion. Evidence and repository state remain the completion authority.
- Pausing, resuming, or clearing a Goal affects the active task contract; it does not undo repository changes or create a task queue.
- Independent tasks are not silently started just because the current Goal blocks.
- This decision is Proposed because usage may show that even manual issues are unnecessary or that task switching later warrants a separate design.

## Evidence and references

- [Codex documentation](https://developers.openai.com/codex/) and [AGENTS.md guidance](https://developers.openai.com/codex/guides/agents-md).
- [Codex Goal budget configuration source at the reviewed revision](https://github.com/openai/codex/blob/2739e828581c3deee59a270cd740675dc278ed14/codex-rs/config/src/config_toml.rs).
- Comparative only: Anthropic's [long-running agent harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) and [effective agents](https://www.anthropic.com/engineering/building-effective-agents).
