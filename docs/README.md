# Documentation Guide

Use the [repository README](../README.md) for the project overview and [`AGENTS.md`](../AGENTS.md) for durable repository rules. Start with this map, then read only the documents relevant to the current task.

| Path | Owns | Read it when |
|---|---|---|
| [`00_PROJECT_PLAN.md`](00_PROJECT_PLAN.md) | Project goals, development phases, design principles, and non-goals | You need high-level scope or milestone context. It is not a task queue. |
| [`adr/`](adr/) | Accepted architecture and environment decisions | A task touches an established design boundary or proposes changing one. |
| [`engineering/README.md`](engineering/README.md) | Repository layout, artifact discipline, runtime and validation contract | Adding runtime code, selecting a language, or deciding what evidence a change needs. |
| [`setup/`](setup/) | Host, WSL, Docker, and development-container setup | Setting up or diagnosing the development environment. |
| [`setup/codex.md`](setup/codex.md) | Local/Cloud Codex boundaries, hooks, notifications, and closeout | Choosing where Codex work runs or finishing a task. |
| [`workflows/`](workflows/) | Repeatable procedures for baseline bring-up, navigation, and robot porting | Implementing or validating one of those workflows. |
| [`workflows/goal_driven_development.md`](workflows/goal_driven_development.md) | Codex prompt-versus-Goal conventions, continuation, evidence, and review gates | Starting multi-step or long-running Codex work. |
| [`skills/`](skills/) | Focused technical notes for robots, MuJoCo, ROS 2, and LiDAR/SLAM | Working in the corresponding technology area. |
| [`checklists/`](checklists/) | Environment and reproducibility checks | Verifying setup or preparing reproducible results. |

## Reading and maintenance

- For a normal task, read `AGENTS.md`, the repository README, and the relevant source. Use the table above to select any additional documentation; do not read every directory by default.
- For architecture-sensitive changes, read the affected ADR before editing. For a repeatable task, follow its workflow and the applicable checklist or skill note.
- Keep each document in the directory that owns its purpose. Update existing guidance when its responsibility changes; link to it instead of copying it elsewhere.
