# Documentation Guide

This map identifies the documents that own RobotSim guidance.

| Path | Owns | Read it when |
|---|---|---|
| [`00_PROJECT_PLAN.md`](00_PROJECT_PLAN.md) | Project goals, development phases, design principles, and non-goals | You need high-level scope or milestone context. It is not a task queue. |
| [`adr/`](adr/) | Status-bearing architecture and environment decision records; proposals are not accepted by default | A task touches an established design boundary or proposes changing one. |
| [`engineering/README.md`](engineering/README.md) | Repository layout, artifact discipline, runtime and validation contract | Adding runtime code, selecting a language, or deciding what evidence a change needs. |
| [`research/`](research/) | Durable reports for bounded, isolated simulation and model investigations | Reviewing experiment outcomes and their evidence paths. |
| [`engineering/simulation_state_contract.md`](engineering/simulation_state_contract.md) | Topology-independent simulation state, command, lifecycle, and telemetry semantics | Defining or implementing a simulation state boundary without selecting its transport. |
| [`setup/`](setup/) | Host, WSL, Docker, and development-container setup | Setting up or diagnosing the development environment. |
| [`setup/codex.md`](setup/codex.md) | Local/Cloud Codex boundaries, hooks, notifications, and closeout | Choosing where Codex work runs or finishing a task. |
| [`workflows/`](workflows/) | Repeatable procedures for baseline bring-up, navigation, and robot porting | Implementing or validating one of those workflows. |
| [`workflows/05_G1_ROS2_ADAPTER.md`](workflows/05_G1_ROS2_ADAPTER.md) | Build and validate the topology-independent ROS 2 workspace and G1 SDK2 adapter | Working on the initial G1 ROS boundary. |
| [`workflows/goal_driven_development.md`](workflows/goal_driven_development.md) | Codex prompt-versus-Goal conventions, continuation, evidence, and review gates | Starting multi-step or long-running Codex work. |
| [`workflows/agent_behavior_eval.md`](workflows/agent_behavior_eval.md) | Executable local Agent behavior fixtures, graders, and manual evidence limits | Evaluating observable Agent infrastructure behavior. |
| [`skills/`](skills/) | RobotSim technical notes; these are not native Codex Skills. Native project Skills use `.agents/skills/<name>/SKILL.md`. | Working in the corresponding technology area. For validation routing, use the [RobotSim validation-planning Skill](../.agents/skills/robotsim-validation-planning/SKILL.md). |
| [`checklists/`](checklists/) | Environment, reproducibility, and [Agent Infra evaluation cases](checklists/agent_infra_eval.md) | Verifying setup, preparing reproducible results, or evaluating agent-infrastructure changes. |
