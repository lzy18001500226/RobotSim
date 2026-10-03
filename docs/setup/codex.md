# Codex execution and closeout

RobotSim uses local Codex for integration work that needs its canonical robotics environment. Cloud Codex is for repository work that can be validated headlessly. Keep claims within the environment actually tested.

RobotSim Dockerfiles and Compose files define the reproducible headless/runtime baseline. Unity, WSLg, GPU rendering, and their integration remain host-side/local capabilities; the container baseline does not reproduce them.

For multi-step work that should continue across sessions, use the repository's [Goal-driven development workflow](../workflows/goal_driven_development.md). It defines the evidence, blocker, budget, and review boundaries for native Codex Goals; it does not create a shared task queue or expand the user's authorization.

## Where work runs

| Environment | Appropriate work | Validation boundary |
| --- | --- | --- |
| Local Codex | Canonical integration on WSL2 Ubuntu 22.04 with ROS 2 Humble; Unity, WSLg, GPU rendering, DDS, vendor runtimes, and hardware integration | Use local results for GUI, GPU, full ROS, DDS, and hardware behavior. |
| Codex Cloud | Headless C/C++ and Python changes, MuJoCo headless work, static and unit tests, documentation, CI fixes, code review, and repository refactors | Cloud results do not prove Unity, GPU, full ROS 2 Humble, DDS, vendor runtime, or hardware behavior. |

New Codex Web Cloud task creation is currently known to fail in some cases with `Unable to determine project root for task`. Continue from an existing repository-bound Cloud workspace when available. Do not add project architecture or task infrastructure around this temporary product limitation.

See [host and WSL setup](01_HOST_WSL_DOCKER.md), [development container setup](02_DEV_CONTAINER.md), and the [Codex Cloud environment guide](https://developers.openai.com/codex/cloud/environments/).

## Project-local Codex hooks

`.codex/hooks.json` contains a small, optional `PreToolUse` guard for plainly destructive Git commands and obvious attempts to stage or write credential files. It is a last-line guard, not a substitute for reviewing a command or protecting secrets. The repository remains usable when hooks are disabled.

Codex loads project hooks only through its project configuration and trust controls. Review the hook and accept Codex's project-trust prompt through the normal UI before enabling it. Never bypass hook trust, approval policy, or the user's review decision. Do not copy project settings into global `~/.codex` configuration.

There is no automatic terminal notification hook. A turn-ending hook cannot reliably tell whether the task is complete, ready for review, or blocked. Invoke the notifier explicitly at the matching closeout point below. Notification errors are reported as best-effort warnings and do not change the task result.

For current hook behavior, see the [Codex hook configuration types](https://github.com/openai/codex/blob/main/codex-rs/config/src/hook_config.rs), [hook discovery and trust handling](https://github.com/openai/codex/blob/main/codex-rs/hooks/src/engine/discovery.rs), and the [official AGENTS.md guidance](https://developers.openai.com/codex/guides/agents-md).

## Task notifications

`scripts/agent/notify_task.py` sends opt-in `ready_for_review`, `completed`, or `blocked` email through SMTP using only `ROBOTSIM_*` environment variables. It uses STARTTLS with certificate verification, stores duplicate-state markers named by a hash of the task/status pair outside the repository, and has a dry-run mode. It has no dependency on MoSim or ChatGPT AgentMail.

Required values are `ROBOTSIM_SMTP_HOST`, `ROBOTSIM_SMTP_FROM`, and `ROBOTSIM_NOTIFY_TO`. `ROBOTSIM_SMTP_PORT` defaults to 587 and `ROBOTSIM_SMTP_TIMEOUT` defaults to 10 seconds. If SMTP authentication is required, set both `ROBOTSIM_SMTP_USERNAME` and `ROBOTSIM_SMTP_PASSWORD`; supplying only one skips delivery. `ROBOTSIM_NOTIFY_STATE_DIR` optionally selects the duplicate-state directory; the default is under the user's cache directory, outside the checkout.

Locally, provide these values through the user's shell environment or a local secret manager. Do not put addresses or credentials in Git, command-line arguments, or chat. In Codex Cloud, set non-secret SMTP host, port, sender, recipient, timeout, and state-directory values as environment variables; when notifications are enabled, provide the SMTP username and password through Codex Cloud environment secrets and allow the SMTP relay through the environment's network settings. Confirm the selected Cloud secret delivery mode can provide credentials to this SMTP path: proxy-only secrets are substituted for supported HTTPS requests and are not raw values for a direct SMTP connection. If direct delivery is unavailable, keep authenticated SMTP notifications disabled or use an approved HTTPS mail relay; never copy a secret into ordinary variables, the repository, or this guide. Notification is disabled by missing configuration and remains non-blocking.

Example invocation (use a stable task ID such as the PR number and a non-sensitive summary):

```bash
python3 scripts/agent/notify_task.py ready_for_review --task-id "$TASK_ID" --summary "$TASK_SUMMARY" --dry-run
python3 scripts/agent/notify_task.py ready_for_review --task-id "$TASK_ID" --summary "$TASK_SUMMARY"
```

Use `completed` or `blocked` for the corresponding terminal outcome. `--dry-run` formats the exact subject and body without requiring SMTP settings or sending mail.

## Closeout

Implementation task on an approved branch (`issue/**` or `codex/**`):

```text
validate -> self-review -> commit -> push -> Actions creates or detects PR
  -> task-branch CI -> human review and maintainer gates -> merge when authorized
```

The [`auto-task-pr.yml`](../../.github/workflows/auto-task-pr.yml) workflow runs only for pushes to the approved prefixes in the canonical RobotSim repository. It detects an existing open PR by head branch and otherwise creates one targeting `main`, using the standard PR template from `main`. Issue branches such as `issue/36-auto-pr-handoff` are linked to the matching issue, and the new PR includes the branch and pushed head SHA. Later pushes add commits to the same PR without rewriting its description. The workflow does not approve, merge, or bypass branch rulesets.

The workflow has `contents: read` and `pull-requests: write` only. It does not check out or execute task-branch code, and validates branch names without passing them through a shell. Fork repositories are excluded. The existing Agent Infrastructure and G1 smoke CI workflows also run on approved task-branch pushes, so CI evidence for the pushed head does not depend on follow-up PR events. GitHub suppresses most workflow runs caused by `GITHUB_TOKEN`; on some configurations, `pull_request` runs from a workflow-created PR require approval. See [GitHub's `GITHUB_TOKEN` event behavior](https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs).

If the workflow cannot create PRs, a repository administrator may need to enable **Settings → Actions → General → Workflow permissions → Allow GitHub Actions to create and approve pull requests**. The workflow still requests only `pull-requests: write`; it does not approve PRs or require a personal token. See [GitHub's repository Actions settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository) and [`GITHUB_TOKEN` permissions](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#permissions).

Research-only work that does not push an approved task branch is outside this automation. It does not create a branch, PR, or task queue for read-only research. CI evidence, implementation claims, architecture decisions, security review, and hardware validation still require the normal human review gates.

Routine authorized task:

```text
validate -> self-review -> commit -> push -> auto-created/detected PR -> required checks/review
  -> merge only when authorized -> update main -> completed notification
```

Review-gated task:

```text
validate -> self-review -> commit -> push -> auto-created/detected PR
  -> ready_for_review notification -> stop for review
```

Blocked task:

```text
preserve the relevant evidence -> blocked notification -> stop
```

Stop at a reviewable PR for architecture decisions, destructive or high-risk changes, dependency/security changes, unresolved GUI/hardware validation, and tasks the user explicitly gates for review. Never bypass branch protection or force merge. Current user instructions always take precedence.
