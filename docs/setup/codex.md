# Codex execution and closeout

RobotSim uses local Codex for integration work that needs its canonical robotics environment. Cloud Codex is for repository work that can be validated headlessly. Keep claims within the environment actually tested.

RobotSim Dockerfiles and Compose files define the reproducible headless/runtime baseline. Unity, WSLg, GPU rendering, and their integration remain host-side/local capabilities; the container baseline does not reproduce them.

For multi-step work that should continue across sessions, use the repository's [Goal-driven development workflow](../workflows/goal_driven_development.md). It defines the evidence, blocker, budget, and review boundaries for native Codex Goals; it does not create a shared task queue or expand the user's authorization.

## Where work runs

| Environment | Appropriate work | Validation boundary |
| --- | --- | --- |
| Local Codex | Canonical integration on WSL2 Ubuntu 22.04 with ROS 2 Humble; Unity, WSLg, GPU rendering, DDS, vendor runtimes, and hardware integration | Use local results for GUI, GPU, full ROS, DDS, and hardware behavior. |
| Codex Cloud | Headless C/C++ and Python changes, MuJoCo headless work, static and unit tests, documentation, CI fixes, code review, and repository refactors | Cloud results do not prove Unity, GPU, full ROS 2 Humble, DDS, vendor runtime, or hardware behavior. |

See [host and WSL setup](01_HOST_WSL_DOCKER.md), [development container setup](02_DEV_CONTAINER.md), and the [Codex Cloud environment guide](https://developers.openai.com/codex/cloud/environments/).

## Project-local Codex hooks

`.codex/hooks.json` contains an optional `PreToolUse` guard for plainly destructive Git commands and obvious attempts to stage or write credential files, plus a local `Stop` handler for one best-effort notification when a Codex turn ends. The guard is a last-line check, not a substitute for reviewing a command or protecting secrets. The repository remains usable when hooks are disabled.

Codex discovers project hooks from `.codex/hooks.json`, subject to project/folder trust and hook approval. Review the hook and trust the real repository checkout through the normal UI before enabling it. Then run `/hooks` in that trusted checkout and confirm both `PreToolUse` and `Stop` are listed. Check the repository root, branch, and working-tree status before acting, especially after opening another checkout or worktree. Never bypass hook trust or approval policy, and do not copy project settings into global `~/.codex` configuration.

The local `Stop` hook receives Codex's stop payload, including `session_id`, `turn_id`, `last_assistant_message`, and `stop_hook_active`. It sends one `completed` notification per session/turn only when the local-only `ROBOTSIM_LOCAL_STOP_HOOK=1` opt-in is set, and ignores a recursive Stop event when `stop_hook_active` is true. Normal hook outcomes leave stdout empty, as Codex expects; failures may be diagnosed on stderr. A manual `stop --dry-run` still prints the formatted message without sending mail. The hook does not run for individual tool calls, tests, commits, or progress updates. Notification errors are best-effort and do not change the task result.

Project command hooks must not be assumed to run in Codex Cloud. For each Cloud instruction, explicitly call `scripts/agent/notify_task.py` exactly once at closeout, after validation and immediately before the final answer, with the actual terminal outcome (`ready_for_review`, `completed`, or `blocked`). Leave `ROBOTSIM_LOCAL_STOP_HOOK` unset in Cloud: even if project hooks become active there, the Stop handler will skip, so the explicit Cloud closeout remains the only notification path. Use a stable task/turn ID so retries are deduplicated. Do not make a second manual closeout call after the Cloud call.

For current hook behavior, see the [Codex hook configuration types](https://github.com/openai/codex/blob/main/codex-rs/config/src/hook_config.rs), [hook discovery and trust handling](https://github.com/openai/codex/blob/main/codex-rs/hooks/src/engine/discovery.rs), and the [official AGENTS.md guidance](https://developers.openai.com/codex/guides/agents-md).

## AgentMail notifications

`scripts/agent/notify_task.py` uses AgentMail's HTTPS REST send endpoint with the inbox API key as `Authorization: Bearer …`; requests include an `Idempotency-Key`. It uses Python's standard library and adds no runtime package dependency. The endpoint, request fields, authentication, and idempotency behavior were checked against AgentMail's official Python SDK v2.0.8 source at commit [`9df61ab`](https://github.com/agentmail-to/agentmail-python/tree/9df61ab2fa07ff3aec40308d3e8f78a8f52da9c1): [send endpoint and request](https://github.com/agentmail-to/agentmail-python/blob/9df61ab2fa07ff3aec40308d3e8f78a8f52da9c1/src/agentmail/inboxes/messages/raw_client.py), [Bearer authentication](https://github.com/agentmail-to/agentmail-python/blob/9df61ab2fa07ff3aec40308d3e8f78a8f52da9c1/src/agentmail/core/client_wrapper.py), and [SDK usage](https://github.com/agentmail-to/agentmail-python/blob/9df61ab2fa07ff3aec40308d3e8f78a8f52da9c1/README.md). The sender is the configured AgentMail inbox; the recipient is the maintainer's normal QQ mailbox.

When notifications are enabled, configure:

| Variable | Value | Handling |
| --- | --- | --- |
| `AGENTMAIL_API_KEY` | AgentMail API key | Secret. Never put it in Git, a command line, a log, or chat. |
| `AGENTMAIL_INBOX_ID` | `lzy18001500226@agentmail.to` | Non-secret inbox identifier. |
| `ROBOTSIM_NOTIFY_TO` | Maintainer's normal QQ mailbox | Personal configuration; do not hard-code it in the repository. |
| `ROBOTSIM_NOTIFY_STATE_DIR` | Optional local state path | Optional; defaults under the user's cache directory, outside the checkout. |
| `ROBOTSIM_LOCAL_STOP_HOOK` | Set to `1` in the local Codex environment only | Local Stop-hook opt-in. Leave unset in Codex Cloud. |

### Local Codex setup

1. Create an AgentMail API key in the AgentMail account that owns the inbox.
2. Store the key in the operating system's secret manager or the user's shell secret manager, and expose it to Codex as `AGENTMAIL_API_KEY`. Do not place it in a shell command, shell startup file in plaintext, repository file, or chat.
3. Set `AGENTMAIL_INBOX_ID=lzy18001500226@agentmail.to` and set `ROBOTSIM_NOTIFY_TO` to the maintainer's normal QQ mailbox through the user's local environment/secret manager.
4. Set `ROBOTSIM_LOCAL_STOP_HOOK=1` in the local Codex environment. Restart local Codex so it receives the configured environment. Trust the actual RobotSim checkout, run `/hooks`, and verify that `Stop` and `PreToolUse` are active.

### Codex Cloud setup

1. In the Codex Cloud Environment settings, add `AGENTMAIL_API_KEY` as an environment secret. Do not paste it into a task prompt or repository file.
2. Configure `AGENTMAIL_INBOX_ID=lzy18001500226@agentmail.to` and the personal `ROBOTSIM_NOTIFY_TO` value in the Cloud environment's protected variable/secret settings. Do not store the QQ address in this repository, and do not set `ROBOTSIM_LOCAL_STOP_HOOK` in Cloud.
3. Allow HTTPS access to `api.agentmail.to` in the Cloud environment network settings. A Cloud task must explicitly call the notifier once at closeout; project command hooks are not assumed to run there.
4. Notification is opt-in. If configuration is missing or AgentMail is unavailable, the notifier reports a best-effort skip/failure and exits successfully so the task can finish.

For Cloud closeout, use a stable identifier for the instruction/turn and the matching actual outcome. For example:

```bash
python3 scripts/agent/notify_task.py ready_for_review \
  --task-id "issue-28-pr-123-review" \
  --task "Issue #28 AgentMail notifications" \
  --summary "Validation passed and the review PR is open" \
  --worker "Codex Cloud" \
  --issue 28
```

Change the event to `completed` or `blocked` when that is the actual terminal outcome. Do not send per-tool, per-test, per-commit, or progress notifications. Messages contain a short task label, worker, branch/issue when available, and final summary. Local state markers and AgentMail's idempotency key prevent duplicate sends for the same event.

### Human-gated live tests

The automated tests replace the AgentMail HTTP transport with mocks; CI never sends live email. Do not use Cloud setup validation or `--dry-run` as proof of delivery. After the maintainer configures the real values, the exact live checks still required are:

1. **Local:** restart Codex, trust the real checkout, confirm `Stop` appears in `/hooks`, complete one harmless test instruction, and verify that exactly one short notification reaches the normal QQ mailbox.
2. **Cloud:** start one harmless Cloud task with the secrets configured, let the Cloud task make its single explicit closeout call with a fresh task ID, and verify that exactly one notification reaches the same QQ mailbox. Do not rely on project hooks for this check.

These are human-gated E2E checks. No real email is sent by this PR's Cloud validation or CI.

## Closeout

Routine authorized task:

```text
validate -> self-review -> commit -> push -> PR -> required checks/review
  -> merge only when authorized -> update main -> completed notification
```

Review-gated task:

```text
validate -> self-review -> commit -> push -> PR
  -> ready_for_review notification -> stop for review
```

Blocked task:

```text
preserve the relevant evidence -> blocked notification -> stop
```

Stop at a reviewable PR for architecture decisions, destructive or high-risk changes, dependency/security changes, unresolved GUI/hardware validation, and tasks the user explicitly gates for review. Never bypass branch protection or force merge. Current user instructions always take precedence.
