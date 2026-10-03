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

Before acting in a new, resumed, or parallel-worktree task, verify the checkout:

```bash
git rev-parse --show-toplevel
git branch --show-current
git status --short --branch
```

Confirm the root and branch are the intended ones, and preserve any existing changes.

## Project-local Codex hooks

`.codex/hooks.json` contains an optional `PreToolUse` guard for plainly destructive Git commands and obvious attempts to stage or write credential files, plus a local `Stop` handler for one best-effort notification when a Codex turn ends. The guard is a last-line check, not a substitute for reviewing a command or protecting secrets. The repository remains usable when hooks are disabled.

Codex discovers project hooks from `.codex/hooks.json`; discovery is distinct from project/folder trust and hook approval. Review the hook and trust the real repository checkout through the normal UI before enabling it. Then run `/hooks` in that trusted checkout and confirm both `PreToolUse` and `Stop` are listed and enabled. If a hook is absent or disabled, do not assume it ran. Recheck `/hooks` after changing hook configuration; a changed handler may need renewed trust. Check the repository root, branch, and working-tree status before acting, especially after opening another checkout or worktree. Never bypass hook trust or approval policy, and do not copy project settings into global `~/.codex` configuration.

The local `Stop` hook receives Codex's stop payload, including `session_id`, `turn_id`, `last_assistant_message`, and `stop_hook_active`. For ordinary turns it sends one `completed` notification per session/turn. If the final message contains a valid marked research-completion envelope, it emits that structured event instead of the generic completion. Both paths require the local-only `ROBOTSIM_LOCAL_STOP_HOOK=1` opt-in; recursive Stop events are ignored. Normal hook outcomes leave stdout empty, as Codex expects; failures may be diagnosed on stderr. A manual `stop --dry-run` still prints formatted output without sending mail or writing an Issue comment. The hook does not run for individual tool calls, tests, commits, or progress updates. Notification errors are best-effort and do not change the task result.

Project command hooks must not be assumed to run in Codex Cloud. For implementation tasks, explicitly call `scripts/agent/notify_task.py` exactly once at closeout with the actual terminal outcome (`ready_for_review`, `completed`, or `blocked`). For read-only research tasks, send one `research-completion` event instead, as described below. Leave `ROBOTSIM_LOCAL_STOP_HOOK` unset in Cloud: even if project hooks become active there, the Stop handler will skip, so the explicit Cloud closeout remains the only notification path. Use a stable task/turn ID so retries are deduplicated. Do not make a second generic closeout call after a research event.

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
| `GH_TOKEN` or `GITHUB_TOKEN` | GitHub API token | Secret. Needs Issues read/write access to `lzy18001500226/RobotSim` for research-event comments; never put it in Git, a command line, a log, or chat. |
| `ROBOTSIM_LOCAL_STOP_HOOK` | Set to `1` in the local Codex environment only | Local Stop-hook opt-in. Leave unset in Codex Cloud. |

### Local Codex setup

1. Create an AgentMail API key in the AgentMail account that owns the inbox.
2. Store the AgentMail key and, when research-event persistence is needed, a GitHub token with Issue comment read/write access in the operating system's secret manager. Expose them as `AGENTMAIL_API_KEY` and `GH_TOKEN` (or `GITHUB_TOKEN`) to Codex. Do not put either value in a shell command, plaintext shell startup file, repository file, or chat.
3. Set `AGENTMAIL_INBOX_ID=lzy18001500226@agentmail.to` and set `ROBOTSIM_NOTIFY_TO` to the maintainer's normal QQ mailbox through the user's local environment/secret manager.
4. Set `ROBOTSIM_LOCAL_STOP_HOOK=1` in the local Codex environment. Restart local Codex so it receives the configured environment. Trust the actual RobotSim checkout, run `/hooks`, and verify that `Stop` and `PreToolUse` are active.

### Codex Cloud setup

1. In the Codex Cloud Environment settings, add `AGENTMAIL_API_KEY` and, when research-event persistence is needed, `GH_TOKEN` (or `GITHUB_TOKEN`) as environment secrets. The GitHub token needs Issue comment read/write access to this repository. Do not paste secrets into a task prompt or repository file.
2. Configure `AGENTMAIL_INBOX_ID=lzy18001500226@agentmail.to` and the personal `ROBOTSIM_NOTIFY_TO` value in the Cloud environment's protected variable/secret settings. Do not store the QQ address in this repository, and do not set `ROBOTSIM_LOCAL_STOP_HOOK` in Cloud.
3. Allow HTTPS access to `api.agentmail.to` for notifications and `api.github.com` for research-event comment persistence in the Cloud environment network settings. A Cloud task must explicitly call the notifier once at closeout; project command hooks are not assumed to run there.
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

## Read-only research completion

For read-only research, the originating GitHub Issue comment is the canonical durable project record. AgentMail is an attention channel and secondary retrieval surface; ChatGPT-side AgentMail access is optional and is not needed for correctness. Implementation work continues to use pull requests. A research task emits one structured completion event, not a separate generic `completed` notification.

The v1 event uses `schema_version: "robotsim.research-completion.v1"` and the fields `event_id`, `repository`, `task_id`, `worker`, `final_status`, `summary`, `blockers`, `recommended_next_action`, `evidence`, and `timestamp`. `final_status` is one of `completed`, `blocked`, `deferred`, `cancelled`, or `failed`. Use the canonical repository name `lzy18001500226/RobotSim`; `task_id` must be stable and formatted `issue-<number>-<stable-slug>` so the notifier can persist to the originating Issue. Set `event_id` to `auto`; the notifier derives a stable `sha256:` ID from schema version, repository, task ID, and worker. Keep the exact event and timestamp when retrying. Reusing an event ID with changed content is rejected rather than overwritten.

Use concise, safe summaries and blockers. Evidence entries must be durable HTTPS references; local paths, private/local hosts, credential-bearing URLs, unknown fields, and transcript fields are rejected or omitted. The notifier redacts recognized credential-shaped text and local filesystem paths from text fields. It sends only the structured event, never the surrounding final answer or transcript.

Local Codex recognizes one JSON envelope in the final message, wrapped as an HTML comment so it does not clutter the user-facing report:

```html
<!-- robotsim-research-completion:v1
{
  "schema_version": "robotsim.research-completion.v1",
  "event_id": "auto",
  "repository": "lzy18001500226/RobotSim",
  "task_id": "issue-38-agentmail-research",
  "worker": "Codex",
  "final_status": "completed",
  "summary": "Recorded the research findings and reuse map.",
  "blockers": [],
  "recommended_next_action": "Review the findings and open a focused follow-up issue.",
  "evidence": ["https://github.com/lzy18001500226/RobotSim/issues/38"],
  "timestamp": "2026-10-04T12:00:00Z"
}
-->
```

The Stop hook validates the envelope, writes a hidden event marker and structured payload to the Issue comment, then sends the same event through the existing AgentMail configuration and idempotency path. If GitHub comment capability is missing, it reports that persistence did not happen and skips AgentMail; add `GH_TOKEN`/`GITHUB_TOKEN` with Issue comment access before retrying. Once the Issue comment exists, AgentMail failure does not remove or replace it; retry the same event to recover delivery.

Codex Cloud must explicitly call the same notifier once, with the event JSON on stdin; project hooks are not assumed to run there:

```bash
python3 scripts/agent/notify_task.py research-completion <<'JSON'
{"schema_version":"robotsim.research-completion.v1","event_id":"auto","repository":"lzy18001500226/RobotSim","task_id":"issue-38-agentmail-research","worker":"Codex Cloud","final_status":"completed","summary":"Recorded the research findings and reuse map.","blockers":[],"recommended_next_action":"Review the findings and open a focused follow-up issue.","evidence":["https://github.com/lzy18001500226/RobotSim/issues/38"],"timestamp":"2026-10-04T12:00:00Z"}
JSON
```

Do not make a second generic closeout call for that research task. The GitHub Issue is sufficient for durable retrieval even when AgentMail is not configured or unavailable.

### Human-gated live tests

The automated tests replace the AgentMail HTTP transport with mocks; CI never sends live email. Do not use Cloud setup validation or `--dry-run` as proof of delivery. After the maintainer configures the real values, the exact live checks still required are:

1. **Local:** restart Codex, trust the real checkout, confirm `Stop` appears in `/hooks`, complete one harmless test instruction, and verify that exactly one short notification reaches the normal QQ mailbox.
2. **Cloud:** start one harmless Cloud task with the secrets configured, let the Cloud task make its single explicit closeout call with a fresh task ID, and verify that exactly one notification reaches the same QQ mailbox. Do not rely on project hooks for this check.

These are human-gated E2E checks. No real email is sent by this PR's Cloud validation or CI.

## Closeout

Implementation task on an approved branch (`issue/**` or `codex/**`):

```text
validate -> self-review -> commit -> push -> Actions creates or detects PR
  -> task-branch CI -> human review and maintainer gates -> merge when authorized
```

The task-push CI remains unprivileged. After an `Agent infrastructure` run completes, [`auto-task-pr.yml`](../../.github/workflows/auto-task-pr.yml) runs from the default branch and checks out only `main`, where its helper code also lives. GitHub requires a `workflow_run` workflow to exist on the default branch, as described in its [event documentation](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#workflow_run). It reads the completed run's event metadata, validates the canonical repository and `issue/**` or `codex/**` branch, confirms that the branch still exists at the run's SHA, then checks for a PR targeting `main`. It never checks out or executes task-branch code or downloads task artifacts. A new PR uses the standard template from `main` and adds the branch, triggering head SHA, inferred Issue, and human-review gates. Issue references are inferred only from `issue/<positive-number>` followed by the end of the name, `-`, or `/`.

An existing open PR is reused; repeated pushes update it through GitHub's branch tracking. Concurrent create conflicts are rechecked before failing, and runs are serialized per branch. If the task branch was deleted before processing, the workflow skips creation and leaves any existing PR untouched. A previously closed or merged PR is not reopened or recreated for a reused branch; start new work on a fresh task branch. The workflow does not approve, merge, or push commits.

The privileged job requests only `contents: read` and `pull-requests: write` for the canonical repository. It uses the standard `GITHUB_TOKEN`, requires no repository secrets, and passes branch names only as API data, never through shell evaluation. Fork and noncanonical repositories are excluded. The Agent Infrastructure and G1 smoke workflows run directly on approved task-branch pushes, so CI evidence does not depend on workflow events caused by the PR-creation token. GitHub suppresses most workflow runs caused by `GITHUB_TOKEN`; on some configurations, `pull_request` runs from workflow-created PRs require approval. See [GitHub's `GITHUB_TOKEN` event behavior](https://docs.github.com/en/actions/concepts/security/github_token#when-github_token-triggers-workflow-runs).

For live use, a repository administrator may need to enable **Settings → Actions → General → Workflow permissions → Allow GitHub Actions to create and approve pull requests**. This setting was not changed or tested for this PR. The workflow still requests only `pull-requests: write`; it does not approve PRs or require a personal token. See [GitHub's repository Actions settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository) and [`GITHUB_TOKEN` permissions](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#permissions).

This change is validated with mocked API calls only; no live PR-creation E2E or repository-setting change is part of this task.

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
