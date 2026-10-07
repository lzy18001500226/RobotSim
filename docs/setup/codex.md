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

The local `Stop` hook receives Codex's stop payload, including `session_id`, `turn_id`, `last_assistant_message`, and `stop_hook_active`. For ordinary turns it sends one `completed` notification per session/turn. If the final message contains one valid marked task-closeout envelope, it persists that structured closeout instead of sending the generic notification. Both paths require the local-only `ROBOTSIM_LOCAL_STOP_HOOK=1` opt-in; recursive Stop events are ignored. Normal hook outcomes leave stdout empty, as Codex expects; failures may be diagnosed on stderr. A manual `stop --dry-run` still prints formatted output without sending mail or writing a GitHub comment. The hook does not run for individual tool calls, tests, commits, or progress updates. Notification errors are best-effort and do not change the task result.

Project command hooks must not be assumed to run in Codex Cloud. Explicitly call `scripts/agent/notify_task.py task-closeout` exactly once before the final response for every task kind. Leave `ROBOTSIM_LOCAL_STOP_HOOK` unset in Cloud: even if project hooks become active there, the Stop handler will skip, so the explicit closeout remains the only path. Reuse the same attempt ID and exact payload when retrying; a genuine rerun gets a new attempt ID. Do not make a second generic notification call for the same task.

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
| `GH_TOKEN` or `GITHUB_TOKEN` | Optional GitHub API token | Secret. Needs Issue and pull-request comment read/write access to `lzy18001500226/RobotSim`; never put it in Git, a command line, a log, or chat. When absent, the notifier can use an already-authenticated `gh auth token` installation. |
| `ROBOTSIM_LOCAL_STOP_HOOK` | Set to `1` in the local Codex environment only | Local Stop-hook opt-in. Leave unset in Codex Cloud. |

### Local Codex setup

1. Create an AgentMail API key in the AgentMail account that owns the inbox.
2. Store the AgentMail key and, if needed, GitHub token with Issue/PR comment read/write access in the operating system's secret manager. Expose them as `AGENTMAIL_API_KEY` and optionally `GH_TOKEN` (or `GITHUB_TOKEN`) to Codex, or use an already-authenticated GitHub CLI. The notifier captures `gh auth token` without printing it. Do not put a credential in a shell command, plaintext shell startup file, repository file, or chat.
3. Set `AGENTMAIL_INBOX_ID=lzy18001500226@agentmail.to` and set `ROBOTSIM_NOTIFY_TO` to the maintainer's normal QQ mailbox through the user's local environment/secret manager.
4. Set `ROBOTSIM_LOCAL_STOP_HOOK=1` in the local Codex environment. Restart local Codex so it receives the configured environment. Trust the actual RobotSim checkout, run `/hooks`, and verify that `Stop` and `PreToolUse` are active.

### Codex Cloud setup

1. In the Codex Cloud Environment settings, add `AGENTMAIL_API_KEY` and `GH_TOKEN` (or `GITHUB_TOKEN`) as environment secrets. The GitHub token needs Issue and PR comment read/write access to this repository. Do not paste secrets into a task prompt or repository file.
2. Configure `AGENTMAIL_INBOX_ID=lzy18001500226@agentmail.to` and the personal `ROBOTSIM_NOTIFY_TO` value in the Cloud environment's protected variable/secret settings. Do not store the QQ address in this repository, and do not set `ROBOTSIM_LOCAL_STOP_HOOK` in Cloud.
3. Allow HTTPS access to `api.agentmail.to` for notifications and `api.github.com` for GitHub task-closeout persistence in the Cloud environment network settings. A Cloud task must explicitly call the notifier once at closeout; project command hooks are not assumed to run there.
4. AgentMail notification is optional. GitHub task-closeout persistence is not: if neither GitHub environment credentials nor an authenticated `gh` CLI can write the canonical Issue/PR comment, closeout is `CLOSEOUT BLOCKED`.

Git transport authentication and GitHub API authentication are separate. An SSH key that can push a branch does not authorize the notifier to create Issue/PR comments, and an HTTPS Git credential is not automatically reused by the notifier. Use an already-authenticated `gh` CLI or the documented environment secret; if neither exists, do not call the closeout published.

The `ready_for_review` status-only command is for attention only; it cannot record task completion. Use the structured task-closeout event below for every terminal task result.

```bash
python3 scripts/agent/notify_task.py ready_for_review \
  --task-id "issue-28-pr-123-review" \
  --task "Issue #28 AgentMail notifications" \
  --summary "Validation passed and the review PR is open" \
  --worker "Codex Cloud" \
  --issue 28
```

Do not send per-tool, per-test, per-commit, or progress notifications. Messages contain a short task label, worker, branch/issue when available, and final summary. Local state markers and AgentMail's idempotency key prevent duplicate sends for the same generic event.

## Durable task closeout

Every task kind uses one canonical `robotsim.task-closeout.v1` event: implementation, experiment, research, review, or audit. Status is `completed`, `blocked`, `deferred`, `failed`, or `cancelled` (`completed` records PASS, `failed` records FAIL, and the other terminal non-pass states record BLOCKED). The compact payload records the canonical repository, Issue-shaped `task_id`, `attempt_id`, worker, task kind/status, summary, branch/head SHA/PR when applicable, validation, durable evidence, blockers, next action, and completion time. Unknown fields and transcripts are rejected.

### Mandatory publication gate

Do not report a task as done after implementation/tests alone. Before any terminal closeout, review and commit all intended task-owned changes, preserve unrelated data, push the task branch, and verify that the push remote has exactly the local HEAD SHA. Then update the corresponding PR or Issue with the structured closeout. Record the status, branch, local and remote SHA, canonical GitHub URL, exact reproduction command, validation result, durable evidence/output reference, and remaining limitations/blockers. If any prerequisite or the GitHub update fails, report `CLOSEOUT BLOCKED`, not `DONE`.

Run the deterministic local gate from the RobotSim checkout; use the task Issue in `task_id` and include `--pr-number` when a PR exists:

```bash
python3 scripts/agent/closeout_gate.py \
  --task-id issue-123-example-task \
  --branch "$(git branch --show-current)" \
  --head-sha "$(git rev-parse HEAD)" \
  --pr-number 456
```

The gate checks the attached checkout, non-default task branch, exact reported HEAD, `git status --porcelain` (including untracked files), the configured `origin` push URL's branch SHA, and the Issue/PR reference. Exit code `2` is `CLOSEOUT BLOCKED`. It does not push, stage, commit, create a PR, or hide unrelated changes. To permit unrelated dirty files, record an exact relative path and a reason in a validation entry using `Unrelated repository path: <path>/ | reason: <why>`; a trailing slash scopes that documented exception to one subtree. Never use a wildcard or broad parent exception for owned task files.

Include concise `validation` entries for `Reproduction command: <exact command>`, `Validation result: <pass/fail/deferred plus result>`, `Evidence/output: <repository-relative path or durable URL>`, and `Limitations: <none or concrete limits>`. Do not use absolute personal paths: the notifier sanitizes them, and raw local paths are not durable references.

The notifier re-runs the gate immediately before writing GitHub. On a failed gate it never writes a `completed` record: it changes the closeout to BLOCKED and attempts to persist that diagnostic on the originating Issue. The CLI exits `2` even if that blocked record was saved. If GitHub persistence itself fails, no durable closeout exists and the final answer remains `CLOSEOUT BLOCKED`. A successful gate appends the exact branch, local/remote SHA, result label, and PR/Issue URL to the canonical record. The Stop hook without a structured envelope now records no terminal completion and sends no generic `completed` notification.

`task_id` has the form `issue-<number>-<stable-slug>`. At task start, create one random attempt ID (for example `python3 -c 'import uuid; print(uuid.uuid4())'`) and retain it with the task notes. A retry of the same closeout reuses that ID and the exact payload. A genuine rerun gets a new ID, including a later `blocked` → `completed` run. Worker is also part of event identity. The `sha256:` event ID is derived from schema version, repository, task ID, attempt ID, and worker, but not status or summary: retries deduplicate, a later attempt is distinct, and changed content under one attempt is rejected. The legacy research envelope derives a compatibility attempt from its timestamp; use the explicit attempt ID for all new work.

Use a concise summary. `evidence` must contain durable public HTTPS references; local paths, private/local hosts, and credential-bearing query/fragment URLs are rejected. Recognized credentials and local filesystem paths in text fields are redacted before persistence. The notifier sends only the structured event, never surrounding final-answer text or a transcript. Local-only Windows/WSL paths are not durable evidence; if useful, describe the diagnostic without the path.

GitHub is the canonical record. If `pr_number` is present, the notifier verifies that PR in the canonical RobotSim repository, requires the PR description to close the originating Issue, and posts the event to that PR's conversation. Otherwise it posts to the Issue number parsed from `task_id`. The closing reference lets someone starting from the Issue discover the PR and then recover the closeout. Comments contain stable hidden markers:

```html
<!-- robotsim-task-closeout:v1:sha256:<event-hash>:<payload-digest> -->
<!-- robotsim-task-closeout-json:v1
{ ... canonical task-closeout.v1 JSON ... }
-->
```

The digest binds the marker to its canonical JSON payload; retries accept it only when the GitHub comment author matches the authenticated account and the digest validates. The GitHub token itself is never stored in the comment. Retries scan all repository Issue and PR comment pages, because event identity is repository-global even when an event can be posted to different destinations. After posting, writers re-read authenticated records across the repository, retain the lowest GitHub comment ID as the one canonical record, and delete other valid records wherever they were posted. A same-ID/different-payload writer reports a conflict; it is never treated as a successful retry. The retained comment stays on its original Issue or PR. A local lock serializes concurrent writers only when they share the same state directory. Native Windows has no cross-process `flock`; only its process-local thread lock applies. On Linux/WSL, `flock` protects processes only when they share the same state-directory filesystem. Local and Cloud generally do not share that directory, so repository-wide GitHub comment reconciliation is the cross-host arbitration. GitHub comments remain the canonical durable record. AgentMail is secondary attention/retrieval: if GitHub persistence fails, no mail is sent and no durable handoff is reported; if mail fails after GitHub succeeds, the task result remains persisted and a retry can recover notification delivery.

### Local Codex

Put exactly one marked JSON envelope in the final assistant message. The Stop hook persists it instead of sending the generic completion, so one turn cannot notify through both paths:

```html
<!-- robotsim.task-closeout.v1
{
  "schema_version": "robotsim.task-closeout.v1",
  "event_id": "auto",
  "repository": "lzy18001500226/RobotSim",
  "task_id": "issue-44-unified-task-closeout",
  "attempt_id": "00000000-0000-4000-8000-000000000001",
  "worker": "Codex",
  "task_kind": "implementation",
  "status": "completed",
  "summary": "Unified closeout is ready for review.",
  "branch": "issue/38-research-completion-event",
  "head_sha": "0123456789abcdef0123456789abcdef01234567",
  "pr_number": 42,
  "validation": ["Python unit tests passed"],
  "evidence": ["https://github.com/lzy18001500226/RobotSim/pull/42"],
  "blockers": [],
  "recommended_next_action": "Review PR #42.",
  "completed_at": "2026-10-04T12:00:00Z"
}
-->
```

Malformed or multiple envelopes fail closed and do not fall through to a generic notification. Recursive Stop events remain ignored, normal synchronous Stop output stays empty, and failures are reported only on stderr.

### Codex Cloud

Cloud tasks explicitly call the same `task-closeout` command once before returning their final answer; Cloud does not depend on project hooks:

```bash
python3 scripts/agent/notify_task.py task-closeout <<'JSON'
{"schema_version":"robotsim.task-closeout.v1","event_id":"auto","repository":"lzy18001500226/RobotSim","task_id":"issue-44-unified-task-closeout","attempt_id":"00000000-0000-4000-8000-000000000001","worker":"Codex Cloud","task_kind":"implementation","status":"completed","summary":"Unified closeout is ready for review.","branch":"issue/38-research-completion-event","head_sha":"0123456789abcdef0123456789abcdef01234567","pr_number":42,"validation":["Python unit tests passed"],"evidence":["https://github.com/lzy18001500226/RobotSim/pull/42"],"blockers":[],"recommended_next_action":"Review PR #42.","completed_at":"2026-10-04T12:00:00Z"}
JSON
```

The explicit Cloud command exits nonzero when the payload is invalid or canonical GitHub persistence fails, so the task must not report a durable closeout in that case. AgentMail is secondary: an AgentMail failure after GitHub persistence is reported on stderr but does not change the command's successful exit status. The Local `Stop` hook always exits successfully so persistence failure cannot block turn completion; it keeps stdout empty and sends sanitized diagnostics to stderr.

Do not send a separate generic `completed` call for that task. Once the comment exists, future ChatGPT sessions need only the repository plus Issue or PR number: list that object's comments and parse the JSON comment paired with the `robotsim-task-closeout:v1` marker. Starting from an Issue, follow its linked PR when the closeout is stored there. To recover all attempts, collect every valid marker; use `completed_at` descending to identify the latest closeout, then read `attempt_id`, worker, and status from the structured JSON. No transcript, local path, or conversation history is needed.

The example IDs, SHA, summary, and evidence above are illustrative. Supply the attempt UUID created at task start and, for a PR, the current head branch/SHA and an active closing reference such as `Closes #<task-issue>` or `Closes lzy18001500226/RobotSim#<task-issue>`. References to another repository, unrelated Issues, and references inside code examples do not qualify.

### Human-gated live tests

The automated tests replace the AgentMail HTTP transport with mocks; CI never sends live email. Do not use Cloud setup validation or `--dry-run` as proof of delivery. After the maintainer configures the real values, the exact live checks still required are:

1. **Local:** restart Codex, trust the real checkout, confirm `Stop` appears in `/hooks`, complete one harmless test instruction with a fresh attempt ID, and verify the GitHub closeout plus exactly one short notification reaches the normal QQ mailbox.
2. **Cloud:** start one harmless Cloud task with the secrets configured, let the task make its single explicit closeout call with a fresh attempt ID, and verify the GitHub closeout plus exactly one notification reaches the same QQ mailbox. Do not rely on project hooks for this check.

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
validate -> self-review -> commit -> push -> auto-created/detected PR (PR body closes task Issue)
  -> required checks/review -> merge only when authorized -> task-closeout status=completed
```

Review-gated task:

```text
validate -> self-review -> commit -> push -> auto-created/detected PR (PR body closes task Issue)
  -> task-closeout status=completed, next action=review PR -> stop for review
```

Blocked task:

```text
preserve durable evidence -> task-closeout status=blocked -> stop
```

Stop at a reviewable PR for architecture decisions, destructive or high-risk changes, dependency/security changes, unresolved GUI/hardware validation, and tasks the user explicitly gates for review. Never bypass branch protection or force merge. Current user instructions always take precedence.
