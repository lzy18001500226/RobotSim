# Independent review: Issue #48 AgentMail notifier

Reviewed repository `main` at `bbf2090b0ccb7b50190dac33d7b04997361c9fd5` ("Unify durable Codex task closeout"). Issue #48 was open and had no linked PR at review time. This review covered the current notifier source, hook configuration, docs, and tests statically. No code was modified; no tests or live email were run.

## Review result

**No secret-handling, endpoint, redirect, SMTP, or GitHub-before-email ordering blocker found in the notifier source.** Issue #48's live-delivery acceptance remains unverified. The tests mock both APIs, and the issue/docs reserve sender/inbox checks for a local or human-bound E2E. Do not report the live notification as delivered or close the E2E gap from these tests alone.

## Findings

### Security and AgentMail API

- `AGENTMAIL_API_KEY` is read from the environment and sent only as a Bearer header to the fixed `https://api.agentmail.to` endpoint. It is not passed on the command line or persisted. API failures log only status/type; response bodies and headers are discarded.
- Both AgentMail and GitHub requests use `_open_request` with `_NoRedirect`, so a 3xx response is an error rather than forwarding either API token to a redirect destination. The tests directly cover the handler rejecting a redirect; they do not perform a real AgentMail request.
- The send path uses `/v0/inboxes/{inbox_id}/messages/send`, JSON `to`/`subject`/`text`, Bearer authentication, and `Idempotency-Key`. These match the cited official AgentMail Python SDK source. That SDK documents idempotency keys as expiring 24 hours after send completion.
- Text fields redact recognized credentials and local paths; evidence URLs reject userinfo, local/private destinations, and credential-like query/fragment parameters. Structured closeout sends only the normalized JSON event, not a transcript.
- No SMTP client, SMTP endpoint, QQ SMTP password, or authorization-code setting exists in the notifier. `ROBOTSIM_NOTIFY_TO` is a configured recipient address; authentication is the AgentMail API key. The sender is selected by the AgentMail inbox ID.

### Ordering, persistence, retries, and duplicates

- Structured closeout persists to the canonical GitHub Issue/PR first. A GitHub persistence failure returns before AgentMail is called; the explicit command exits nonzero for this case.
- After GitHub succeeds or confirms an identical existing closeout, it attempts AgentMail. AgentMail failure is reported on stderr but leaves GitHub canonical and the explicit command successful. A retry with the exact event payload reuses the GitHub event ID and AgentMail idempotency key; GitHub does not repost the durable record, and AgentMail can deduplicate an ambiguous send within its documented 24-hour window.
- Local `.pending`/`.sent` files prevent duplicate sends on a shared state directory; file creation is exclusive and the directory/files use restrictive modes. Local and Cloud state directories are not assumed to be shared. Therefore cross-environment deduplication relies on AgentMail's 24-hour provider key; a same-event retry after that window on a different state directory can send again. The local state directory can suppress longer retries on that same persistent machine.
- **Non-blocking reporting issue:** `_claim_delivery` returns `duplicate or already in progress` when a `.pending` claim exists, but `_deliver` maps that to `Outcome("duplicate", ...)`, and `process_task_closeout` prints “AgentMail already sent.” A concurrent sender may still be in progress, or a prior process may have died before delivery. This can overstate delivery; a distinct pending/unknown result would be more accurate. GitHub remains durable either way.
- AgentMail mail and GitHub notification mail are separate effects: the notifier posts a GitHub comment first, which may independently trigger GitHub Notifications, then submits a separate request to `api.agentmail.to`. Mock tests can distinguish hosts, but only a real mailbox/provider inspection can prove the AgentMail sender and distinguish its message from GitHub mail.

### Local Stop vs explicit notifier

- `.codex/hooks.json` registers `Stop` as an optional hook invoking `notify_task.py stop`. The local path requires `ROBOTSIM_LOCAL_STOP_HOOK=1`; when the final message has a valid closeout envelope it processes that event instead of sending a generic turn notification. Local Stop remains non-blocking.
- `python3 scripts/agent/notify_task.py task-closeout` is a separate explicit stdin-driven CLI path. It does not inspect or require the Stop-hook opt-in, so it can be invoked when Codex App does not load/enable project hooks. The docs correctly say Cloud must call it explicitly.
- Tests cover GitHub-failure ordering, AgentMail-failure retry, local Stop behavior, a stable idempotency key after ambiguous transport failure, and one REST request shape. They mock all network calls; they do not prove live delivery or exactly-one mailbox receipt.

## Issue #48 acceptance status

The code path is reviewable and the durable ordering is correct. The Issue's live acceptance—Stop visibility, one local closeout, exactly one AgentMail message from the inbox, distinct from GitHub Notifications—remains **DEFERRED / NOT RUN** from this review. No live secret-backed send was attempted.
