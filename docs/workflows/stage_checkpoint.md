# Accepted-stage checkpoints and handoff

Use this procedure after a reproducible stage is accepted, before changing its
validated sources, handing ownership to another chat, or ending the work period.
An unfinished overall task can still have a published, recoverable checkpoint.
The primary runtime owner controls its index, HEAD, commits and pushes; use the
[shared-branch contract](../setup/codex.md#shared-branch-ownership-and-diff-only-handoffs).

## Produce the checkpoint

1. Name the accepted stage and remaining task. Inspect the actual result and
   acceptance criteria. Record the model/SDK/container identities, complete
   scoped source list (including new helpers, configuration and version pins),
   frozen source fingerprint, reproduction command and governed evidence.
   Compare the working sources with the accepted run's saved source hashes.
   A fingerprint calculated after an unvalidated edit cannot qualify that edit.
2. At a quiescent boundary, the owner reviews and stages only that source scope,
   commits it, pushes the assigned task branch, and checks the remote SHA. Other
   unfinished work can remain outside this scope with an explicit disposition.
   Keep overall-task acceptance and meaningful merge gates separate: a stage
   checkpoint does not authorize merging unfinished robot behavior.
3. If a maintainer explicitly requested a diff-only handoff, export the scope
   below instead of moving another owner's HEAD. Hand over the source copies,
   patch and metadata together. The maintainer reviews and commits it, then
   records the adopted commit/remote SHA on the existing Issue/PR. Until that
   happens, publication remains unverified.
4. Update the existing short current-state handoff and Issue/PR with the stage,
   checkpoint SHA or packet, evidence and first remaining failure/next action.
   Use one named stage notification if authorized; it is not task completion.
   Before another source-changing iteration or chat handoff, verify that the
   accepted source is recoverable from the pushed commit or complete packet.

If a stage check fails, retain its diagnostics and choose the next authorized
correction. If a completed stage cannot be committed now, deliver its packet
and name the owner/action needed; do not silently accumulate accepted stages
only in an uncommitted worktree. User instructions such as diff-only or no-push
remain binding. Preserve all unrelated work.

## Source packet helper

Run from the owning WSL checkout. Supply an explicit reviewed source list, not
the whole dirty tree. The following shows argument structure; replace the
example scope with the complete source list of the actual accepted stage:

```bash
python3 -B -m scripts.agent.stage_checkpoint --fingerprint-only \
  --source scripts/research/issue46/native_runtime/stl_hand_startup.py
```

Capture that fingerprint before the stage execution, or verify it against the
preserved validated source. Export with the matching value and actual evidence:

```bash
python3 -B -m scripts.agent.stage_checkpoint \
  --stage truth-transfer-accepted \
  --source scripts/research/issue46/native_runtime/stl_hand_startup.py \
  --validated-fingerprint <validated-source-fingerprint> \
  --reproduce '<exact command and actual model/config/version inputs>' \
  --evidence /home/lzy18001500226/projects/RobotSim/reviews/issue-46/<accepted-run>/result.json \
  --output-parent /home/lzy18001500226/projects/RobotSim/reviews/issue-46
```

Repeat `--source` and `--evidence` for the stage's files. Supply explicit
`--deleted-source` entries for removed paths; represent a
rename as the old deleted path plus the new source path. Include the same
deletion entries in the pre-stage fingerprint. Packets preserve tombstones and
deletion patches so restoration cannot silently retain a retired entry point.
Published verification checks each selected file's committed bytes and
executable bit, including files ignored by Git status. Published checkpoints
add `--verify-published`; it requires a clean selected scope on a dedicated
branch whose remote HEAD equals local HEAD. Unrelated dirty work is preserved.
The default export writes `publication=NOT_CHECKED` and never commits, pushes,
runs the reproduction command, or sends a notification.

The unique output contains `source/` with full regular-file contents and modes
(including untracked files), `review.patch` against the recorded base HEAD,
`stage_handoff.json`, and the verified `artifact_paths.json` manifest. Source,
HEAD, scoped Git status and evidence must remain unchanged during export.
Rejected/partial exports are not completed handoffs. The receiver verifies the
manifest and source hashes before using a packet, and restores into an isolated
review location rather than overwriting an active runtime checkout.

The exporter verifies identity, integrity and optional publication. Stage
acceptance is a producer assertion backed by evidence; this helper does not
evaluate physical success or infer whether the source list is complete.
Record missing capabilities, unrun tests and the exact remaining task plainly.
Credentials, `.git/`, local `.codex/` configuration (except project `hooks.json`),
upstream checkouts and review outputs are outside the exportable source scope. Keep immutable vendor asset/license pins
in the handoff rather than copying an upstream checkout.

## Required handoff fields

| Field | Receiver's check |
| --- | --- |
| Goal, accepted stage, remaining failure and next action | Stage acceptance is narrower than whole-task completion. |
| Owner, checkout, branch and base/checkpoint SHA | Git operations stay with the assigned owner. |
| Source list, fingerprint, new files and source packet/commit | Restore the accepted behavior without the old chat's memory. |
| Model/SDK/container pins and reproduction command | Required inputs exist and versions are explicit. |
| Acceptance results and durable/local evidence references | Inspect measured results; separate physical execution from replay/unit checks. |
| Unrelated work, limitations and publication state | Packet-ready, committed, pushed, PR-ready and merged are distinct. |
| Protected evidence and resume inputs | Apply the [batch archive checklist](../checklists/simulation_batch_archive.md). |
