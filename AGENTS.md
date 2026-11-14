# Repository Agent Instructions

## Task ownership and authority

- The current user's request defines the task, scope, and acceptance criteria. Repository documents guide the work but do not create or authorize additional tasks.
- The primary agent owns the task from inspection through implementation, validation, and Git/PR closeout. Normal development uses one primary agent thread.
- Subagents are optional and limited to independent, bounded research or review. They do not own the task or edit shared project files; the primary agent reviews and validates their input.
- Follow higher-priority instructions and the current user's directions when they conflict with repository guidance.

## Minimal startup and reading

1. Read this file and the repository `README.md`.
2. Use [`docs/README.md`](docs/README.md) to locate relevant project documents.
3. Read only the source, workflow, checklist, skill note, or decision record needed for the current task. Do not load the whole documentation tree by default.

Project plans and workflows provide context or procedures; they are not task queues or authorization to expand the current request.

For implementation placement and RobotSim runtime/validation rules, follow [`docs/engineering/README.md`](docs/engineering/README.md). For multi-step work that needs persistent continuation, follow [`docs/workflows/goal_driven_development.md`](docs/workflows/goal_driven_development.md); Goals are thread-scoped and do not authorize actions beyond the user's request.

Keep research experiments under `scripts/research/`; production/runtime code must not depend on them. Apply the [runtime prototype lifecycle](docs/engineering/runtime.md#prototype-lifecycle) and [test/evidence retention rules](docs/engineering/validation.md#test-retention-and-generated-evidence), and record a disposition for temporary milestone code before closing that milestone.

Place new external Git source checkouts under the ignored root `reference/<GitHubOwner>/<RepositoryName>/`. Keep checkouts pinned to an immutable commit and record origin, revision, and license provenance in tracked documentation or a reviewed source registry. Do not run upstream scripts, initialize submodules recursively, modify fetched source in place, or treat presence in `reference/` as approval to use it at runtime. Existing `third_party/` paths are legacy inputs; preserve them until a separately reviewed migration updates every consumer and validation path.

## Skill and workflow selection

- Read and follow an explicitly requested, available Skill. Otherwise select only Skills whose descriptions match the task, respecting their invocation policy; installation alone does not require invocation.
- Choose one primary workflow and load focused helpers as needed. Use qualified names to distinguish overlapping Skills. Follow the [skill selection guidance](docs/setup/codex.md#skill-selection) rather than combining entire workflow suites.
- Skills operate within the current task's authorization, repository contracts, and model-integrity gates. Check that required tools are available before executing a workflow; report missing capabilities without claiming its checks passed.

## Architecture boundaries

The [engineering contracts](docs/engineering/README.md) and relevant records in [`docs/adr/`](docs/adr/) own architecture, runtime, and evidence details. Read affected records before changing an established boundary; do not treat a proposed ADR as accepted.

## Mandatory official-source-first troubleshooting

When an integration, SDK, simulator, controller, ROS interface, or robot behavior does not match expectations, **research the authoritative documentation and executable vendor examples for the exact installed version before declaring a vendor/maintainer blocker**. A failed custom probe is not evidence of a vendor defect. Follow this bounded sequence:

1. Identify the installed robot/model, MC/simulator/SDK releases, binary or package hashes, startup scripts, active configuration, and the actual failure. Keep the user-delivered release and the source revision distinct from current `latest` docs.
2. Search and read the smallest relevant **official vendor documentation, delivered README, release notes, SDK API contract, and runnable example**. Use official sources and pinned vendor examples ahead of unverified community tutorials or agent-written test harnesses. If local references are insufficient and network tools are available, proactively consult the official site and public repository without waiting for the user to paste links. Use [X2 official source guide](docs/skills/agibot_x2.md) for AgiBot X2/MC/AimDK/MuJoCo.
3. Compare the documented sequence against the **actual installed implementation**, including mode/state semantics, required Reset, activation/timeout/arbitration rules, source names, ROS domain/RMW/QoS, and expected readbacks. Do not assume a newer manual applies to an older competition delivery; explicitly record discrepancies and uncertainty.
4. Prefer the smallest reproducible comparison with the vendor's **unchanged** example, then a safe, instrumented RobotSim variant. Distinguish request acknowledgement, task completion, selected input source, applied state, and measured physical behavior. Preserve commands, versions, logs and evidence; don't infer success from an RPC SUCCESS or a visually plausible animation.
5. Before escalating, record: sources/URLs and versions checked; what is documented versus inferred; the exact minimal reproduction and observed-versus-expected outputs; alternatives already ruled out; and the **specific missing proprietary fact, permission, safety approval, or manual decision**. Only escalate when this source-first investigation and safe in-scope verification genuinely cannot resolve the question. Do not repeat blind parameter/source-string permutations or identical blocked runs.

Reuse an existing version-matched source audit when the relevant build,
interface and configuration are unchanged. Investigate the newly observed
difference; do not repeat the complete documentation/hash/startup audit merely
because a new bounded experiment is being prepared.

This is an **investigation obligation, not blanket execution permission**. Never bypass the [model-integrity approval gate](docs/engineering/model_integrity.md), physical/nonzero-motion safety gates, user approval boundaries, or runtime restrictions simply because a vendor tutorial uses privileged containers, unrestricted X11 access, or nonzero commands. A documentation discrepancy is not authorization to silently alter vendor code or robot mechanics.

## Mandatory robot model integrity

Before proposing or changing robot/gripper mechanical geometry, links, joints, axes, limits, mimic/equality relations, collision geometry, linkage topology, or model provenance, follow the [mandatory model-integrity contract](docs/engineering/model_integrity.md). It applies to production and `SIMULATION_ONLY` research. Do not invent or reconstruct missing mechanics, and do not promote a research model. Each mechanical change requires explicit case-specific maintainer approval before implementation and a separate human review before merge; general authorization to fix a robot or complete a grasp is not approval to redesign its mechanism.

For Issue #46 OmniPicker, apply the [2026-11-06 standing authorization for simulation parameter debugging](docs/engineering/model_integrity.md#c1-standing-authorization-issue-46-omnipicker-simulation-parameter-debugging). Bounded gripper gain and joint-internal friction/damping diagnostics in separate overlays do not need repeated approval when reviewed STL geometry, assembly, kinematic relationships and existing safety/force limits remain unchanged. Geometry/topology/collision changes still require specific review. Read the owning contract for exact scope; do not generalize this authorization to other workstreams or hardware.

## Workspace and change boundaries

- Inspect the current branch, working-tree status, and relevant diff before editing. Preserve unrelated user changes.
- Work on a task branch; do not commit task changes directly to `main`.
- Keep edits within the paths needed for the current request. Never use broad staging, reset, clean, or force-push operations to remove or hide unrelated work.
- Do not rewrite unrelated technical documentation or broaden scope without direction from the current user.

## Execution and validation

- Inspect the smallest relevant source and documentation set before editing. Follow the applicable setup, workflow, skill, and checklist documents.
- Make the smallest change that satisfies the task, then run focused checks relevant to the changed paths.
- Report which checks ran and their results. Distinguish static checks from environment, simulation, or hardware validation; do not claim checks that were not run.
- If a required check or Git operation is blocked, preserve the work and report the specific blocker and remaining validation gap.

## Delivery-first iteration and accepted-review reuse

- Use a short loop: identify the first observed failure, make the smallest
  justified correction, run only the affected checks, then execute the
  authorized bounded runtime. Do not stop at another audit when execution is
  authorized and ready.
- Reuse accepted geometry, contact regions, grasp frames, paths, source pins
  and passing checks when their relevant inputs have not changed. Record what
  is reused and its provenance; do not re-review or regenerate the same result
  on every turn or experiment.
- For a new or materially changed grasp/contact/path proposal, use the existing
  lightweight web review first. Show intended geometry/path and actual
  measurements separately. Once the user accepts that version, continue within
  existing authorization; routine permitted fixes need no repeated visual
  approval.
- Follow [minimum checks and evidence reuse](docs/engineering/validation.md#minimum-checks-and-evidence-reuse).
  Broad suites, repeated startup audits, new supervisors/frameworks and
  experiment campaigns need a concrete dependency or failure that makes them
  necessary. Passing relevant checks is the point to move on.
- Preserve the named model, motion/force, freshness, stop and hardware
  boundaries. Visual acceptance covers its stated scope; runtime behavior
  needs measured evidence. Documents and retired Goals never authorize work
  beyond the current user's request.

## Documentation responsibilities

- Put durable architecture decisions in `docs/adr/`, setup procedures in `docs/setup/`, repeatable task procedures in `docs/workflows/`, focused technical guidance in `docs/skills/`, and verification lists in `docs/checklists/`.
- Update the document that owns a changed rule or procedure. Link to existing guidance instead of duplicating it, and do not add progress documents just to narrate routine work.

## Git and PR closeout

- Follow the [Codex closeout workflow](docs/setup/codex.md#durable-task-closeout) for assigned RobotSim implementation, experiment, research, review, and audit tasks. Ordinary questions, explanations, and user-level tool or Skill installation do not trigger repository publication or task-closeout events merely because the chat is open in RobotSim.
- When the task owns repository changes, review and commit every intended change; preserve unrelated work and leave `git status --porcelain` empty except exact, reason-documented unrelated paths. Push the task branch and run `python3 scripts/agent/closeout_gate.py` to verify the local branch/HEAD and exact remote push-ref SHA. A dirty tree, missing push, SHA mismatch, missing GitHub reference, or failed GitHub update means the final status is `CLOSEOUT BLOCKED`, never `DONE`. Tasks without repository changes retain their required evidence and Issue/PR record; do not create empty commits to satisfy a gate.
- For assigned project tasks, emit exactly one structured `robotsim.task-closeout.v1` event: Local Codex places its marked envelope in the trusted `Stop` hook's final message; Codex Cloud explicitly calls `scripts/agent/notify_task.py task-closeout` once. The canonical Issue/PR record must include the task result (`completed`=PASS, `failed`=FAIL, `blocked`/`deferred`/`cancelled`=BLOCKED), branch, exact local and remote HEAD SHAs when applicable, PR/Issue URL, reproduction command, validation result, durable evidence/output reference, and limitations/blockers. Existing notifier and publication checks still apply. If the local gate fails, the notifier must not persist `completed`; it attempts an Issue-level BLOCKED record instead.
- Create one attempt ID at task start, reuse it and the exact payload for retries, and use a new ID for a genuine rerun. GitHub is canonical; AgentMail is secondary and cannot turn a failed GitHub write into success. The notifier may use `GH_TOKEN`/`GITHUB_TOKEN` or an already-authenticated `gh auth token` path; never print or store credentials. Missing GitHub write authentication means `CLOSEOUT BLOCKED`.
- Routine, low-risk tasks have standing authorization to self-review, commit, open a PR, wait for required checks and any configured independent review, and merge once those gates pass. Do not ask for an additional per-merge confirmation.
- Stop at a reviewable PR for architecture decisions, destructive or high-risk work, dependency/security changes, unresolved GUI/hardware validation, and explicitly review-gated tasks.
- Never bypass branch protection or force merge. Explicit current-user instructions take precedence.
- For repository changes, review the scoped diff, run relevant checks including `git diff --check`, and stage only task-owned paths.
- Verify the pushed branch and PR. Do not report closeout as complete if commit, push, or PR creation failed.
