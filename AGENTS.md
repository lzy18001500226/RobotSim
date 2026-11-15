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
   Use a skill only when its workflow is needed for the current request; a
   related keyword or narrow follow-up does not justify a broader workflow.

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

When an SDK, simulator, controller, ROS interface or robot behavior fails,
consult version-matched official documentation and delivered executable
examples before declaring a vendor/maintainer blocker. A failed custom probe
alone does not establish a vendor defect. For X2, use the
[official source guide](docs/skills/agibot_x2.md).

1. Identify the actual failure and installed robot/model, MC/simulator/SDK
   releases, hashes, startup scripts and active configuration. Keep delivered
   packages distinct from source revisions and current latest documentation.
   Reuse an existing audit if those inputs are unchanged; inspect the new
   difference instead of repeating the inventory.
2. Read the smallest relevant official README, API contract, release note or
   runnable example; consult official online sources when local evidence is
   insufficient. Compare installed startup/Reset, mode/FSM and asynchronous task
   semantics, arbitration/timeouts/source identity, domain/RMW/QoS and readbacks.
   Record version mismatches rather than assuming a newer manual applies.
3. Prefer the smallest authorized comparison with an unchanged vendor example,
   then an instrumented RobotSim variant. Distinguish acknowledgement, task
   completion, selected source, applied controller state and physical behavior.
   Preserve commands, versions, timestamps and logs.
4. Escalate only with the sources/versions checked, exact reproduction,
   observed versus expected result, ruled-out alternatives and the specific
   missing fact, permission or decision. Choose a new evidence-backed
   hypothesis instead of repeating failed parameter/source-string guesses.

Investigation grants no execution permission. Keep model-integrity, motion,
hardware and user approval boundaries; a tutorial or documentation mismatch
does not authorize privileged containers, unrestricted X11, vendor edits or
unapproved mechanics.

## Mandatory robot model integrity

Before proposing or changing robot/gripper mechanical geometry, links, joints, axes, limits, mimic/equality relations, collision geometry, linkage topology, or model provenance, follow the [mandatory model-integrity contract](docs/engineering/model_integrity.md). It applies to production and `SIMULATION_ONLY` research. Do not invent or reconstruct missing mechanics, and do not promote a research model. Each mechanical change requires explicit case-specific maintainer approval before implementation and a separate human review before merge; general authorization to fix a robot or complete a grasp is not approval to redesign its mechanism.

## Workspace and change boundaries

- Inspect the current branch, working-tree status, and relevant diff before editing. Preserve unrelated user changes.
- Reuse the assigned task branch, registered worktree and existing PR. Follow-up
  work is not a reason to create another worktree; resolve paths from Git.
  Do not commit task changes directly to main.
- When another agent owns the same active branch, follow
  [shared-branch ownership](docs/setup/codex.md#shared-branch-ownership-and-diff-only-handoffs)
  before changing the index or HEAD.
- Keep edits within the paths needed for the current request. Never use broad staging, reset, clean, or force-push operations to remove or hide unrelated work.
- Do not rewrite unrelated technical documentation or broaden scope without direction from the current user.

## Execution and validation

- Inspect the smallest relevant source and documentation set before editing. Follow the applicable setup, workflow, skill, and checklist documents.
- Make the smallest change that satisfies the task, then run focused checks relevant to the changed paths.
- Report which checks ran and their results. Distinguish static checks from environment, simulation, or hardware validation; do not claim checks that were not run.
- If a required check or Git operation is blocked, preserve the work and report the specific blocker and remaining validation gap.

## Local tools and desktop interaction

- Prefer terminal commands, existing APIs and programmatic reads for local
  process management, authorized simulation startup/shutdown, logs, state
  inspection and screenshots.
- Prefer available renderer, window-capture or screenshot APIs/commands that
  capture without activating a window or moving the pointer. Taking a screenshot
  alone is not a reason to take over the user's desktop.
- Use Computer Use only when necessary GUI interaction lacks a suitable
  programmatic interface. Scope it to the required application and interaction;
  avoid stealing focus or moving/clicking the mouse or typing while the user is
  using the computer.
- These are workflow preferences, not claims about tool capabilities. Check the
  tools actually available; do not describe screenshots or GUI operations as
  unsupported merely to justify a preferred workflow.

## Delivery-first iteration and accepted-review reuse

- Default to completing the requested engineering task with a working,
  reproducible end-to-end flow. Research, benchmarks and general frameworks
  serve a concrete delivery need; they are not additional deliverables by default.
- Read the user's latest correction in context. For noisy wording or minor
  reversible choices, state a reasonable working assumption and proceed.
  Clarify only a consequential ambiguity that existing evidence cannot resolve;
  do not ask again for an already authorized routine action.
- Keep the plan proportional: name the next measurable outcome and first
  uncertainty, then act. Use observed failure -> smallest correction -> affected
  checks -> authorized bounded execution -> measured result. Repeated reading,
  plans or reports are not progress without a new decision or result.
- Reuse the existing SDK, controller, route/IK/state-machine entry points and
  accepted evidence. Add a helper, supervisor, retry layer or new abstraction
  only for a demonstrated need in the current task, not hypothetical future use.
- Reuse accepted geometry, contact regions, grasp frames, paths and source pins
  while their relevant inputs are unchanged. For materially changed geometry
  or path intent, use the existing lightweight web review; accepted routine
  fixes need no repeated visual approval.
- Follow [minimum checks and evidence reuse](docs/engineering/validation.md#minimum-checks-and-evidence-reuse).
  Each investigation/check must distinguish a named uncertainty. Once the
  relevant checks pass, take the next authorized action; if a blocker affects
  only one part, continue independent in-scope work.
- Lead updates with actual progress, the first failure and the next action.
  Report implementation, runtime verification and publication separately.
  Preserve named model, motion/force, freshness, stop and hardware boundaries;
  task completion requires measured evidence, not success-shaped reports.

## Documentation responsibilities

- Put durable architecture decisions in `docs/adr/`, setup procedures in `docs/setup/`, repeatable task procedures in `docs/workflows/`, focused technical guidance in `docs/skills/`, and verification lists in `docs/checklists/`.
- Update the document that owns a changed rule or procedure. Link to existing guidance instead of duplicating it, and do not add progress documents just to narrate routine work.

## Git and PR closeout

- Follow the [durable task-closeout workflow](docs/setup/codex.md#durable-task-closeout)
  for assigned RobotSim implementation, experiment, research, review and audit
  tasks. Ordinary questions/explanations and user-level tool/Skill installation
  do not trigger repository publication or closeout merely because this chat
  is open in RobotSim. The owning workflow defines event/schema details.
- Review the scoped diff, run relevant checks including git diff --check, and
  stage only task-owned paths. Commit meaningful functional or reproducible
  checkpoints, not each probe or tool call. Preserve unrelated edits; never
  reset, clean or broadly stage them to force a clean status.
- For authorized publication, commit and push, then run
  scripts/agent/closeout_gate.py to verify the branch, exact local/remote HEAD
  and GitHub reference. A failed required publication/gate is CLOSEOUT BLOCKED,
  not DONE. Record exact unrelated-path exceptions and limitations.
- For an accepted intermediate milestone, use the
  [explicit milestone notifier](docs/setup/codex.md#milestone-email-notifications)
  once with a stable stage ID and actual evidence. A turn ending or an API
  acknowledgement is not a milestone; ordinary chat must not trigger email.
- Emit the documented structured closeout event once; GitHub is canonical.
  A requested diff-only collaboration is an intermediate handoff, not authority
  to commit, push, merge or announce published completion.
- Routine low-risk tasks have standing authorization for self-review and normal
  Git/PR closeout. Merge only after required checks/reviews pass. Stop at a
  reviewable PR for architecture, destructive/high-risk or security/dependency
  changes, unresolved GUI/hardware validation and explicit review gates.
  Never bypass branch protection or force merge; current user limits prevail.
