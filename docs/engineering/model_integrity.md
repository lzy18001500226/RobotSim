# Robot Model Integrity

This is a mandatory, high-priority RobotSim contract for vendor robot and gripper models. It applies to production assets, adapters, conversions, test fixtures, and `SIMULATION_ONLY` research. A simulation-only label narrows a claim; it does not grant permission to invent a mechanism.

## A. Official-source-first

- Prefer the authoritative vendor robot description and original geometry for the exact robot/version being modeled.
- Record the upstream repository, immutable commit or release, source file path, relevant SHA-256 values, and applicable license/provenance notes.
- A repository-level license does not automatically establish redistribution rights for every model or mesh file.
- Never silently substitute generated geometry, a different product revision, or another robot's mechanism for official assets.

## B. No unauthorized mechanical reconstruction

Without explicit, case-specific maintainer approval, agents must not create, invent, simplify, or reconstruct mechanical model behavior. Prohibited changes include:

- replacing an official hand or gripper with a custom mechanism;
- inventing links, fingers, hinges, bearings, axes, joint origins, limits, or assembly placements;
- changing a fixed joint to revolute or vice versa;
- changing linkage, mimic, gear, spring, tendon, or closed-loop relationships;
- adding equality, weld, or loop constraints not established by an authoritative source;
- changing collision geometry, contact exclusions, or collision masks to make a grasp or simulation pass;
- presenting an experimental equivalent mechanism as the manufacturer's model.

These restrictions also apply to offline experiments and research branches.

## C. Case-specific human approval

Before implementing any mechanical-model change, stop and present the maintainer with:

1. the exact source files and pinned upstream revision;
2. the observed defect or missing information, with reproducible evidence;
3. the proposed delta and exact affected links, joints, transforms, geometry, and constraints;
4. authoritative mechanical/vendor documentation that supports the delta;
5. expected kinematic, collision, and dynamics effects; and
6. the validation plan and claim boundary.

Proceed only after explicit approval for that specific proposal. General instructions to fix a gripper, improve contact, or complete a grasp do not approve mechanical redesign. Mechanical-model changes are high-risk and are never an automatic self-merge case.

## D. Stop when source semantics are missing

If source geometry, assembly, loop semantics, joint type, pin pairing, collision policy, or physical parameters are insufficient, stop before building a guessed mechanism. Record the exact missing information and request manufacturer/maintainer clarification. Do not tune guessed parameters or repeat speculative variants to force a result.

## E. Immutable upstream and source-preserving adaptations

- Treat fetched vendor files as immutable. Put an authorized adaptation in a separate, reviewable overlay and retain the exact upstream-to-adaptation diff.
- A conversion is allowed only when its source revision, units, transforms, axes, limits, inertias, geometry mapping, and constraint semantics are understood and checked.
- Preserve unsupported or custom upstream tags as unresolved; do not silently choose their mechanical meaning.
- Keep source asset identity separate from simulator-only actuators, fixed-pose constraints, collision filters, fixtures, and controller assumptions.

## F. No automatic model promotion

Research and generated models must not become canonical robot assets, production runtime dependencies, or default integration models without explicit maintainer approval. A successful simulation is evidence only for the measured setup; it does not prove source fidelity or hardware authenticity.

Classify model inputs and outputs as `OFFICIAL_UPSTREAM`, `VERIFIED_SOURCE_PRESERVING_ADAPTATION`, `UNVERIFIED_EXPERIMENTAL_MODEL`, `GENERATED_EVIDENCE`, or `UNKNOWN_REQUIRES_HUMAN_REVIEW`. Unknown provenance remains unknown. Keep historical evidence, but isolate unverified experimental models from production and runtime.

## G. Enforcement boundary

`scripts/agent/check_governance.py` verifies that this contract remains linked from root `AGENTS.md` and rejects direct runtime-source references to `scripts/research/` or local `reviews/` evidence. This static check does not prevent an agent from editing a model file, validate a claimed approval, or establish vendor fidelity. Human review remains required. Stronger enforcement through `CODEOWNERS` and branch protection is a separate repository-settings decision; it is not implied by this document or by passing CI.

The current Issue #46 inventory is a review-only [model-integrity and cleanup manifest](../research/issue46_omnipicker_model_integrity_cleanup_manifest.md). Its proposed dispositions do not authorize cleanup.
