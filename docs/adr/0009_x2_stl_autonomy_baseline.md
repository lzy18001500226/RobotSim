# ADR-0009: X2 STL autonomy baseline and scene-source boundary

Status: Accepted

## Context

Issue #46 produced several historical OmniPicker experiments. Their model
identities, generated collision assets, and full-X2 visual fixtures are not
interchangeable. The maintainer has now visually accepted one complete X2
assembly with bilateral `X2-OmniPicker-STL` hands in the RAICOM competition
room.

Earlier review attempts accidentally selected the separate RAICOM runtime X2
body or rendered collision proxies together with visual meshes. Those are
different concerns and must not silently select a different robot model.

## Decision

1. `X2-OmniPicker-STL` is the only selectable active OmniPicker identity.
   `X2-OmniPicker-CAD` remains historical evidence and requires an explicit
   caller selection.
2. The frozen STL model has model SHA-256
   `088c1f44963773b6a3cebe0ca3e428b409fa16077bdb930e7b142c5c75981857`
   and trajectory SHA-256
   `117bc4e81edbef8b341341c1cb81baba5a7bde61b590eaea93553ec77c4b540a`.
3. Human-reviewed full-X2 scene artifacts remain `GENERATED_EVIDENCE` under
   governed `reviews/`; they are passed explicitly to research review tools.
   Tracked runtime or scene configuration must not reference a local review
   directory or a disposable worktree.
4. RAICOM room/table/chair are environment sources. They may be attached
   around the approved X2-STL review scene, but the RAICOM runtime `x2.xml`
   must not replace the approved robot body in a visual review.
5. Collision proxy render groups may be hidden for a human visual review only.
   Their `contype`, `conaffinity`, geometry bytes, collision exclusions, and
   other physics behavior remain unchanged.

## Consequences

- A static human review can combine the approved X2-STL fixture and RAICOM
  environment only through an explicit, hash-validated research command.
- Promoting a full-X2 static fixture or generated collision proxies to a
  tracked runtime scene is a separate model-integrity change. It requires a
  source-preserving change package and a new human review.
- Navigation, perception, manipulation, and task execution use model IDs and
  frame contracts; none may load an OmniPicker XML by an old worktree path.

## Qualification

This decision freezes visual/model selection only. It does not qualify native
MC arm control, locomotion, RGB-D perception, balance while reaching, object
contact, physical grasp, or manufacturer-equivalent transmission.
