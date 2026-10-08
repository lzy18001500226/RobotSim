# Issue #46 Outcome-Driven X2 Grasp Corridor

## Disposition

**Path B was selected:** the preceding isolated source-faithful X2 work had no physical lift PASS. Two geometry-derived alternatives were evaluated. Both fail the static corridor gates, so no X2 physics was run. This is not evidence that every OmniPicker pose is infeasible.

**Source-faithful static grasp: FAIL for both tested alternatives.**

**Physical bilateral contact: NOT RUN.**

**Physical 30 mm lift: NOT RUN.**

**Design approval required: YES** before changing collision representation, source geometry, bottle, or end effector in the mainline.

The useful next step is a maintainer decision about the intended physical envelope of the wrist-roll and loop structures. This investigation does not authorize a production model change or another pose search.

## Provenance

- Research branch: `research/issue46-outcome-grasp-corridor-20261009`.
- Starting RobotSim HEAD: `f30720505b99bb0b85686ba2a4aa3d7700243890`.
- X2 source: `AgibotTech/agibot_x2_urdf`, commit `575cc6b988f976c23550e0db85aa1e5475d3652d`, Mulan PSL v2.
- URDF: `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`; SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`.
- Runtime: Python `3.10.12`; MuJoCo Python/native `3.3.6`; `MUJOCO_GL=egl`.
- Native MuJoCo library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Canonical bottle was unchanged: 70 mm body diameter, 244.5 mm overall height, 0.570 kg, with body, shoulder, neck, and cap collision parts.
- Canonical scene helper SHA-256: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.
- Outcome evaluator SHA-256 captured in the complete body run: `64f602ee9dd79e4db854cb7c57bfeee47400f022bda7cda24d42ad9dd8cd8fe7`.
- Source corridor helper SHA-256: `8304c076b79478f904bac41a7cd82dfdfaaf8b52e0e0d9daaf2eb74c973edde0`.
- Prior accepted corridor result SHA-256: `46f708bb8841456b3fe11b68eeeaef4fb4c704bc304b845e5da50102a4ac9120`.
- `run-07/result.json` SHA-256: `71ec6e4be5d71553c61b6ce4a9e06615cbe0966961d0619ab8987c1e120b3466`.
- `run-08/result.json` SHA-256: `f6d056fc199501d46d4d2982faf4d974bfdefe209b43b37ab6c1bf5c3c70b675`.

`run-07` contains the complete neck candidate and the initial body attempt. The body attempt did not retain the open/approach trace after finding no aperture bracket. `run-08` is the bounded evidence recovery for that same body candidate and contains the full OPEN, PREGRASP, approach, and pair-clearance data. The `run-07` runtime record did not capture a separate evaluator SHA; its report's `runner_sha256` is the source corridor helper hash above. No raw run files were changed.

## Candidate Results

The wrist pose was derived analytically from the compiled insertion axis and the source roll limits, selecting the permitted roll that minimizes vertical insertion. Both hypotheses use yaw/pitch/roll `[0, 0.523599, 1.50971]` rad and approach direction `[0.9963995866, 0.0802640308, 0.0273047475]`. This is near-horizontal, not exactly horizontal. The candidates differ by the target bottle region only.

| Candidate | OPEN and approach | Opposing target jaws | Static 30 mm corridor | Result |
| --- | --- | --- | --- | --- |
| Body side, `bottle_body` | At OPEN, jaw collision distances are `-8.414 mm` and `-28.986 mm`; wrist-roll/body minimum is `-53.103 mm`; robot/table minimum is `-15.923 mm`. At the 80 mm pregrasp, bottle clearance is `+21.548 mm`, but table clearance is `-13.739 mm`. | No aperture bracket: both target jaw geoms are already penetrating the body at the OPEN and source-closed endpoints. | Not valid; the initial OPEN state and approach fail. | FAIL |
| Neck side, `bottle_neck` | Target jaw clearances at OPEN are `+27.099 mm` and `+26.848 mm`, but the wide distal link intersects the bottle body by `36.562 mm` with 10 robot/bottle contact candidates. The 80 mm pregrasp is clear; approach first fails at sample 9/21 with `wide3`/body distance `-0.489 mm`. | Static closest-point queries reach `+1.023 µm` and `+1.043 µm` on opposite sides of the neck target, with monotonic jaw clearances. This is only a geometric bracket. The full gripper has non-target collision during closure, so the acceptance gate fails. | Fails already at zero translation: `wide3` intersects the shoulder by `29.995 mm`; this is not a lift. | FAIL |

For the neck candidate, the tip witness offsets along the jaw normal are approximately `-26.301 mm` and `+26.301 mm`. The compiled jaw normal is `[0.063390, -0.493807, -0.867258]`, so the witness offsets alone do not imply an ideal radial pinch. The bottle remains supported by the table in all static queries. No force-bearing grasp or lifted state is claimed.

The compiled checks included the complete multipart bottle and original active collision geoms. Source joint limits passed in both candidates. The body candidate had no initial robot self-contact; the neck candidate had no robot self-contact and clear table geometry. The failures are bottle/table and non-gripping robot/bottle clearance failures, not source-limit or self-collision failures.

## Relation To Prior Collision Evidence

The preceding A/B/C diagnostic remains relevant and was not repeated:

- The pinned source collision wrist-roll mesh hull volume was measured at `3.3611x` the visual-mesh hull; `69%` of collision vertices lay outside the visual hull. At the source contact frame, sampled collision triangles extended to `-34.792 mm` relative to the bottle body while visual triangles were `+23.423 mm` clear. This is evidence of a substantial collision/visual envelope difference, not proof that either mesh is physically correct.
- Replacing wrist collision with the visual mesh did not clear the full gripper corridor; the measured non-gripping interference remained `-35.489 mm` in that diagnostic.
- Six source-triangle convex pieces per wrist/loop mesh retained `-37.143 mm` interference at the original pose and `-44.035 mm` at the centered pose. Narrow and wide loop piece volume ratios were `1.609` and `1.600` relative to the single hull.
- The 81-component hull diagnostic increased the recorded contact rows to 38 and reached `-56.153 mm`; it is not a suitable production collision representation.

These counterfactuals do not identify the physically correct replacement surface. They also do not support a simplistic visual-mesh or convex decomposition fix.

The external packet includes `collision-counterfactuals/wrist_collision_vs_visual_projection.png`, a top/side projection of the original pinned collision mesh (red) and vendor visual mesh (blue) against the bottle silhouette. It visualizes the tested visual-mesh substitution only; it is not a proposed production proxy. The current body and neck OPEN/PREGRASP/contact geometry images remain in their respective run folders. No alternate proxy geometry is presented as a proposal because available evidence does not identify which envelope matches the physical housing.

## Minimal Design Proposal

1. Keep the pinned source-faithful collision model as the baseline and preserve all current evidence.
2. Ask the maintainer to confirm whether the wrist-roll and loop collision STL files describe the external physical housing, an intentionally conservative safety envelope, or an importer-oriented envelope. If available, use vendor CAD or measured hardware dimensions as the reference.
3. Only after that decision, authorize a separate diagnostic model for the specifically verified non-contact enclosure surfaces. Keep visuals, kinematics, source limits, bottle geometry, mass, and friction unchanged. Make any proposed collision replacement source- or measurement-derived and show a before/after surface overlay.
4. Require the diagnostic model to pass zero active non-gripping penetration, a collision-free approach, opposing jaw contact, and static lift clearance before running physics. Do not treat a diagnostic collision proxy as a source-faithful PASS or promote it without a separate maintainer approval.

The previous visual-mesh substitution and generic convex partitions already failed, so this proposal is a physical-envelope verification gate, not a request to repeat those counterfactuals.

## Evidence And Reproduction

Durable external evidence directory:

`C:\Users\HP\Desktop\Robot\reviews\issue-46-x2-single-hand\outcome-grasp-corridor-20261009\`

- `run-07/`: neck candidate result JSON, full state and pair-clearance CSVs, and OPEN/PREGRASP/contact geometry views. The body record in this run is partial and is superseded by the same candidate's complete static trace in `run-08/`.
- `run-08/`: complete body candidate result JSON, state and pair-clearance CSVs, and OPEN/PREGRASP front, side, top, and close-up views.
- `collision-counterfactuals/`: a copied original-collision-versus-visual-mesh projection from the earlier A/B/C evidence; the original is preserved in its prior evidence directory.
- `root-cause-abc-20261008/`: prior collision-envelope and counterfactual evidence cited above.
- No MP4 exists because `mj_step` was never called.

Exact commands are in [issue46_outcome_driven_grasp_corridor_20261009_reproduce.md](issue46_outcome_driven_grasp_corridor_20261009_reproduce.md). Both commands are static geometry queries. They do not run bottle physics, change production files, or evaluate whole-humanoid arm IK.

Issue #46 remains open. PR #58 was verified OPEN and Draft; this research does not merge or change it.
