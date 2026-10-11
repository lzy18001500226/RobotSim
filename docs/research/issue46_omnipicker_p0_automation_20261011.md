# Issue #46: X2 OmniPicker P0 Acceptance

## Result

**FAIL — `ACTUATION_FAILURE`: the right claw crosses its official lower position limit during the isolated fixture approach, before bottle contact.** The first attempt and one slower, evidence-driven approach repeat the same failure mechanism. Stop here; do not infer that the OmniPicker cannot grasp the bottle, because no grasp contact was reached.

| Gate | Result | Evidence |
| --- | --- | --- |
| Official source and assembly | PASS, with historical simulation-only collision exclusions recorded | `run05/X2_OMNIPICKER_ASSEMBLY_GATE.json` |
| Independent FK | PASS | `run05/independent_fk_comparison.json` |
| Actuated open/close, left/right/both | PASS, simulation-only mimic emulation | `run05/OPEN_CLOSE_GATE.json` and three traces/videos |
| 70 mm cylinder static corridor | PASS, static geometry only | `run12/B2_STATIC_GEOMETRY_GATE.json` |
| Isolated physical contact and 50 mm lift | FAIL before first hand contact | `run11/C_CONTACT_LIFT_TRACE.jsonl`, `run12/C_CONTACT_LIFT_TRACE.jsonl`, `run12/C_RESULT.json` |
| Full X2 contact lift | NOT RUN (C prerequisite failed) | — |
| Full X2 release/settle | NOT RUN | — |

The report does not claim a physical bottle grasp or lift. `C_maximum_lift.png` and `C_isolated_contact_lift.mp4` show the failed pre-contact approach endpoint, not a lift.

## Source and runtime

- Source: [AgibotTech/agibot_x2_urdf](https://github.com/AgibotTech/agibot_x2_urdf), commit `575cc6b988f976c23550e0db85aa1e5475d3652d`.
- URDF: `X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf`; SHA256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`; license Mulan PSL v2.
- Runtime: WSL2 Ubuntu 22.04, Python 3.10.12, MuJoCo Python/native 3.3.6, NumPy 1.26.4, Pinocchio 2.7.0, 0.002 s timestep, gravity `(0,0,-9.81)`, 100 solver iterations, EGL.
- Native MuJoCo library SHA256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`.
- Canonical scene helper SHA256: `41165d99be43a55fc3e3d9c76175f35a9e98539c4421f380bfe44b95c59652ae`.
- RobotSim research branch at test time: `research/issue46-omnipicker-p0-automation-20261011`, base HEAD `1fa0ca7ac48ec3e1fe2d8e93fb748cf8333a637a`.

## Assembly and motion evidence

The compiled model contains 62 bodies, 106 geoms, 35 joints, 50 meshes, `nq=35`, and no source equality constraints or actuators (`neq=0`, `nu=0`). The independent FK comparison covered 61 body frames in three configurations. Maximum translation error was `6.776e-16 m`; maximum rotation error was `5.162e-8 rad`, within the recorded `1e-8 m` and `1e-7 rad` limits.

The URDF's distinct right-wrist visual (`right_wrist_roll_extend_link.stl`) and collision (`right_wrist_roll_link.stl`) meshes were both retained. The collision geom is active and the visual geom is non-colliding. Their recorded nearest-vertex cloud metric is asymmetric: collision-to-visual p95/max `117.266/127.255 mm`; visual-to-collision p95/max `6.290/11.535 mm`. This is a nearest-vertex cloud metric, not a surface Hausdorff distance; no mesh substitution was made. The compiled/source inventory comparison found matching axes and ranges. The assembly build records 11 pre-existing simulation-only collision exclusions and zero newly added exclusions. Accordingly, the assembly PASS is qualified by those historical exclusions and is not an unqualified source-faithful collision claim.

Left-only, right-only, and simultaneous OPEN/CLOSE/HOLD/OPEN each completed 3,250 MuJoCo steps. All three runs recorded zero source-limit violations, zero unexpected contacts, zero saturated steps, zero non-finite state, and zero rollout qpos writes. Maximum mimic residuals were `0.0003881 rad` left and `0.0003727 rad` right. These runs use **SIMULATION_ONLY actuated mimic emulation**, not a hardware-calibrated or physical transmission model.

## Cylinder geometry and failed dynamic attempt

The static fixture uses a free-jointed MuJoCo cylinder with diameter `0.070 m`, height `0.2445 m`, mass `0.570 kg`, and friction `(1.4, 0.02, 0.001)` on the canonical table (`top z=0.8 m`). The official compiled right wrist and complete right OmniPicker descendant collision geometry were retained. The compiled open jaw gap is `103.091 mm`. The static open approach has `3.625 mm` minimum non-pad/bottle clearance and a conservative `55.404 mm` gripper/table clearance lower bound. At the selected closure driver coordinate `0.334`, the compiled `R_hand_narrow3_Link` and `R_hand_wide3_Link` collision surfaces reach opposite sides of the bottle; the geometry-only contact penetrations are `0.528 mm` and `1.213 mm`. The static contact force samples are not dynamics or load-transfer evidence. The 25-sample approach and 1,401-sample closure have no active table contact and no non-gripping bottle collision. This is a zero-step geometry result, not proof of a physical grasp.

The first dynamic run (`run11`) stopped at step 281, `t=0.562 s`: `right_claw_joint=-1.0000740162 rad` against the unchanged `[-1,0] rad` source range. A single trajectory-only correction doubled fixture approach duration from `1.5 s` to `3.0 s`; jaw gains, effort caps, geometry, friction, and source limits were unchanged. The corrected run (`run12`) stopped at step 318, `t=0.636 s`, at `-1.0000227265 rad`, a `22.727 µrad` lower-limit violation. Thus slower approach reduced the overshoot but did not clear the hard limit.

At the failing step, target was `-0.999 rad`, velocity was `-0.0178105 rad/s`, acceleration was `-0.157717 rad/s²`, commanded/applied claw torque was `+2.85225e-5 Nm`, and the actuator was not saturated. State was finite; mimic residual was `0.00017433 rad`. The only bottle contact was with the table: hand support was `0 N` and table normal support was `5.7897 N`. No hand/bottle contact occurred and lift was `0 m`. The evidence points to insufficient operational endpoint margin/tracking under the moving-fixture transient, not to bottle contact, friction, or a NaN/high-energy instability.

Across the dynamic attempts there were zero bottle qpos writes, no bottle weld/equality, no mocap carry, no artificial bottle force, no hidden support, and no new collision exclusions. The source joint range was unchanged. The free bottle remained table-supported; its recorded maximum penetration during settling was `0.696 mm` and remained below the experiment's `2 mm` diagnostic cap.

The saved MP4 was decoded with the run environment's imageio/FFmpeg plugin: 16 frames, 640×480, H.264, 20 fps, 0.8 s. It is a valid visual record of the incomplete approach and does not depict a physical lift.

## Reproduction and raw evidence

Run05 acceptance runner SHA256: `44ba4fba2cf5cea895b917b097e6ef8614bc33678006abaca08f0623bcc3220e`. Run12 isolated runner SHA256: `b67f1fd4ba1a6fd38680a269182ff7a084d4be14aa594345536c7887361388b6`.

Run05 command (set the three local path variables to the pinned source/helper and evidence root recorded in the external run identity):

```bash
PYTHONPATH="$PIN_PREFIX/cmeel.prefix/lib/python3.10/site-packages:$PIN_PREFIX:$MUJOCO_SITE" LD_LIBRARY_PATH="$PIN_PREFIX/cmeel.prefix/lib:$PIN_PREFIX/cmeel.prefix/lib64" MUJOCO_GL=egl python3 scripts/research/issue46_omnipicker_p0_acceptance.py --vendor-root "$X2_VENDOR_ROOT" --canonical-helper "$CANONICAL_HELPER" --output "$EVIDENCE_PARENT/omnipicker-p0-automation-20261011-run05"
```

Run12 command:

```bash
PYTHONPATH="$PIN_PREFIX/cmeel.prefix/lib/python3.10/site-packages:$PIN_PREFIX:$MUJOCO_SITE" LD_LIBRARY_PATH="$PIN_PREFIX/cmeel.prefix/lib:$PIN_PREFIX/cmeel.prefix/lib64" MUJOCO_GL=egl python3 scripts/research/issue46_omnipicker_p0_isolated.py --output "$EVIDENCE_PARENT/omnipicker-p0-automation-20261011-run12"
```

The exact machine-specific commands and runtime path bindings are retained in the external run identities and `REPRODUCE.md`. Raw evidence is in the external `run05`, `run11`, and `run12` folders; the run directories were not edited during closeout. The consolidated packet contains `FINAL_REPORT.md` and `P0_FINAL_RESULT.json`.

## Next step

Stop this run. Before any new contact test, review the right-claw operational OPEN target and the fixture approach transient as a distinct controller/initialization correction. Keep the official `[-1,0] rad` source limit unchanged and preserve this failed result. Full-X2 integration remains gated on a successful isolated physical lift.
