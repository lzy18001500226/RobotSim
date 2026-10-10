# Issue #46: Dual Robotiq Collision-Clear Adapter Study

## Result

**BLOCKED before adapter construction.** The bounded geometry work did not find a rigid right-side mount translation that clears the wrist, elbow, and hip over the retained jaw cycle. No bracket XML was produced, so no new mounted cycle, bottle corridor, or bottle pickup was run. This is a geometry result for the tested fixed-arm pose, not proof that all right-side adapters or arm poses are infeasible.

The starting repository head was `1eb9f2f1671a209c5873094293f161b926f1a31b`, on `research/issue46-dual-robotiq-recovery-20261010`. Repository changes are limited to the research geometry runner and its report/reproduction/result artifacts. Production robot files, PR #58, source limits, gripper joints, collision masks, and physical properties were not changed.

## Source and Runtime Identity

| Item | Identity |
|---|---|
| AgiBot source | `AgibotTech/agibot_x2_urdf`, `575cc6b988f976c23550e0db85aa1e5475d3652d` |
| Menagerie source | `google-deepmind/mujoco_menagerie`, `0059d4335f8156206f63a35662313385f7ad6d74` |
| Pinned `X2-Ultra.xml` | SHA-256 `20119fc61f43812daf2e72182a5c9e3b255f274e2e8da4c91c429f203093b292` |
| Pinned `X2-Ultra_omnipicker.urdf` | SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d` |
| Compiled baseline XML | SHA-256 `f99328cd16680aced60196a8ccd87ea1b875a021890b059e6d988f237e1b25ad` |
| Retained 1,700-step left gripper trace | SHA-256 `ceffbcb5fbab8b208b34f02acf6229b7bfc323c410e6a9879febde1dc96934ff` |
| MuJoCo | Python/native `3.3.6`; Python `3.10.12`; `MUJOCO_GL=egl` |
| Menagerie Robotiq XML | SHA-256 `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de` |
| Geometry runner | `scripts/research/issue46_x2_robotiq_standoff_adapter.py`, SHA-256 `5d166b6d503b67f6cc2d82c90dce4b0b43890ba4fc01847bb63afc622919c7e9` |

The exact generated input XML, trace, source runner snapshots, outputs, and reproduction commands are in the external evidence directory listed in `REPRODUCE.md`.

## Wrist Variant and Inertia Audit

The pinned OmniPicker URDF (`X2-Ultra_omnipicker.urdf`) assigns:

| Link | Visual mesh | Collision mesh | Mass | Inertial origin xyz (m) | Diagonal inertia (kg m^2) |
|---|---|---|---:|---|---|
| Left wrist roll | `left_wrist_roll_extend_link.stl` | `left_wrist_roll_extend_link.stl` | 0.107 kg | `[-0.00191378, -0.00010270, -0.03431117]` | `[0.00007581, 0.00014082, 0.00008760]` |
| Right wrist roll | `right_wrist_roll_extend_link.stl` | `right_wrist_roll_link.stl` | 0.107 kg | `[-0.00165168, 0.00009437, -0.03438670]` | `[0.00007557, 0.00014016, 0.00008717]` |

The pinned `X2-Ultra.xml` MuJoCo model instead uses `left_wrist_roll_link.stl` and `right_wrist_roll_link.stl` for its wrist meshes. Its compiled wrist inertials are left mass `0.338 kg`, inertial position `[0.003017, -0.001392, -0.086394]`, diagonal inertia `[0.000892625, 0.000750874, 0.000240898]`; right mass `0.337 kg`, position `[0.003263, 0.001281, -0.086468]`, diagonal inertia `[0.00089008, 0.000748845, 0.000240465]`.

The tested dual-gripper baseline combines the extended URDF visuals and left collision mesh with the standard right collision mesh, while retaining the X2 Ultra MJCF wrist inertials. This is the existing SIMULATION_ONLY assembly, not a vendor-qualified Robotiq wrist adapter. The asset difference is verified; whether it corresponds to different physical hardware envelopes is not established by CAD or measurement. The bracket study did not alter either mesh or either inertial set.

Mesh SHA-256 values: left extended `f379d912d341e6224ec16a62ec9a3c21621e43b6aae1bec0ba93e9a947d7507f`; right extended visual `94c4c7b32306504e74efa3934a473be54c601a8e61c68476b0ef78fc397657de`; right standard collision `f6ed0a2a01407c6f314b6b2c04e5df8914f60963f3b546622f6734f03d487c72`.

Source wrist-roll joint ranges are mirrored: left `[-1.509710, 0.724312] rad`, right `[-0.724312, 1.509710] rad`.

The retained source-mount transform is right wrist to Robotiq base `xyz=[0.00196, 0.00035, -0.062] m`, `rpy=[3.14, 0, 0]`. It reuses the pinned OmniPicker fixed-joint transform and is not a qualified Robotiq bracket.

## Geometry Tests

The runner maps the retained 1,700-step left finger trace to the right Robotiq joints while holding the right X2 arm at the diagnostic model's initial pose. For the clearance calculation it considers collision-enabled robot and right-gripper geoms, skips welded-body pairs and immediate parent-child mounting interfaces, and reports signed `mj_geomDistance`. The scalar direction profiles sample every tenth physics state; the full-cycle baseline and any accepted candidate are checked at every retained state. The 3D solve constrains the recorded open, close, and reopen critical frames and is not treated as a pass unless a subsequent full trace passes.

After removing a legacy candidate-emission path, the final runner passed `py_compile` and all five geometry cases were rerun into fresh validation directories. They reproduced the same blocking clearances; the raw final-runner outputs are preserved under `runner_validation_final_runner/`. These are geometry/FK evaluations, not dynamic gripper cycles.

| Geometry-derived test | Result |
|---|---|
| Source local `+Z` insertion axis | Baseline minimum `-28.949 mm` (right wrist roll / right spring link, reopen step 961). At `+50 mm`, minimum remains `-14.437 mm` (wrist / spring, open-hold step 11). At `+75 mm` and `+100 mm`, the nearest result is `0 mm`, not the required `+3 mm`; the limiting pairs change to elbow/wrist against gripper links. Rejected. |
| Preserved step-584 contact normal | The `+50 mm` profile worsens to `-34.832 mm` (wrist / left driver, close step 781); the 75 and 100 mm samples still have negative clearances. Rejected. |
| Full-cycle worst-pair witness direction | The step-959/961 wrist/spring witness normal was finite-difference checked: `-1 mm` gives `-29.949 mm`, `+1 mm` gives `-27.949 mm`. The `+25/+50/+75/+100 mm` profile remains between `-32.384` and `-33.454 mm` as the limiting pair moves from wrist/spring to hip/coupler. Rejected. |
| Minimum-norm 3D translation, 50 mm radius cap | SLSQP returned `success=false` (`Positive directional derivative for linesearch`); its point `[-36.997, 11.775, -50.000] mm` violated the cap (returned norm `63.304 mm`) and still had `-0.942 mm` at wrist/spring around close step 584. This is an invalid solver result, not a candidate. |
| Minimum-norm 3D translation, 75 mm radius cap | SLSQP returned `success=false` (`Inequality constraints incompatible`) at `[-61.133, 6.553, -1.587] mm`, norm `61.504 mm`. Critical-frame minima include `0 mm` wrist/driver clearances and only `+2.628 mm` on the remaining wrist/spring pair, below the `+3 mm` requirement. No candidate was emitted. |

The 75 mm run is the single evidence-based bound extension after the 50 mm solution hit its cap. There was no arbitrary xyz, orientation, or joint-pose sweep; scalar profiles were limited to 0/25/50/75/100 mm along three geometry-derived directions. Since the geometry gate did not pass, the runner did not create an adapter model or test bracket collision geometry.

## Existing Dynamic Evidence and Gates

The copied videos and traces under `prior_mounted_dynamics/` are retained from the earlier adapter study and are not results for a new bracket:

- Left-only OPEN/CLOSE/OPEN: prior diagnostic PASS, 1,700 steps, minimum pad gap `8.874 mm`.
- Original right source mount: FAIL at step 584 (`0.584 s`), wrist-roll/follower penetration `0.085261 mm`, normal force `15.369 N`, stopped with `43.915 mm` pad gap.
- Earlier 180-degree local-Y candidate: FAIL; collision with elbow/wrist-yaw assembly, right gap reached only `88.298 mm` from `93.400 mm`.
- The earlier simultaneous cycle is not evidence for a new adapter; its right side has the same failure as its candidate model.

No new right-only or simultaneous cycle was eligible because no collision-clear adapter was produced. The earlier source-mount bottle corridor remains a static FAIL at `-15.360 mm` minimum wrist/bottle distance and was not rerun. Bottle grasp, lift, transfer, release, and placement were not run. The previously mentioned 50 mm pickup is not verified by this work and is not claimed.

SOURCE_FAITHFUL Robotiq coupler dynamics remain FAIL. Existing SIMULATION_ONLY 0.001 rad coupler-limit activation margin and the already authorized exact head/torso exception remain diagnostic-only; this work did not modify them.

## Decision

No mechanically coherent rigid translation adapter was demonstrated for the current right-arm initial pose. Keep the existing left-side PASS and right-side failure evidence. The next engineering input needed is a qualified right-wrist/Robotiq mounting envelope or an authorized materially different right-arm work pose from which a bracket can be designed and checked. Do not treat the source visual mesh as proof of the correct collision envelope, and do not promote this assembly as hardware-faithful.

No new adapter overview, front/side bracket renders, or adapter OPEN/CLOSE video exist because the bracket failed the geometry gate before construction. The copied collision close-ups and prior cycle videos are labeled as earlier source-mount/failed-adapter evidence.
