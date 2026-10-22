# Issue #46 — OmniPicker D-Point Alignment Review

**Status:** `READY_FOR_GITHUB_ENGINEERING_REVIEW`

**Mechanical model:** `UNVERIFIED_EXPERIMENTAL`
**MuJoCo dynamics:** `NOT RUN`

This review reuses the pinned X2 v1.4.0 URDF, the approved candidate-B proposal, and the existing FreeCAD/STEP measurement outputs. It does not edit the vendor URDF/STLs, the local candidate MJCF, the active physical model, or any prior review evidence. The D-site and root-coordinate changes below are proposals only.

## Source Evidence

| Input | Identity |
|---|---|
| X2 OmniPicker URDF | `AgibotTech/agibot_x2_urdf@575cc6b988f976c23550e0db85aa1e5475d3652d`; SHA-256 `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d` |
| Original STEP assembly | SHA-256 `5d61f8d7e94fa3c789ddd497e4574aa2a508e24aed447d841585dfdbb8e8c20b`; local-only, upstream URL/license for the STEP archive remain unverified |
| Placement-aware FreeCAD source | SHA-256 `c5a1bca530eb52649544d0eaa87b1bf8881858040002e9631352e18b0f775c1f` |
| CAD-to-URDF frame registration | `frame_registration.json`, SHA-256 `dcaed08ba724b64cfa5a85f0967bd1609c83d41b6175d682ecd4605e4ff25f3e` |
| Source pivot correspondence | `joint_correspondence.csv`, SHA-256 `3802fd6e325c1abbd24eef6073799e39d0e109736eb90448975d66f27826cbb9` |
| Source candidate motion | SHA-256 `893604ac0a366d6e3997e661e7e83cd708549ca66c22e1cab890425d6d7809ca`; compact 66-row approved-window extract SHA-256 `7b544d5d5ebeb18e32a75ab6b9cd31ac24e8f63d03a83fc238988b941f4684f3` |
| Source display-scene data | SHA-256 `07ceab9431c97da7e7b53fbce90697100deea621518db12c6acfd7114f48a92d`; 0.5 mm display tessellation, local-only |
| Approved candidate-B package | `MODEL_REVIEW.md` SHA-256 `acd3085e279fe50a43f30a0f4b519793a8f3b7fc8dfe73fcf0a56cbc75e6ab6a`; mechanical proposal SHA-256 `f8d2ccef825c661c1c4b2ecf766e8525253a073d0fe9b5a95ea3aa304527c3c5`; MJCF proposal SHA-256 `c4b55b9f7353ebc5cb448474209684794cc6c1e5e83d3f0f5c5e9f8e85a9152a` |
| Previous official-URDF static preflight | Candidate MJCF SHA-256 `8f2f39d2a8fc3e8fbc9ec729a11b16a10b462fe3a525f4db9bd92745d6cb1b6f`; preflight JSON SHA-256 `ad8f520501402eecec2caadcf069ceb706450f994f71085c780cdb320ec0d6f9`; source-reference contact check SHA-256 `dbdc97a1309366c194cc7a9586eb829bc211d457131299a3275cc4f45629b088` |

The original STEP and scene JSON are not copied into this PR. The compact motion extract is sufficient to rerun the listed transform and candidate-relation calculations.

## D Anchors and Transform Chain

Each side has two MJCF sites intended to represent one shared physical D pin center:

- Jaw-side site: the meshless `L_hand_narrow4_Link` / `L_hand_wide4_Link` source frame, attached to `L_hand_narrow3_Link` / `L_hand_wide3_Link` at the official 10.4 mm terminal offset.
- Loop-side site: `candidate_narrow_D_loop` / `candidate_wide_D_loop` on the corresponding official loop link. The current local sites come from the approved MJCF proposal.

These are two frame representations of the same CAD C-D pin center, not two independently measured pins. Existing STEP evidence identifies D4 pin `230000166301` near the off-loop axis, but its bearing/slot/host ownership and constraint type are unresolved; this report does not promote that part into a joint.

The source CAD D coordinates are the sample-0 D pivots in `joint_correspondence.csv`. Each is transformed to the official URDF base by:

```text
p_base_mm = R_cad_to_base * [D_x_mm, 0, D_z_mm] + t_cad_to_base_mm
```

`R_cad_to_base` is the recorded 3×3 registration matrix and `t_cad_to_base_mm = [0, -24.1601592428, -132.5856581623]`. The official body transforms then use the pinned URDF origins/RPYs, B/C offsets, root axes `(0,0,-1)`, source root phases, and B/C/E relative coordinates. The compact script implements this chain directly; it does not import the 94 STEP solids or alter their geometry.

```text
R_cad_to_base = [[ 0.000000000, -1.000000000, 0.000000000],
                 [ 0.981601854,  0.000000000, 0.190939257],
                 [-0.190939256,  0.000000000, 0.981601854]]
t_cad_to_base_mm = [0.000000000, -24.1601592428, -132.5856581623]
```

For each branch the evaluated transforms are explicitly:

```text
T_A = T(origin_A_xyz, origin_A_rpy) · R(axis=-Z, q_URDF_A)
T_B = T_A · T(origin_B_xyz, origin_B_rpy) · R(axis=-Z, q_URDF_B)
T_C = T_B · T(origin_C_xyz, origin_C_rpy) · R(axis=-Z, q_URDF_C)
T_E = T(origin_E_xyz, origin_E_rpy) · R(axis=-Z, q_URDF_E)
p_jaw_D  = T_C · p_jaw_D_local
p_loop_D = T_E · p_loop_D_local
```

At the registered source sample, `q_URDF_A` is the absolute phase `-0.13412598906686654 rad` narrow and `+0.13412598906686654 rad` wide; the candidate-relative B/C/E coordinates are zero. The separately fitted wide CAD phase is `+0.134124879357947 rad`, differing from the mimic-preserving value by `1.10971e-6 rad`. During the sweep, candidate relative angles are converted to URDF coordinates with the measured sign map before applying these transforms.

In the saved derived MJCF convention, each root body carries the source-phase rotation and each root hinge `ref` is that same absolute source phase; `mj_resetData` starts at `qpos=ref`, so the root's physical pose is the registered source pose. B/C/E hinge `ref` values are zero at the candidate source sample. The proposed command conversion applies to the relative coordinate `qpos-ref`; it does not change the URDF axes or the root reference pose.

The resulting A/B/C/D/E map is:

| Side/pivot | Official URDF joint/frame | Parent → child | Official origin `(xyz m; rpy rad)` | CAD pivot `(x,z mm)` | Fit residual | Coordinate status |
|---|---|---|---|---:|---:|---|
| Narrow A | `left_claw_joint` | base → narrow1 | `(0,-0.0195,0.0565); (-2.9951,-1.5708,-0.15964)` | `(-31.529496,186.496857)` | `0.000221 mm` | Official revolute, range `[-1,0]`; source phase `-0.134125989 rad` |
| Narrow B | `L_hand_narrow2_joint` | narrow1 → narrow2 | `(0.030852,0.018551,0); (0,0,0)` | `(-58.990804,209.775100)` | `0.296059 mm` | Fixed upstream; candidate hinge axis `-Z`, CAD/URDF coordinate signs invert |
| Narrow C | `L_hand_narrow3_joint` | narrow2 → narrow3 | `(0.018118,-0.015740,0); (0,0,0)` | `(-49.888861,231.982185)` | `0.122935 mm` | Fixed upstream; candidate hinge axis `-Z`, CAD/URDF coordinate signs invert |
| Narrow D | `L_hand_narrow4_Link__source_frame` | narrow3 → meshless frame | `(0,-0.0104,0); (0,0,0)` | `(-40.248617,235.973296)` | `0.691435 mm` | Site/anchor only; not a new DOF |
| Narrow E | `L_hand_narrow_loop_joint` | base → narrow loop | `(0,-0.021633,0.073870); (-2.9951,-1.5708,-0.15964)` | `(-36.939502,203.139628)` | `0.000362 mm` | Fixed upstream; candidate hinge axis `-Z`, CAD/URDF coordinate signs invert |
| Wide A | `L_hand_wide1_joint` | base → wide1 | `(0,0.0195,0.0565); (-2.9951,-1.5708,-0.15964)` | `(6.752976,193.943488)` | `0.000221 mm` | Official revolute, range `[0,1]`, mimic `-1`; independent phase fit `+0.134124879 rad` |
| Wide B | `L_hand_wide2_joint` | wide1 → wide2 | `(0.030852,-0.018551,0); (0,0,0)` | `(23.486015,225.818328)` | `0.295855 mm` | Fixed upstream; candidate hinge axis `-Z`, CAD/URDF coordinate signs invert |
| Wide C | `L_hand_wide3_joint` | wide2 → wide3 | `(0.018118,0.015740,0); (0,0,0)` | `(6.723361,242.994283)` | `0.122585 mm` | Fixed upstream; candidate hinge axis `-Z`, CAD/URDF coordinate signs invert |
| Wide D | `L_hand_wide4_Link__source_frame` | wide3 → meshless frame | `(0,0.0104,0); (0,0,0)` | `(-3.710038,243.080709)` | `0.691442 mm` | Site/anchor only; not a new DOF |
| Wide E | `L_hand_wide_loop_joint` | base → wide loop | `(0,0.021633,0.073870); (-2.9951,-1.5708,-0.15964)` | `(5.529921,211.400697)` | `0.000362 mm` | Fixed upstream; candidate hinge axis `-Z`, CAD/URDF coordinate signs invert |

The candidate uses the mimic-preserving wide source phase `+0.134125989 rad`, `1.10971e-6 rad` above its independent geometric fit. B/C/E candidate generalized coordinates are zero at the source sample; the conversion from candidate CAD coordinates to official URDF joint coordinates is sign-inverted for each measured `-Z` hinge.

The source sample relation `q_root + q_B - q_E = 0` is the existing equal-rate 1:1 candidate hypothesis (101 samples); it is not an AgiBot-confirmed transmission rule. Its near-zero candidate centerline residual proves only that the saved candidate coordinates close their mathematical source loop.

| Side | CAD D in source CAD frame (mm) | CAD D transformed to URDF base (mm) | Current URDF jaw D world (mm) | Current URDF loop D world (mm) |
|---|---:|---:|---:|---:|
| Narrow | `[-40.248616976, 0, 235.973296455]` | `[0, -18.611710398, 106.731208196]` | `[0.012603988, -18.527689186, 106.045012603]` | `[0.099594748, -11.911627286, 105.469995931]` |
| Wide | `[-3.710038103, 0, 243.080708765]` | `[0, 18.611710399, 106.731208196]` | `[-0.012963338, 18.527747045, 106.045013646]` | `[-0.099949927, 11.911684474, 105.470004068]` |

At source phase, the current official jaw-local D sites are `[0, -10.4, 0]` mm narrow and `[0, +10.4, 0]` mm wide. The current loop-local D sites are `[31.600001246, -9.721824969, -0.028105602]` mm narrow and `[31.599998755, 9.721806342, 0.027979362]` mm wide.

## Reproduced Residuals

| Side | Jaw-minus-loop vector in MJCF base (mm) | Euclidean separation (mm) | Max absolute connect row (mm) | Saved MuJoCo max `efc_pos` (mm) |
|---|---:|---:|---:|---:|
| Narrow | `[-0.086991, -6.616062, 0.575017]` | `6.641573` | `6.616062` | `6.616062` |
| Wide | `[0.086987, 6.616063, 0.575010]` | `6.641573` | `6.616063` | `6.616063` |

The 6.6416 mm value is the vector's Euclidean norm. MuJoCo's 3-row `connect` constraint was summarized as the largest absolute row component, approximately 6.6161 mm. They describe the same D anchors; the 0.0255 mm difference is a norm-versus-component metric difference.

The current jaw D point is already about `0.6914 mm` from the CAD-transformed D point. The current loop D point is about `6.8185 mm` from it. Thus the approved loop-site transform is the dominant source of the original separation.

## Why the Old Phase Result Was Misleading

The previous `±0.202251 rad` calculation allowed only one scalar rotation of the current loop D vector about the fixed E axis. It held fixed the current official jaw D site, the E pivot, link lengths, root pose, and all other joints. That best aligns the current loop point to the *current URDF jaw site* and leaves `0.73716 mm` because their E-relative radii differ by that amount. It is not the CAD-to-URDF source-frame transform and is not a justified mechanical phase change.

The source CAD point gives a different diagnostic E-axis angle (`-0.206787984 rad` narrow, `+0.206786395 rad` wide) and about `0.063 mm` residual for that one-variable fit. The small residual comes from the measured point/radius fit; no manufacturer tolerance was found, so it is not classified as acceptable. The exact `0.737 mm` old-target residual is the difference between the current E-relative radii: narrow loop `33.061681 mm` versus current jaw `32.324519 mm` (wide `33.061673 mm` versus `32.324515 mm`).

The minimal source-supported coordinate correction is to express the *same CAD D point* in both official link-local frames. It requires these candidate MJCF site coordinates:

| Site | Current local position (mm) | CAD-transformed local position (mm) |
|---|---:|---:|
| Narrow jaw | `[0, -10.4, 0]` | `[0.691289490, -10.408336990, -0.011495781]` |
| Narrow loop | `[31.600001246, -9.721824969, -0.028105602]` | `[32.861209960, -3.021010867, -0.039601384]` |
| Wide jaw | `[0, 10.4, 0]` | `[0.691281259, 10.408390232, 0.011860848]` |
| Wide loop | `[31.599998755, 9.721806342, 0.027979362]` | `[32.861206432, 3.021046105, 0.039840210]` |

This is a point-coordinate re-expression through the recorded CAD/base and official body transforms. It changes no joint origin, axis, limit, root mimic, collision geometry, or transmission equation. The local coordinates round-trip to the shared CAD D point at source pose and keep the two D sites within `0.00061 mm` across the 33 candidate samples per side in the approved `|q_candidate| ≤ 0.32 rad` window. This remains a kinematic mapping check, not a manufacturer validation or dynamics result. The exact proposed MJCF site and command-map diff is in `issue46_dpoint_coordinate_proposal.diff`; it is **not applied**.

## Root Sign and Limit Audit

The source-to-URDF root map is also sign-inverted relative to the CAD candidate coordinate. At candidate narrow `q=-0.01 rad`, the official root pose inferred from the A→B pivot vector is `q_URDF=-0.124125989 rad` (source phase `-0.134125989` plus `0.01`), not `-0.144125989 rad`. The direct `phase + q_candidate` mapping causes `0.777 mm` B-pivot error at that sample; the sign-converted mapping reduces it to `0.292 mm`, consistent with the existing B-axis fit residual. At candidate `q=-0.32`, the direct mapping misses B by `22.649 mm`; the sign-converted mapping misses it by `0.165 mm`.

The measured coordinate mapping is:

```text
q_URDF_abs = q_source_phase - q_candidate_CAD_rel
q_URDF_B/C/E = -q_candidate_CAD_B/C/E
```

All official axes remain `(0,0,-1)`. Because all candidate coordinates invert together, the candidate scalar relation retains its zero form: `q_root_URDF_rel + q_B_URDF - q_E_URDF = 0`. The source candidate data satisfies its CAD-coordinate relation to `7.8e-16 rad`; pivot-inverted official coordinates satisfy the relation within `5.7e-5 rad` over the checked samples. The measured B/C/E passive joint-coordinate signs also invert: maximum sign-map residuals are `8.2e-5 rad` for B, `1.97e-4 rad` for C, and `3.4e-5 rad` for E. The root, B, C, and E axes themselves remain unchanged.

This sign correction exposes a hard limit conflict. Candidate narrow `q=-0.32` maps to official `left_claw_joint=+0.185874 rad`, above its unchanged upper limit `0`; the wide side maps to `-0.185874 rad`, below its unchanged lower limit `0`. Only candidate narrow `q ∈ [-0.134125989, 0]` and wide `q ∈ [0, 0.134125989]` fit the official root ranges. The previously used `phase + q_candidate` runner mapping must not be used. A source-direction correction cannot be implemented by reversing the official root axis.

| Candidate CAD q | Official narrow q with measured sign map | Original limit result | A→B error with old direct-add map | A→B error with sign-converted map |
|---:|---:|---|---:|---:|
| `0` | `-0.134125989` | inside | `0.296 mm` source-fit baseline | `0.296 mm` |
| `-0.01` | `-0.124125989` | inside | `0.777 mm` | `0.292 mm` |
| `-0.13` | `-0.004125989` | inside | `9.337 mm` | `0.246 mm` |
| `-0.14` | `+0.005874011` | outside | `10.050 mm` | `0.242 mm` |
| `-0.32` | `+0.185874011` | outside | `22.649 mm` | `0.165 mm` |

The passive pivot inversion yields `q_URDF_B/C/E ≈ -q_CAD_B/C/E`; the maximum residuals from this sign map across the 33 source samples are `8.2e-5`, `1.97e-4`, and `3.4e-5 rad`. The official-coordinate candidate rate equation remains zero to `5.7e-5 rad` in the pivot inversion. The source-candidate `q_root+q_B-q_E` equation is itself a separate experimental hypothesis, not vendor transmission evidence.

## Opening Direction and Interference

Using the existing 0.5 mm display triangulation of the original source jaw B-reps, the closest sampled vertices across the main narrow/wide jaw bodies move closer as candidate `q` decreases: `26.433 mm` at `q=0`, `3.034 mm` at `q=-0.15`, `0.108 mm` at `q=-0.19`, and `0.017 mm` at `q=-0.20`. At `q=-0.25` through `-0.32` the nearest vertices separate again, but that occurs after over-folding; it is not a valid opening stroke. This is a source-mesh surface-sampling result, not a calibrated gripping-face aperture or a tolerance measurement. The previously reported jaw-tip landmark distance is not used as aperture evidence.

| Candidate CAD `q_narrow` (rad) | Nearest sampled jaw-body vertices (mm) |
|---:|---:|
| `0.00` | `26.433` |
| `-0.05` | `18.953` |
| `-0.10` | `11.130` |
| `-0.15` | `3.034` |
| `-0.19` | `0.108` |
| `-0.20` | `0.017` |
| `-0.25` | `0.135` |
| `-0.30` | `0.188` |
| `-0.32` | `0.403` |

These values are nearest-vertex distances for the source main jaw bodies (`Part__Feature054/056` and `Part__Feature061/062`) after the recorded rigid C-D pose transform. They show approach and crossing, not a valid contact aperture. The local sampling is `0.5 mm`; the separate positive B-rep common volumes below establish actual overlap for the tested outer/middle pairs at `q=-0.19`.

At the first single-stroke sample (`q=-0.19`), the existing exact FreeCAD B-rep record reports positive common volume for:

| Source STEP pair | Exact common volume |
|---|---:|
| `Part__Feature005` × `Part__Feature008` (wide outer/middle) | `21.369667 mm³` |
| `Part__Feature021` × `Part__Feature024` (narrow outer/middle) | `21.369667 mm³` |

The saved B-rep sweep reuses exact results for these critical pairs at selected samples; other AABB candidates are explicitly unresolved. This is genuine candidate STEP solid intersection, not a MuJoCo hull artifact. The official STL/MuJoCo source-pose hull contacts are a separate result and are also present in the fixed-descendant reference:

| Official STL hull pair at the mapped source pose | MuJoCo hull penetration proxy |
|---|---:|
| `L_hand_narrow1_Link` × `L_hand_narrow_loop_Link` | `6.111 mm` |
| `L_hand_narrow3_Link` × `L_hand_narrow_loop_Link` | `4.418 mm` |
| `L_hand_wide1_Link` × `L_hand_wide_loop_Link` | `6.134 mm` |
| `L_hand_wide3_Link` × `L_hand_wide_loop_Link` | `4.414 mm` |

Their exact official STL surface status remains `UNVERIFIED`; no triangle-intersection backend is installed in the pinned MuJoCo environment, and collision exclusions or mesh replacements were not used.

The q-range allowed by the official root limits ends at candidate `q_CAD=-0.134126 rad`; the nearest source jaw-body mesh vertices there are about `6.3 mm` apart. The exact source B-rep overlap at `q_CAD=-0.19` is beyond that limit under the measured sign map. It establishes an invalid candidate over-fold, but does not prove a legal official-root pose is already colliding. The four hull contacts above remain a separate exact-surface blocker.

## Actuator and Equality Initialization

- Official wide-root mimic and the two experimental candidate rate equalities initialize at zero residual in the saved preflight.
- The existing D `connect` equality does not initialize with the old sites; the source-mapped sites above are the proposed correction.
- The SIMULATION_ONLY position servo remains `kp=0.5`, `kv=0.05`, force cap `±0.5 Nm`. A dynamics runner must set its initial target to the source root angle before the first `mj_step`, then apply `phase - q_candidate_rel`; no per-step qpos forcing is allowed.
- No dynamic rollout was run for this review. Bottle/grasp physics and video remain `NOT RUN`.

## Decision and Next Step

The D residual is explained by a demonstrable anchor-frame mapping error, and a minimal site-coordinate proposal is prepared. The official-root sign mapping is also demonstrably reversed from the old runner mapping, but the full candidate closing stroke conflicts with the unchanged manufacturer root limits. The candidate STEP path additionally has exact outer/middle B-rep intersections near closure.

The smallest defensible next step is human review of the proposed D-site coordinate transform and the root-coordinate/limit conflict, followed by confirming the source collision disposition for the four official convex-hull contact pairs. The signed root-coordinate map makes the planned `q_CAD=-0.32` endpoint illegal under the unchanged URDF limit; candidate closing/over-fold at `q_CAD=-0.19` is beyond that limit and has exact B-rep outer/middle overlap. Do not start the bounded `mj_step` rollout or claim a valid opening/closing path until a source-supported candidate stroke exists entirely within the official root limits and the B-rep interference gate is resolved.

Reproduction:

```bash
python3 scripts/research/issue46_dpoint_alignment.py
python3 scripts/research/issue46_dpoint_alignment.py --scene-json /path/to/scene_data_single_stroke.json
```

The first command reproduces D transforms, residuals, sign mapping, root-limit checks, and the 66 candidate rows. The optional second command verifies the scene hash and reproduces the jaw-mesh samples and saved exact B-rep overlap records.

The optional scene analysis requires NumPy and SciPy. It was verified from Windows Anaconda Python 2.1.3 / SciPy 1.15.3 through the WSL UNC paths; the Ubuntu system `python3` in this environment lacks NumPy, so no packages were installed. The standard-library-only command works with the Ubuntu system Python.
