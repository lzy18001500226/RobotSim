# Issue #46: OmniPicker Contact Geometry and Lateral Escape

**Result: FAIL — no stable load-bearing free-cylinder grasp.** The measured pad normals drive a substantial common lateral force, and the geometry-derived correction that would improve their opposition is blocked by wrist collision before approach. No lift was attempted in this investigation.

## Scope and provenance

This report covers the contact-frame reconstruction, the `SIMULATION_ONLY_FIXED_OBJECT_DIAGNOSTIC`, and one zero-step geometry candidate. It does not establish source-faithful actuator behavior, a full-X2 grasp, or a bottle lift. The fixed-object diagnostic removes the cylinder free joint before compilation; its forces are not evidence of free-object support.

- RobotSim branch: `research/issue46-omnipicker-p0-automation-20261011`
- Starting local commit: `cf78f86d9498dd16d6d23877dad829fbb2049dc0`
- AgiBot source: [`AgibotTech/agibot_x2_urdf`](https://github.com/AgibotTech/agibot_x2_urdf), commit `575cc6b988f976c23550e0db85aa1e5475d3652d`
- Source URDF SHA-256: `35d228c9e7da2d38846e91940365becf461e83bcc44062db659bdfd3e23c0d4d`
- Narrow/wide finger collision STL SHA-256: `0fb55604e97d18c9545567e082e071746a968f624a0aa31ece3a0cc4fc1217b0` / `a06d0c7e16ddd06e28a8e75c5a27c2e5fa025c00b79d973983f389cd08a380fb`
- Runtime: WSL2 Ubuntu 22.04, Python 3.10.12, NumPy 1.26.4, MuJoCo Python/native 3.3.6, `dt=0.002 s`, Euler integrator, solver 2 with 100 iterations, gravity `[0,0,-9.81] m/s^2`.
- Native MuJoCo library SHA-256: `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495`
- Original fixed-object run runner SHA-256: `07c052de4f1195f94d339c41cdea6616340ed987d45327eade0198b4063adb21` (recorded in run identity). The current research runner has a reporting-only correction and SHA-256 `b511112b1c5d65470536846f7368db2fafaeaf0055b2cbd330f945c7793d453a`.

## Phase 1 — contact frame and source geometry

The saved free-cylinder trace reports `geom1=cylinder`, `geom2=finger`. `mj_contactForce` is interpreted in the contact frame and transformed to world; the force sign is chosen as the force acting **on the cylinder**. The opposing force-normal angle remains `124.023472 deg`, so the reported geometry is not a contact-frame sign artifact.

| Contact family | Normal force | Force on cylinder, world (N) | Source STL witness distance |
|---|---:|---|---:|
| `R_hand_narrow3_Link` | 1.48309 N | `[+1.37412, -0.57390, +0.01496]` | 0.000735 mm |
| `R_hand_wide3_Link` | 1.42136 N | `[-1.31385, -0.55723, -0.00645]` | 0.000890 mm |

The net pad force is `[+0.06027, -1.13113, +0.00851] N`: lateral XY magnitude `1.13274 N`, vertical component `0.00851 N`. Both contact forces have a negative world-Y component. The normals oppose mostly in X, but their shared Y component pushes the free cylinder sideways. The reconstructed finger witnesses are within 1 micrometre of the pinned source STL surfaces. Contact friction is unchanged at effective `mu=1.4`.

The unchanged bottle is a 70 mm diameter, 244.5 mm tall, 0.570 kg cylinder. Its weight is 5.592 N. The ideal symmetric two-pad vertical-friction screen is `mg/(2 mu)=1.99704 N` per pad; the previously defined 1.25 margin target is `2.49629 N` per pad. These are screening calculations, not observed support forces. The free-cylinder reference reached 5.003 mm relative slip; maximum lift remained 0 mm and preload did not reach its target.

## Phase 2 — fixed-object force-generation diagnostic

The diagnostic preserved the source finger collision meshes, controller gains, effort caps, joint limits, contact properties, and all 11 historical SIMULATION_ONLY collision exclusions; no new exclusions were added. The cylinder was a static body with no joint (`neq=0`): no weld, equality, mocap, attachment, or active-rollout object-state write was used. This deliberately prevents lateral escape and is not a grasp test.

The OPEN/approach/CLOSE/HOLD stages reached bilateral contact; the one-second HOLD had bilateral contact on 100% of steps, averaging 0.369 N narrow / 0.383 N wide. During preload, the run stopped at step 3982 (`t=7.964 s`) on the existing 20 mrad jaw tracking guard: follower error `-0.0200202 rad`. It had no source-limit violation, non-finite state, unexpected robot contact, or effort saturation. At the stop sample, requested/applied torques were 0.178704 Nm driver and -0.198429 Nm follower, below unchanged caps 0.224987 / 0.225078 Nm. Maximum penetration was 1.121 mm, below the 2 mm limit.

The raw result's original 0.20 s force-window summary included the final guard-failing sample. The runner now updates this summary only after that sample passes all safety checks. Recalculation from the untouched raw trace over 100 consecutive passing steps (3882–3981, `t=7.764–7.962 s`) gives:

| Pad | Minimum force in passing 0.20 s window | Mean force | Maximum force in that window |
|---|---:|---:|---:|
| Narrow | 1.36377 N | 1.60643 N | 1.83053 N |
| Wide | 1.44529 N | 1.73234 N | 2.01479 N |

Thus the diagnostic briefly approached the ideal screen on one side, but did not maintain even the ideal per-pad screen for the full window, and remained below the 2.49629 N margin target. At the failing step the instantaneous forces were 1.83522 / 2.02015 N; the narrow pad remained below the ideal screen, and the sample is disqualified by the tracking guard. These stop-sample forces are not a safe stable preload.

## Phase 3 — one geometry-derived candidate

The candidate translated the cylinder `+0.0162941 m` along world Y, derived from the mean Y coordinate of the two source-surface contact reconstructions. It was intended to move the same pad patches toward more opposed radial contact. Zero-step preflight evaluated the OPEN approach and 1,401 closure apertures; no physics was run.

- Bottle footprint and table support: pass; minimum table clearance `+75.489 mm`.
- OPEN approach and non-pad clearance: fail; minimum non-pad distance `-9.360 mm`, with active bottle contact by `right_wrist_roll_link` beginning at 10 mm of approach slide.
- Closure clearance: fail because of the same wrist/body interference.

No second candidate was run: this candidate failed before closure with a concrete non-gripping collision, and the available contact trace did not support another independent source-geometry correction within this bounded fixture study. This is not a claim of global OmniPicker infeasibility.

## Outcome and limits

- Contact frame / source mesh reconstruction: **PASS**.
- Fixed-object force-generation diagnostic: **FAIL** at `JAW_TRACKING_ERROR_20MM`; no stable two-pad preload met the screening target.
- Geometry candidate 1: **FAIL** at OPEN approach due to wrist-roll collision.
- Free-cylinder load-bearing grasp: **FAIL**; lateral slip reached 5.003 mm and load transfer was not sustained.
- 1 mm / 5 mm / 50 mm physical lift: **NOT RUN** in this task; measured lift in the prior free-cylinder run was 0 mm.
- Full X2 integration / hardware acceptance: **NOT RUN**.

The evidence points to contact orientation and lateral escape as the primary free-body failure, with limited safe preload as a contributing constraint. It does not justify more gain tuning, friction changes, new exclusions, or source-mesh modification. Any next experiment should first change the gripper-to-cylinder approach configuration enough to obtain opposing force directions while clearing the wrist; this investigation stops here.

The ad-hoc runner disposition is `KEEP-AS-TOOL` in `scripts/research/`: it reproduces the isolated diagnostic, and its stable-window metric now excludes safety-failing samples. It remains research-only and is not a production controller.

## Evidence

The review packet is at `/mnt/c/Users/HP/Desktop/Robot/reviews/issue-46-x2-single-hand/omnipicker-p0-contact-geometry-20261011/`. It contains byte-for-byte copies of the selected baseline trace/model/video/images, the candidate trace/images, and the earlier free-cylinder reference result/trace/video. The source run folders and all failed attempts remain preserved at their original locations. See `REPORT.md`, `result.json`, and `REPRODUCE.md` in the packet.

Remote publication was not attempted. The user explicitly requires local-only work until an authorized secure push method is available; PR #58 was not changed.
