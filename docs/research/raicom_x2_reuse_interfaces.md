# RAICOM 2026 X2: Reuse and Interface Boundaries

Issue #77 · Static source synthesis · 2026-10-19

## Decision

No inspected checkout provides a verified robot-startup-to-placement flow. AgiBot-Community is the best source for the ROS/perception/IK integration shape, but its packaged MuJoCo model has no articulated claw DOF or actuator. AimDK hand apertures are normalized left/right commands; the URDF claw joints are radian coordinates with a mimic relation, and no mapping between them is defined. Do not treat the stock simulation as capable of closing the hand or validating a grasp until a RobotSim-owned adapter/overlay is independently validated.

Placement/release is absent from all five task implementations. AgiBot-Community also contains a 120-second forced-success grasp-status timer. Its interfaces are reusable references, not acceptance evidence.

## Reusable Source Matrix

All repository snapshots below were clean detached HEADs during the source-only audit. “Verified” means statically confirmed in tracked code/docs; no runtime behavior was tested.

| Fixed source | Concrete entrypoints and covered stages | I/O and implementation evidence | Gaps, mocks, geometry, and reuse verdict |
|---|---|---|---|
| [AgiBot-Community/raicom2026-agibot-X2](https://github.com/AgiBot-Community/raicom2026-agibot-X2/tree/a97761820b398b6cab12b632ce9f313c394eeb7e) · a97761820b398b6cab12b632ce9f313c394eeb7e | Coordinator plus race_task navigation, voice_asr, interaction_agent, vision_node, x2_grasp_auto_trigger, x2_grasp_executor_server and x2_grasp_worker. Documented order starts infrastructure first and coordinator last. | ASR text → JSON intent → navigation goal/status → vision command → object_name + base_link target_point → IK worker → arm/hand commands. RGB-D deprojection and TF code are present. | Best integration reference, not directly executable grasp in bundled x2.xml: no claw DOFs/actuator. Hand/simulator mapping is unspecified. W_COLOR=0 makes color irrelevant to target score. A 120-second timer can publish grasp success independently of the worker. Worker lifts and holds; it does not place/release. Reuse interfaces only, fail closed on the timer, and validate a hand adapter before grasp. |
| [fofarup/Raicom2026-X2-Sim](https://github.com/fofarup/Raicom2026-X2-Sim/tree/ddbd3fd926846160424ff7fb74daba42b9d2ff5f) · ddbd3fd926846160424ff7fb74daba42b9d2ff5f | Single competition_node sequence: prepare → task1 → task2 → task3. Shortest monolithic orchestration; includes custom navigation, interaction, Pinocchio IK, and lift/hold code. | In simulation, speech/selection use keyboard or CLI and the digit/color task reads a local image. Service needs map to fixed world coordinates; grasp point-cloud objects are interpreted by fixed point indices. | No dynamic visual object selection or placement. Simulator overlay builds box fingers, slider joints, symmetry equality, custom LiDAR, and stand-in objects. Required 535,974,486-byte vendor archive is an LFS pointer in the checkout. Keep as orchestration/math reference, not canonical hand or runtime. |
| [Link-U-OS/link_u_os_competition](https://github.com/Link-U-OS/link_u_os_competition/tree/c3ffe0d4980e6b7193170cc4f7f29aea680294cc) · c3ffe0d4980e6b7193170cc4f7f29aea680294cc | Docker/setup manual and basic mode examples. | Names the MC, MuJoCo simulator, and AimDK message packages and gives their startup order. | No RAICOM navigation/perception/IK/grasp pipeline. Those runtime archives are external to this checkout. Use as setup/interface context only. |
| [causiniris/raicom2026_x2_autonomy](https://github.com/causiniris/raicom2026_x2_autonomy/tree/22fc981de7c0afffd2cb42fa0b80275bca05e327) · 22fc981de7c0afffd2cb42fa0b80275bca05e327 | Docker run_all starts sim, MC, and one autonomy process; C++ flow waits for reset/stand, then locomotion and zone-1 navigation. | Odometry/IMU feedback and filtered-pose velocity control. | Navigation-only. No interaction, perception, target selection, IK, grasp, or placement. Runbook target differs from source default; STEP3_RESULT is emitted after goal start, not verified arrival. Do not run run_all after Local03 standing because it starts its own sim/MC. |
| [HugoChaan/link_u_os_competition](https://github.com/HugoChaan/link_u_os_competition/tree/13523834155353663e14094384fb4663baae3c9b) · 13523834155353663e14094384fb4663baae3c9b | Separate task1-4 runners; task1 is a calibrated odometry closed-loop walk; task3/4 are output/orchestration helpers. | Task1 requires an already-standing robot and explicit target/confirmation. Task3 injects recognition results via CLI. Task4 injects person role/transcript while calling robot services. | Useful bounded task/safe-stop examples, not an integrated manipulation pipeline. Two nav/planner submodules are uninitialized. No arm IK, object grasp, or placement. |

## Verified MC, AimDK, ROS 2, and IK Interfaces

These are package/source facts, not proof that the local simulator connects every endpoint.

| Boundary | Static interface facts | Unverified point |
|---|---|---|
| MC mode | AimDK SetMcAction/GetMcAction services expose PASSIVE_DEFAULT, DAMPING_DEFAULT, JOINT_DEFAULT, STAND_DEFAULT, LOCOMOTION_DEFAULT, HEAD_ONLY, and UPPERBODY_REMOTE_SPLIT. | Runtime transition timing and current MC state have not been observed. |
| Locomotion arbitration | SetMcInputSource registers a named source with priority 0–100 and timeout. /aima/mc/locomotion/velocity carries source, forward/lateral m/s, and angular rate. AimDK docs describe 200 ms stale-command zeroing and a separate 2000 ms LOCOMOTION fallback. | One .msg comment labels angular_velocity as m/s while docs/examples use rad/s; confirm against the pinned runtime. |
| MC state | /aima/mc/common/state reports mode/source and controller state. | No live graph or publisher/subscriber match was checked. |
| Upper body | UpperBodyCommandArray uses arm_pos[14] in left-7/right-7 order and hand_pos[2] as normalized left/right apertures; hand_sub_mode selects the hand control mode. | header.frame_id is metadata, not a Cartesian target. This does not map hand_pos to a URDF joint by itself. |
| Direct joint HAL | JointCommandArray positions/velocities/efforts/stiffness/damping use rad, rad/s, and N·m. | Direct HAL commands have no documented timeout; only use bounded, continuously refreshed commands and safe shutdown. |
| Hand HAL | HandCommandArray has left/right hand_type and per-side hands arrays; OmniPicker is type CLAW=2, one command per side. Apertures are normalized 0–1. | HandCommand.name is debug metadata, not a declared mapping to left_claw_joint/right_claw_joint. No aperture-to-radian calibration was found. |
| Navigation | AimDK PncTaskRequest is a vendor PNC task interface with map-frame PoseStamped targets and lifecycle commands. | It is not a Nav2 action, and the packaged simulator’s PNC/map/localization runtime is not verified. |
| IK | x2_ik_sdk uses Pinocchio, current arm state/seed, and model-frame XYZ; the examined solver is position IK. | No orientation target, collision planning, trajectory safety, or end-to-end ROS-to-simulator result is established. |

## MuJoCo and Grasp Boundaries

The statically inspected packaged sim x2.xml has 31 named body joints and 31 motor actuators plus a free root, covering body/arms/head. Its wrist/hand meshes are static: it has no claw joint DOF, hand actuator, or articulated hand state. The official OmniPicker URDF does contain left_claw_joint/right_claw_joint and mimic joints, but the ROS hand arrays use normalized apertures and the URDF uses radians. The inspected packages define no adapter or endpoint calibration between those representations.

Therefore unsupported or unverified in the stock simulator:

- Opening/closing an articulated hand from HandCommandArray or UpperBodyCommandArray.
- Contact-driven grasp, object lift, retention, and drop detection.
- Any result whose success comes from the 120-second timer or command completion rather than measured object state.
- Placement, release, settling, and destination verification.
- MC/AimDK topic-to-simulator routing, PNC navigation availability, or any hardware response.

Fofarup’s box-claw/LiDAR overlay is a project simulation extension and may help as a diagnostic scaffold, but it changes the mechanical representation and depends on an external LFS runtime archive. Do not call it canonical hardware fidelity.

## Integration Sequence After Standing

Use the current Local03 standing baseline as a prerequisite; do not reset or restart it for this work.

1. **Navigation slice:** connect one localization/velocity bridge and one nav node. Send a calibrated goal; require observed nav_status=reached and zero velocity after arrival.
2. **Interaction:** connect exactly one ASR input to recognized_text, interaction_agent to input_json, and coordinator to a whitelisted service category. Exercise injected text before enabling live ASR.
3. **Target selection:** issue vision_cmd after navigation; verify RGB/depth/CameraInfo freshness, class/confidence, TF, and a finite fresh base_link target_point. If color is scored, first fix the current W_COLOR=0 behavior.
4. **IK/pre-grasp:** confirm model, joint order, tool frame, limits, target reachability, and a collision-free pre-grasp before motion.
5. **Hand adapter:** define normalized left/right aperture semantics and a verified mapping to simulator hand DOFs. Build this as a RobotSim-owned overlay only after separate approval; do not patch vendor assets.
6. **Grasp/lift:** require measured contact, object lift, and retention. Disable forced-success paths and reject command-only success.
7. **Placement/release:** add destination approach, open/release, settle, and object pose/support verification; no inspected implementation supplies this stage.
8. **Combine launchers:** only after each slice passes independently; fail closed on stale localization/targets, IK failure, missing hand state, grasp loss, or failed placement.

## Exact Source Versions

| Source | Exact identity |
|---|---|
| AgiBot Community RAICOM | a97761820b398b6cab12b632ce9f313c394eeb7e |
| Fofarup RAICOM sim | ddbd3fd926846160424ff7fb74daba42b9d2ff5f |
| Link-U-OS RAICOM | c3ffe0d4980e6b7193170cc4f7f29aea680294cc |
| Causiniris autonomy | 22fc981de7c0afffd2cb42fa0b80275bca05e327 |
| HugoChaan Link-U | 13523834155353663e14094384fb4663baae3c9b |
| Official X2 OmniPicker URDF | AgibotTech/agibot_x2_urdf commit 575cc6b988f976c23550e0db85aa1e5475d3652d; X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf |
| AimDK docs/messages | 1.1.0 ZIP; SHA-256 dc80fe5d2907881ed77beb230680867efac6bd28b458f83d2ff4e9479abe3615 |
| MC x86 runtime | 1.0.0-20260522 ZIP; SHA-256 b3bc2417aad656b6a7f19b2274f07d5cb8c7f60b1317482e0122244e57d812d0 |
| MuJoCo sim x86 runtime | 1.0.0-20260522 ZIP; SHA-256 fab28561b5c55767d3775f94c1bb94bad840a3f7c8215e22dccff5ab84a9dc5e |
| Packaged sim robot x2.xml | SHA-256 3ff43f05beb57412a804ba9fe05cd9adcdfce78e9ce73a95a71ac58ad20d91a3 |

## Validation Boundary

This document synthesizes two completed local reports and their pinned source references. It does not claim ROS graph connectivity, simulator integration, grasp success, hardware response, or placement success. No source, model, runtime configuration, reference checkout, build, simulator, or test was changed or run for this task.
