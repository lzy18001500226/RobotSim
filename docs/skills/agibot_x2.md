# Technical note: AgiBot X2 — authoritative documents and troubleshooting

RobotSim's X2 integration uses vendor assets and contracts that are **version-specific**. This page is the compact source index for the root [official-source-first troubleshooting rule](../../AGENTS.md#mandatory-official-source-first-troubleshooting), **not** a claim that the current RobotSim runtime implements everything in the latest manual.

## Source index — consult before inventing an MC/ROS/MuJoCo workaround

| Source | Link | Use when |
| --- | --- | --- |
| AgiBot AimDK X2 documentation index (online `latest`) | https://x2-aimdk.agibot.com/zh-cn/latest/index.html | Discover current vendor manuals and version notes; verify applicability to delivered release. |
| Official X2 MuJoCo motion simulation | https://x2-aimdk.agibot.com/zh-cn/latest/sim_contents.html | Startup order, `start_sim.sh` / `em_run.sh`, ROS environment, Docker/local setup, SD/LD and example walk verification. |
| Official motion-mode switching | https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/modeswitch.html | SD / locomotion state meanings and documented automatic transition behavior; compare MC version before assuming it applies. |
| Official locomotion control | https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/locomotion.html | Velocity interfaces and mode requirements. |
| Official MC signal/input-source configuration | https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/MC_control.html | Registration, enablement, priority arbitration, zero/nonzero source selection, timeouts and control-safety boundaries. |
| Official Python interface examples | https://x2-aimdk.agibot.com/zh-cn/latest/example/Python.html | Locate runnable `py_examples` and their actual request fields / topic messages. |
| RAICOM-distributed startup manual (pinned) | https://github.com/Link-U-OS/link_u_os_competition/blob/c3ffe0d4980e6b7193170cc4f7f29aea680294cc/Raicom2026/README.md | Compare *that delivery's* simulator/MC archives, Docker flags, SD then GUI Reset and setup order. |
| RAICOM `set_mode.py` (pinned) | https://github.com/Link-U-OS/link_u_os_competition/blob/c3ffe0d4980e6b7193170cc4f7f29aea680294cc/Raicom2026/example/py/set_mode.py | Inspect `SetMcAction` requests and its limitations (RPC response alone is not live readback). |
| RAICOM `get_mode.py` (pinned) | https://github.com/Link-U-OS/link_u_os_competition/blob/c3ffe0d4980e6b7193170cc4f7f29aea680294cc/Raicom2026/example/py/get_mode.py | Confirm live mode using `GetMcAction`. |
| Official X2 v1.4.0 OmniPicker robot description (pinned) | https://github.com/AgibotTech/agibot_x2_urdf/blob/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf | Exact link/joint/mesh provenance; subject to model integrity approval before changes. |

The pinned RAICOM repository is a *competition-delivery reference*, not proof that RobotSim's installed precompiled x86 MC matches the online AimDK 1.1.0 behavior. Online `latest` pages can change; record the page section, retrieval date, documented SDK version, and actual installed binary hashes in relevant evidence. A community bridge or another robot's MJCF is secondary reference only; do not promote it into an X2 official contract.

## Required debugging sequence (X2 locomotion example)

1. Record installed MC, `sim_mujoco` and `aimdk_msgs` revisions/hashes and compare actual startup scripts with the matched delivery manual. Verify runtime user, `RMW_IMPLEMENTATION`, ROS domain, topics, safety state, and MC/Sim startup order; account for any documented SD/Reset requirement. A recent official `-s` command may not apply to an older archive.
2. Reproduce official **read-only** get-mode and state/odometry checks first. Match exact service requests, response fields, source identity, timestamp and QoS to the installed SDK and examples. Keep Reset and stable-standing gates where already established.
3. Distinguish `SetMcInputSource ADD` RPC acknowledgement from task completion and **registered** input source from the **currently active** source. A zero-only stream with no selected source may be expected under official priority rules; it does not alone prove failed registration. Distinguish `SetMcAction` success from actual `GetMcAction` / `McCommonState` mode changes.
4. Check whether the **installed** MC's SD/locomotion transition is request-driven or can activate upon a valid motion command. Never build a circular readiness gate that requires a nonzero-triggered state to appear before the trigger *unless documented for that version*.
5. Form a bounded, safety-reviewed experiment only after the source/version comparison: one publisher, actual live velocity bounds, stable standing, stop/watchdog, fault/tilt/height checks, input-source eligibility, measured root translation/heading and strict duration. **Do not send nonzero velocity without the required task-specific maintainer authorization and verified safety plan.**
6. If a vendor prerequisite remains unknown, provide a minimal escalation packet showing exact documented facts, installed-vs-manual mismatches, request/response and live readbacks, and why safe independent work is exhausted.

## Other X2 boundaries

Expected components: official X2 URDF / MJCF, AimDK, MC, sim_mujoco and X2 grasp examples. The official MC may supply vendor locomotion capability; E1R/Gemini335 simulation is a separate project layer.

For OmniPicker, geometry, static CAD mates/part placements, physical joint semantics, drive/transmission constraints and successful `mj_step` physics are distinct evidence levels. Refer to [model integrity](../engineering/model_integrity.md): do not change fixed joints, linkage, collisions or canonical meshes without **specific mechanical-model approval**, even for a `SIMULATION_ONLY` prototype. Searching documentation is required before claiming the source is unavailable, but missing proprietary mechanical semantics may still require a human or vendor decision.
