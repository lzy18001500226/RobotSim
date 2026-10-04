# Issue #43: Local LWM Pick-and-Place Reproduction

**Result:** the pinned baseline completed the cylinder pick-and-place task on this Windows 11 + WSL2 host in the documented camera/video mode and in 3/3 fixed-seed trials in each of two scratch modes. This is a useful local M0 behavior fallback. It is not a contact-only physics reference: the upstream policy's table-clearance guard writes the cylinder free-joint `qpos` after simulation steps.

No RobotSim production files or upstream checkout files were changed. The upstream source checkout is detached and clean; this RobotSim branch contains only this report.

## Source and Runtime Identity

| Item | Value |
| --- | --- |
| Upstream | `https://github.com/lwm97/pickandplaceunitreeg1` |
| Commit | `b07678b131885c55175e23d2342548263eaeb9b0` |
| Checkout | `/tmp/issue43-lwm-b07678b`, detached at the requested commit |
| OS | Windows 11 build `10.0.26200`; WSL2 Ubuntu `22.04.5 LTS` |
| WSL kernel | `6.6.87.2-microsoft-standard-WSL2` |
| WSLg | `1.0.71+Branch.main.Sha.b836effa32844bc32bc858e8dd1c5360791171a5` |
| CPU | Intel Core i7-13700H, 20 logical CPUs |
| Python | CPython `3.10.12`, `/usr/bin/python3` |
| Environment manager | `uv 0.11.11` |
| MuJoCo Python/native runtime | `3.3.6` (`libmujoco.so.3.3.6` from the isolated wheel) |
| Native MuJoCo library SHA-256 | `b9173509d0c282a9b24b7f5825a40177a9967df0cd6395a9dc39522196e44495` |
| Timestep | `0.002 s` |

Pinned-source hashes:

| File | SHA-256 |
| --- | --- |
| `g1_scene.xml` | `a7a9f4db9aeb2aaf33cd58366b7567aa2e4b67c1f5b96e1af5030ddcdb06f33c` |
| `g1_robot.xml` | `6dcabb9681b49f1c9c2129d99e0035147d2aaceb1739e4d95213674405cdc776` |
| `requirements.txt` | `223c5277ec235cbe885a2e44f3f157f0b5782ee6198282345b8ac2f35655007d` |

The isolated environment was installed with:

```bash
$HOME/.local/bin/uv venv --python /usr/bin/python3 /tmp/issue43-lwm-b07678b-venv
$HOME/.local/bin/uv pip install --python /tmp/issue43-lwm-b07678b-venv/bin/python 'mujoco==3.3.6' -r /tmp/issue43-lwm-b07678b/requirements.txt
```

The resolved environment included `mujoco==3.3.6`, `numpy==2.2.6`, `opencv-python==5.0.0.93`, `glfw==2.10.2`, `imageio==2.38.0`, `imageio-ffmpeg==0.6.0`, and `pillow==12.3.0` (plus `absl-py==2.5.0`, `etils==1.13.0`, `fsspec==2026.9.0`, `importlib-resources==7.1.0`, `pyopengl==3.1.10`, `typing-extensions==4.16.0`, and `zipp==4.1.1`).

## Default Path

Exact command, from the upstream checkout:

```bash
env -u MUJOCO_GL /tmp/issue43-lwm-b07678b-venv/bin/python run_simulation.py
```

The default randomized camera-guided controller completed successfully, wrote a 420-frame MP4 and JSON, selected the right hand, did not safety-halt, and reported `table_collision_events=54`, `box_collision_events=0`, and final placement error `31.406774 mm`. Final cylinder center was `[0.316636214, 0.055309172, 0.584361095] m`; measured grip force was zero after release. Wall runtime was approximately 85 seconds. The JSON does not serialize the initial cylinder or target-pad pose; the startup observation was approximately cylinder `[0.308, -0.071, 0.580] m` and pad XY `[0.347, 0.049] m`. Exact initialized poses for the seeded trials are recorded below.

The source README describes the active path as camera/OpenCV detection followed by staged damped-least-squares IK and tactile grip control. It is not a learned policy. Source stages include approach, descend, grasp, lift, transfer, lower, release, and retract.

## Scratch Trial Commands

The fixed-seed CV/no-video invocations used the upstream runner with its random generator seeded by the scratch harness:

```bash
env -u MUJOCO_GL /tmp/issue43-lwm-b07678b-venv/bin/python /tmp/issue43-lwm-b07678b-harness.py cv 43 --out /tmp/issue43-evidence/cv-seed-43.json
env -u MUJOCO_GL /tmp/issue43-lwm-b07678b-venv/bin/python /tmp/issue43-lwm-b07678b-harness.py cv 44 --out /tmp/issue43-evidence/cv-seed-44.json
env -u MUJOCO_GL /tmp/issue43-lwm-b07678b-venv/bin/python /tmp/issue43-lwm-b07678b-harness.py cv 45 --out /tmp/issue43-evidence/cv-seed-45.json
```

The no-camera trials disabled GL at process launch and used the harness's ground-truth input path:

```bash
env MUJOCO_GL=disable /tmp/issue43-lwm-b07678b-venv/bin/python /tmp/issue43-lwm-b07678b-harness.py gt 43 --out /tmp/issue43-evidence/gt-seed-43.json
env MUJOCO_GL=disable /tmp/issue43-lwm-b07678b-venv/bin/python /tmp/issue43-lwm-b07678b-harness.py gt 44 --out /tmp/issue43-evidence/gt-seed-44.json
env MUJOCO_GL=disable /tmp/issue43-lwm-b07678b-venv/bin/python /tmp/issue43-lwm-b07678b-harness.py gt 45 --out /tmp/issue43-evidence/gt-seed-45.json
```

## Seeded Trials

The scratch harness and raw JSON/logs are preserved outside RobotSim under `/mnt/c/Users/HP/Desktop/Robot/issue43-local-lwm-repro/`. The same seeds (`43`, `44`, `45`) and randomized cylinder/pad coordinates were used in both modes.

| Mode | Seed | Controller run s | Initial cylinder XY m | Pad XY m | Final cylinder XYZ m | Pad error mm | 0.5 s drift mm | Post-step cylinder `qpos` writes |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| CV, no video | 43 | 3.658 | `[0.317615, -0.072996]` | `[0.306202, 0.067843]` | `[0.277708, 0.070688, 0.584350]` | 28.636 | 0.789 | 636 |
| CV, no video | 44 | 3.393 | `[0.291128, -0.038702]` | `[0.329346, 0.093837]` | `[0.299294, 0.098352, 0.584291]` | 30.389 | 0.805 | 635 |
| CV, no video | 45 | 3.647 | `[0.325555, 0.058444]` | `[0.342791, -0.090618]` | `[0.314569, -0.107102, 0.585538]` | 32.683 | 0.262 | 751 |
| Ground truth, no camera | 43 | 1.839 | `[0.317615, -0.072996]` | `[0.306202, 0.067843]` | `[0.277567, 0.070541, 0.584361]` | 28.761 | 0.786 | 636 |
| Ground truth, no camera | 44 | 1.730 | `[0.291128, -0.038702]` | `[0.329346, 0.093837]` | `[0.299340, 0.098466, 0.584280]` | 30.361 | 0.808 | 635 |
| Ground truth, no camera | 45 | 1.915 | `[0.325555, 0.058444]` | `[0.342791, -0.090618]` | `[0.314512, -0.107078, 0.585543]` | 32.720 | 0.256 | 750 |

All six seeded trials reported success. The CV mode calls the upstream runner with video rendering disabled, but still renders an initial camera image for its CV detector. The ground-truth mode was a scratch-only harness using the same compiled source model, policy, adaptive stages, actuator controller, and MuJoCo stepping; it substituted the known cylinder and pad poses for the initial camera detection and created no renderer. That mode is the smallest tested no-camera path, not an unchanged upstream CLI option.

Controller-run times above exclude the separate 250-step stability tail. They are task reproduction timings, not benchmark measurements.

## Contact, Lift, and Stability

- All six trials recorded cylinder/finger contacts in 34 of 50 GRASP samples and 40 of 40 LIFT samples. Peak summed fingertip normal force during GRASP was `0.539–0.663 N`.
- During LIFT, the cylinder center rose about `145.0–145.6 mm` from the start to the end of the phase, reaching approximately `0.728–0.730 m` above the scene origin.
- RELEASE recorded cylinder/finger contact in 14 or 17 of 30 samples, depending on seed. All six ended with a cylinder/target-plate contact.
- Each run then stepped another 250 steps (`0.5 s`). XY drift was `0.256–0.808 mm`; final XY placement error was `28.636–32.720 mm`, below the scratch harness's `50 mm` task threshold.
- Phase contact evidence is simulated MuJoCo contact, not physical robot/hardware evidence.

## Object-State Writes and Limitations

The harness records one pre-first-step cylinder free-joint initialization, then counts later changes made by the source's `enforce_box_table_clearance()` guard. There were `635–751` post-step object `qpos` writes per seeded run. The first correction occurred at simulation time `0.122 s`, moving the cylinder upward about `18.15–18.17 mm` after it had fallen to about `z=0.563 m`. Most corrections were during early approach/grasp. Seed 45 also had ten sub-millimeter early-lift corrections (maximum about `0.232 mm`) and two after release; none occurred during transfer.

Consequently, the run demonstrates repeatable task completion with real finger contact and a substantial contact-supported lift, but it is not a pure contact-only demonstration. The runtime clearance guard directly changes authoritative object state. Treat this as a local task-behavior fallback, not as ground-truth evidence for an unmodified physical contact model, until that guard is reviewed separately.

## WSL2 Rendering Behavior

- With `MUJOCO_GL` unset, MuJoCo selected its GLFW context. The default full-video path completed through WSLg/GLX and wrote all 420 frames. `glxinfo -B` reported accelerated OpenGL 4.1 through Mesa 23.2.1, renderer `Microsoft Corporation D3D12 (Intel(R) Iris(R) Xe Graphics)`.
- WSLg exposed `DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`, and `/dev/dxg`; `/dev/dri` was absent. WSLg EGL 1.5 initialization worked on Wayland/X11. A GBM EGL probe failed with `eglInitialize failed`, while an explicit `MUJOCO_GL=egl` 64x64 MuJoCo render probe succeeded.
- `MUJOCO_GL=disable` worked for the scratch ground-truth no-camera process because it did not create a renderer.
- No blocking WSLg issue prevented either the default render path or the no-camera headless path.

The default video run began at about 14:27 local time. A separate competing MuJoCo pick/place process was detected later; physical trials were paused while it was active and resumed only after it exited and process/load snapshots were quiet. No performance conclusion is drawn from these reproduction runtimes.

## Evidence and Readiness

External raw evidence:

- `default/`: default JSON, MP4, and approach/grasp/transfer/success screenshots
- `camera-target-no-video/`: three seeded CV-mode JSON and log files
- `ground-truth-no-camera/`: three seeded no-render JSON and log files
- `issue43-lwm-b07678b-harness.py`: scratch instrumentation and no-camera harness

The raw default MP4 SHA-256 is `ca2c792c8c888e23d5f09f6c4d2ec9731431523961d899e52eee5865b9f02469`; its JSON SHA-256 is `70618df1b85cc7b7330bcbc57d23b5d8fcb3c03e3ab90b7a037a5a5153092326`. The isolated upstream checkout and temporary Python environment remain under `/tmp`; no vendor assets were copied into RobotSim.

**Fallback readiness:** ready as a local M0 scripted behavior fallback on this host, with the explicit limitation that upstream post-step `qpos` correction is part of the behavior. It is not physical hardware validation, a pure-contact physics reference, or a performance benchmark.
