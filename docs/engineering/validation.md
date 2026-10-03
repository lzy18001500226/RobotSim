# Validation contract

Validation is an evidence ladder, not a checklist every change must exhaust. Choose the minimum levels in the matrix, add levels when a change crosses a boundary, and record the exact commands/results. A higher level unavailable in the current environment is `DEFERRED` (or `NOT RUN` when not attempted), never `PASS`.

## Levels

| Level | Evidence |
| --- | --- |
| **L0 — repository/static** | Formatting where configured; config parsing; compile/syntax checks; applicable lint; local-link/workflow sanity; `git diff --check`. |
| **L1 — unit** | Focused C++ or Python unit tests for the component and pure logic. Keep tests with their package; use root `tests/` for repository-wide infrastructure only. |
| **L2 — headless simulation** | Load the model, step physics, check state/finite values and expected invariants, and smoke-test the changed controller or adapter without GUI dependencies. Record seed, model/config, simulator version, and tolerances where relevant. |
| **L3 — ROS integration** | Humble-compatible `colcon build`/`colcon test`, package linters, and `launch_testing` for process/lifecycle, topic/service/action, TF, QoS, and simulation-time behavior touched by the change. |
| **L4 — Unity integration** | Open/run the Unity project in the supported local setup; verify MuJoCo-to-Unity synchronization and the affected sensor/render/scene behavior. Mark visual checks `MANUAL PASS` with environment and evidence when they cannot be automated. |
| **L5 — hardware/HIL** | On the named robot or HIL target, validate vendor runtime, DDS/network path, sensors, hardware timing, and actuator safety. A simulated backend is not L5 evidence. Hardware actuation requires the appropriate safety and human gate. |

## Minimum level by change

| Change | Minimum evidence |
| --- | --- |
| Documentation or policy only | L0 |
| Python repository tooling | L0 + L1 |
| C++ runtime logic | L0 + L1 |
| MuJoCo model, physics, or runtime behavior | L0 + L1 + L2 |
| ROS package/interface/runtime behavior | L0 + L1 + L3, with L2 when simulation behavior is affected |
| Unity scene, synchronization, or sensor frontend | L0 plus relevant levels through L4 |
| Real hardware behavior or deployment | Relevant lower levels plus L5 |

The matrix is a floor, not permission to omit a directly affected integration level. If a ROS or Unity integration cannot run in the current environment, report the missing environment and exact deferred check. Python tests do not establish behavior of a C++ production path.

## Evidence and performance claims

Use one status per applicable check:

- `PASS`: the stated check ran and passed in the named environment.
- `FAIL`: it ran and did not pass; preserve the useful failure evidence.
- `NOT RUN`: it was not attempted.
- `DEFERRED`: it is required for the change but needs an unavailable environment, dependency, or gate. State the exact unlock action.
- `MANUAL PASS`: a person performed a check that is not credibly automated; name the setup and observable result.

Report commands and relevant versions for reproducibility. A build, syntax check, unit test, headless smoke test, manual visual check, and hardware test prove different things; do not promote one level's result into another. For performance claims, capture the target CPU/OS/runtime, workload, warm-up, duration, update rate, latency distribution/tail, jitter, and missed deadlines. Define the budget before measuring and state its scope. A desktop/cloud average or tutorial example is not a real-time guarantee.

## CI growth

CI covers repository agent infrastructure: Python unit tests/compilation, hook JSON parsing, and lightweight repository checks. A separate G1 MuJoCo job also runs the bounded-command unit checks and pinned headless model/stepping smoke test in a temporary Python 3.12 environment. Keep CI focused on reproducible, project-owned checks. Add C++ and package checks when maintained packages exist. Use Humble `ament_lint` and package-local tests when ROS packages are added; use `launch_testing` for the ROS graph behaviors it can exercise. Do not label hosted CI as Unity GPU, Windows/WSLg, vendor DDS, or hardware validation. Those remain local or HIL evidence at L4/L5.

Primary upstream testing references: [ROS 2 Humble colcon testing](https://docs.ros.org/en/humble/Tutorials/Intermediate/Testing/CLI.html), [Humble `launch_testing`](https://docs.ros.org/en/humble/Tutorials/Intermediate/Testing/Integration.html), and [Humble ament lint](https://docs.ros.org/en/humble/Tutorials/Advanced/Ament-Lint-For-Clean-Code.html). REP-2004's quality levels are optional evidence vocabulary, not a blanket RobotSim certification.
