"""Pure, offline-testable safety logic for the X2 M1 first-motion probe."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

VELOCITY_SOURCE_NAME = "node.locomotion_velocity"
ACTION_SOURCE_NAME = "rc"  # exact source used by the installed set_mode.py example
SOURCE_PRIORITY = 40
SOURCE_TIMEOUT_MS = 1000
SOURCE_TASK_TERMINAL_FAILURE_STATES = frozenset((2, 3, 4, 5, 6))
SOURCE_TASK_KNOWN_STATES = frozenset((0, 1, 2, 3, 4, 5, 6, 100, 200, 300, 400))
FORWARD_COMMAND_M_S = 0.2
PULSE_LIMIT_S = 1.0
PUBLISH_HZ = 50.0
COMMAND_QOS_RELIABILITY = "reliable"
COMMAND_QOS_DEPTH = 10
COMMAND_QOS_DURABILITY = "volatile"
ZERO_STOP_MIN_S = 1.0
ZERO_STOP_MIN_MESSAGES = math.ceil(SOURCE_TIMEOUT_MS * PUBLISH_HZ / 1000.0)
POST_STOP_OBSERVE_S = 2.5

MODE_STAND = "STAND_DEFAULT"
MODE_LOCOMOTION = "LOCOMOTION_DEFAULT"
ALLOWED_MODES = frozenset((MODE_STAND, MODE_LOCOMOTION))
MC_ACTION_STATUSES = frozenset((0, 100, 200))  # IDLE, RUNNING, TRANSITION
MC_FSM_ACTIVE_STATES = frozenset((1, 2, 3))  # STARTING, STABLE, MOVING
MC_FSM_SAFE = 4
MC_FSM_SPECIAL = 5
MC_FSM_TEST = 6
BODY_POSE_STAND = 1
REQUIRED_STATE_KEYS = ("odom", "imu", "common", "leg_state", "leg_command", "get_mode", "get_source")


def input_source_action_result(
    add_response: Mapping[str, Any] | None,
    enable_response: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate service acceptance from completion of its asynchronous task."""
    add_code = add_response.get("response_header_code") if isinstance(add_response, Mapping) else None
    enable_code = enable_response.get("response_header_code") if isinstance(enable_response, Mapping) else None
    add_task_state = add_response.get("response_state") if isinstance(add_response, Mapping) else None
    add_task_failed = add_task_state in SOURCE_TASK_TERMINAL_FAILURE_STATES
    add_explicitly_rejected = add_code is not None and add_code != 0
    add_recovery_eligible = add_explicitly_rejected or add_task_failed
    selected_action = (
        "ENABLE" if add_recovery_eligible and enable_code == 0
        else "ADD" if add_code == 0 and not add_task_failed
        else None
    )
    selected = add_response if selected_action == "ADD" else enable_response if selected_action == "ENABLE" else None
    state_value = selected.get("response_state") if isinstance(selected, Mapping) else None
    diagnostic_state_value = state_value if selected_action is not None else add_task_state
    task_state_known = diagnostic_state_value in SOURCE_TASK_KNOWN_STATES
    task_terminal_failure = diagnostic_state_value in SOURCE_TASK_TERMINAL_FAILURE_STATES
    task_state_unready = diagnostic_state_value == 100
    operation_accepted = selected_action is not None
    service_accepted = add_code == 0 or (add_recovery_eligible and enable_code == 0)
    return {
        # `success` means accepted with a recognized non-failure task state;
        # `task_completion_confirmed` remains separate for asynchronous responses.
        "success": operation_accepted and task_state_known and not task_terminal_failure and not task_state_unready,
        "accepted_by_service": service_accepted,
        "selected_action": selected_action,
        "task_id": selected.get("task_id") if isinstance(selected, Mapping) else None,
        "task_state_value": diagnostic_state_value,
        "task_state_known": task_state_known,
        "task_terminal_failure": task_terminal_failure,
        "task_state_unready": task_state_unready,
        "task_completion_confirmed": state_value == 1,
        "enable_fallback_allowed": add_explicitly_rejected or add_task_failed,
        "enable_fallback_reason": (
            "ADD_SERVICE_REJECTED" if add_explicitly_rejected
            else "ADD_TASK_TERMINAL_FAILURE" if add_task_failed else None
        ),
        "add_response": dict(add_response) if isinstance(add_response, Mapping) else None,
        "enable_response": dict(enable_response) if isinstance(enable_response, Mapping) else None,
    }


def classify_source_registration(
    add_response: Mapping[str, Any] | None,
    enable_response: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility-named wrapper used by the offline source registration tests."""
    return input_source_action_result(add_response, enable_response)


def load_leg_joint_limits(path: str | Path) -> tuple[dict[str, dict[str, float]], str]:
    """Read the installed simulation robot_model.yaml limits for the 12 leg joints."""
    import yaml

    source = Path(path)
    payload = source.read_bytes()
    document = yaml.safe_load(payload)
    groups = document.get("joints") if isinstance(document, Mapping) else None
    if not isinstance(groups, list):
        raise ValueError("robot_model.yaml has no joints list")
    leg_entries: list[Mapping[str, Any]] = []
    for group in groups:
        if isinstance(group, (list, tuple)) and len(group) == 2 and group[0] == "LEG":
            if not isinstance(group[1], list):
                raise ValueError("LEG joint limits are not a list")
            leg_entries = group[1]
            break
    if not leg_entries:
        raise ValueError("robot_model.yaml has no LEG group")
    limits: dict[str, dict[str, float]] = {}
    for entry in leg_entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("limits"), Mapping):
            raise ValueError("malformed LEG joint limit entry")
        name = entry.get("name")
        raw = entry["limits"]
        if not isinstance(name, str) or not name or name in limits:
            raise ValueError("missing or duplicate LEG joint name")
        try:
            lower = float(raw["lower_position"])
            upper = float(raw["upper_position"])
            max_velocity = float(raw["max_velocity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"incomplete source limits for {name}") from exc
        if not all(math.isfinite(value) for value in (lower, upper, max_velocity)):
            raise ValueError(f"nonfinite source limits for {name}")
        if lower >= upper or max_velocity <= 0.0:
            raise ValueError(f"invalid source limits for {name}")
        limits[name] = {
            "lower_position": lower,
            "upper_position": upper,
            "max_velocity": max_velocity,
        }
    if len(limits) != 12:
        raise ValueError(f"expected 12 source-limited leg joints, found {len(limits)}")
    return limits, hashlib.sha256(payload).hexdigest()


def validate_runtime_scope(
    *,
    rmw_implementation: str | None,
    ros_domain_id: str | None,
    docker_network_mode: str | None,
    publisher_names: list[str],
    subscriber_names: list[str],
    own_node_name: str,
    publisher_reliability: str,
    subscriber_reliabilities: list[str],
    publisher_durability: str,
    subscriber_durabilities: list[str],
    publisher_history: str,
    publisher_depth: int,
    subscriber_histories: list[str],
    subscriber_depths: list[int],
    expected_mc_node_name: str = "mc_ros2_node",
) -> str | None:
    """Fail closed unless this is the isolated local MC/simulator graph."""
    if rmw_implementation != "rmw_fastrtps_cpp":
        return "unexpected_rmw_implementation"
    if ros_domain_id != "20":
        return "unexpected_ros_domain_id"
    if docker_network_mode != "bridge":
        return "container_network_not_isolated_bridge"
    if publisher_names != [own_node_name]:
        return f"velocity_publisher_conflict:{publisher_names!r}"
    if subscriber_names != [expected_mc_node_name]:
        return f"expected_only_one_local_mc_subscriber:{subscriber_names!r}"
    if publisher_reliability != "reliable":
        return f"command_publisher_qos_not_reliable:{publisher_reliability!r}"
    if len(subscriber_reliabilities) != len(subscriber_names):
        return "subscriber_qos_endpoint_count_mismatch"
    if any(value not in ("reliable", "best_effort") for value in subscriber_reliabilities):
        return f"unknown_mc_subscriber_qos:{subscriber_reliabilities!r}"
    if publisher_durability != "volatile":
        return f"command_publisher_durability_not_volatile:{publisher_durability!r}"
    if len(subscriber_durabilities) != len(subscriber_names):
        return "subscriber_durability_endpoint_count_mismatch"
    if any(value != "volatile" for value in subscriber_durabilities):
        return f"mc_subscriber_durability_not_volatile:{subscriber_durabilities!r}"
    if publisher_history != "keep_last" or publisher_depth != 10:
        return f"command_publisher_history_depth_mismatch:{publisher_history}:{publisher_depth}"
    if len(subscriber_histories) != len(subscriber_names) or len(subscriber_depths) != len(subscriber_names):
        return "subscriber_history_depth_endpoint_count_mismatch"
    if any(history != "keep_last" or depth <= 0 for history, depth in zip(subscriber_histories, subscriber_depths)):
        return f"mc_subscriber_history_depth_unsupported:{subscriber_histories!r}:{subscriber_depths!r}"
    return None


def validate_local_sim_publishers(topic_publishers: Mapping[str, list[str]]) -> str | None:
    """Require one state publisher per simulator stream and one MC leg-command publisher."""
    required = (
        "/aima/hal/odom/state",
        "/aima/hal/imu/torso/state",
        "/aima/hal/joint/leg/state",
        "/aima/hal/joint/leg/command",
    )
    for topic in required:
        publishers = topic_publishers.get(topic)
        if not isinstance(publishers, list) or len(publishers) != 1:
            return f"unexpected_local_runtime_publishers:{topic}:{publishers!r}"
    if topic_publishers["/aima/hal/joint/leg/command"] != ["mc_ros2_node"]:
        return (
            "unexpected_mc_leg_command_publisher:"
            f"{topic_publishers['/aima/hal/joint/leg/command']!r}"
        )
    return None


def validate_first_motion_authorization(
    *, reset_gate_verified: bool, maintainer_authorized: bool, network_mode: str | None,
) -> None:
    """Fail before ROS initialization unless review, Reset, and bridge gates hold."""
    if not reset_gate_verified:
        raise PermissionError("verified post-Reset standing gate is required")
    if not maintainer_authorized:
        raise PermissionError("explicit maintainer authorization is required")
    if network_mode != "bridge":
        raise ValueError("first motion requires the parent runner's verified bridge marker")


def snapshot_fault(
    snapshot: Mapping[str, Any],
    joint_limits: Mapping[str, Mapping[str, float]],
    *,
    state_timeout_s: float,
    min_root_z: float,
    max_tilt_deg: float,
    expected_source: str = VELOCITY_SOURCE_NAME,
) -> str | None:
    if snapshot.get("readback_fault"):
        return f"readback_fault:{snapshot['readback_fault']}"
    ages = snapshot.get("state_age_s")
    if not isinstance(ages, Mapping):
        return "missing_state_ages"
    for key in REQUIRED_STATE_KEYS:
        age = ages.get(key)
        if not isinstance(age, (int, float)) or not math.isfinite(float(age)) or not 0.0 <= float(age) <= state_timeout_s:
            return f"stale_or_missing_state:{key}"

    get_mode = snapshot.get("get_mode_desc")
    common_mode = snapshot.get("common_mode_desc")
    if get_mode not in ALLOWED_MODES or common_mode not in ALLOWED_MODES:
        return f"unrelated_or_missing_mode:{get_mode!r}:{common_mode!r}"
    if snapshot.get("get_mode_header_code") != 0:
        return "get_mc_action_readback_error"
    if snapshot.get("get_mode_status_value") not in MC_ACTION_STATUSES:
        return "unknown_get_mc_action_status"
    if snapshot.get("common_mode_status_value") not in MC_ACTION_STATUSES:
        return "unknown_common_action_status"

    fsm = snapshot.get("fsm_state")
    if fsm == MC_FSM_SAFE:
        return "mc_safety_protection_state"
    if fsm not in MC_FSM_ACTIVE_STATES:
        return f"unrelated_or_unknown_fsm_state:{fsm!r}"
    if snapshot.get("body_status") != BODY_POSE_STAND:
        return f"non_standing_body_status:{snapshot.get('body_status')!r}"

    pose = snapshot.get("pose")
    if not isinstance(pose, Mapping):
        return "missing_odom_pose"
    try:
        root_z = float(pose["z"])
        roll = float(pose["roll"])
        pitch = float(pose["pitch"])
    except (KeyError, TypeError, ValueError):
        return "malformed_odom_pose"
    if not all(math.isfinite(value) for value in (root_z, roll, pitch)):
        return "nonfinite_odom_pose"
    if root_z < min_root_z:
        return "fall_root_height"
    if max(abs(roll), abs(pitch)) > math.radians(max_tilt_deg):
        return "fall_tilt"

    bounds = snapshot.get("forward_bounds")
    if not isinstance(bounds, Mapping):
        return "missing_live_forward_bounds"
    try:
        minimum = float(bounds["min_value"])
        maximum = float(bounds["max_value"])
    except (KeyError, TypeError, ValueError):
        return "malformed_live_forward_bounds"
    if not math.isfinite(minimum) or not math.isfinite(maximum) or not minimum <= FORWARD_COMMAND_M_S <= maximum:
        return "activation_speed_outside_live_mc_bounds"

    source_names = (
        snapshot.get("get_source_name", ""),
        snapshot.get("common_source_name", ""),
    )
    if snapshot.get("get_source_header_code") != 0:
        return "get_current_input_source_readback_error"
    if any(name not in ("", expected_source) for name in source_names):
        return f"input_source_conflict:{source_names!r}"

    joints = snapshot.get("leg_states")
    if not isinstance(joints, list):
        return "missing_leg_joint_states"
    by_name: dict[str, Mapping[str, Any]] = {}
    for joint in joints:
        if not isinstance(joint, Mapping) or not isinstance(joint.get("name"), str):
            return "malformed_leg_joint_state"
        name = joint["name"]
        if name in by_name:
            return f"duplicate_leg_joint_state:{name}"
        by_name[name] = joint
    if set(by_name) != set(joint_limits):
        return "leg_joint_limit_mapping_incomplete"
    for name, joint in by_name.items():
        limits = joint_limits[name]
        try:
            position = float(joint["position"])
            velocity = float(joint["velocity"])
            lower = float(limits["lower_position"])
            upper = float(limits["upper_position"])
            max_velocity = float(limits["max_velocity"])
        except (KeyError, TypeError, ValueError):
            return f"malformed_leg_joint_values:{name}"
        if not all(math.isfinite(value) for value in (position, velocity, lower, upper, max_velocity)):
            return f"nonfinite_leg_joint_values:{name}"
        if position < lower or position > upper:
            return f"source_joint_position_limit:{name}"
        if abs(velocity) > max_velocity:
            return f"source_joint_velocity_limit:{name}"
    return None


class ActivationTracker:
    """Record source arbitration and SD/LD transition evidence without gating startup."""

    def __init__(self, expected_source: str = VELOCITY_SOURCE_NAME) -> None:
        self.expected_source = expected_source
        self.get_source_seen = False
        self.common_source_seen = False
        self.get_locomotion_seen = False
        self.common_locomotion_seen = False
        self.samples: list[dict[str, Any]] = []
        self.first_pose: dict[str, float] | None = None
        self.last_pose: dict[str, float] | None = None

    def observe(self, snapshot: Mapping[str, Any]) -> None:
        row = {
            "get_mode_desc": snapshot.get("get_mode_desc"),
            "common_mode_desc": snapshot.get("common_mode_desc"),
            "get_source_name": snapshot.get("get_source_name", ""),
            "common_source_name": snapshot.get("common_source_name", ""),
            "fsm_state": snapshot.get("fsm_state"),
            "pose": dict(snapshot.get("pose", {})) if isinstance(snapshot.get("pose"), Mapping) else None,
        }
        self.samples.append(row)
        if isinstance(row["pose"], Mapping):
            pose = {key: float(value) for key, value in row["pose"].items() if isinstance(value, (int, float))}
            if self.first_pose is None:
                self.first_pose = pose
            self.last_pose = pose
        self.get_source_seen |= row["get_source_name"] == self.expected_source
        self.common_source_seen |= row["common_source_name"] == self.expected_source
        self.get_locomotion_seen |= row["get_mode_desc"] == MODE_LOCOMOTION
        self.common_locomotion_seen |= row["common_mode_desc"] == MODE_LOCOMOTION

    def classify(self) -> str:
        if not self.get_source_seen or not self.common_source_seen:
            return "SOURCE_NOT_ACQUIRED"
        if not self.get_locomotion_seen or not self.common_locomotion_seen:
            return "MISSING_MODE_TRANSITION"
        return "ACTIVATION_CONFIRMED"

    def odometry_delta(self) -> dict[str, float | int] | None:
        """Report observed activation displacement without inventing a pass distance."""
        if self.first_pose is None or self.last_pose is None:
            return None
        delta_yaw = self.last_pose.get("yaw", 0.0) - self.first_pose.get("yaw", 0.0)
        return {
            "dx_m": self.last_pose.get("x", 0.0) - self.first_pose.get("x", 0.0),
            "dy_m": self.last_pose.get("y", 0.0) - self.first_pose.get("y", 0.0),
            "dyaw_rad": math.atan2(math.sin(delta_yaw), math.cos(delta_yaw)),
            "sample_count": len(self.samples),
        }


class IndependentWatchdog:
    """Timer-backed watchdog that runs outside the ROS executor/spin loop."""

    def __init__(
        self,
        timeout_s: float,
        on_timeout: Callable[[], None],
        *,
        timer_factory: Callable[[float, Callable[[], None]], Any] | None = None,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("watchdog timeout must be positive")
        factory = timer_factory or threading.Timer
        self._expired = threading.Event()
        self._on_timeout = on_timeout
        self._timer = factory(timeout_s, self._expire)
        if hasattr(self._timer, "daemon"):
            self._timer.daemon = True

    @property
    def expired(self) -> bool:
        return self._expired.is_set()

    def _expire(self) -> None:
        if self._expired.is_set():
            return
        self._expired.set()
        self._on_timeout()

    def start(self) -> None:
        self._timer.start()

    def cancel(self) -> None:
        self._timer.cancel()


class ZeroStopWindow:
    """One idempotent zero burst covering the full 1000 ms source timeout."""

    def __init__(
        self,
        publish_zero: Callable[[], None],
        *,
        timeout_s: float = ZERO_STOP_MIN_S,
        publish_hz: float = PUBLISH_HZ,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        command_lock: Any | None = None,
    ) -> None:
        if timeout_s < ZERO_STOP_MIN_S:
            raise ValueError("zero stop must cover one full 1000 ms source timeout")
        if publish_hz != PUBLISH_HZ:
            raise ValueError("zero-stop cadence is fixed to 50 Hz")
        self.publish_zero = publish_zero
        self.timeout_s = timeout_s
        self.period_s = 1.0 / publish_hz
        self.clock = clock
        self.sleep = sleep
        self.command_lock = command_lock or threading.RLock()
        self._start_lock = threading.Lock()
        self._started = threading.Event()
        self._done = threading.Event()
        self.result: dict[str, Any] = {
            "requested_duration_s": timeout_s,
            "actual_duration_s": 0.0,
            "zero_messages": 0,
            "successful_message_times_s": [],
            "published_span_s": 0.0,
            "max_inter_message_gap_s": None,
            "publish_errors": [],
        }

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def run_once(self) -> dict[str, Any]:
        with self._start_lock:
            if self._done.is_set():
                return dict(self.result)
            already_running = self._started.is_set()
            if not already_running:
                self._started.set()
        if already_running:
            self._done.wait(timeout=self.timeout_s + 2.0)
            return dict(self.result)

        start = self.clock()
        deadline = start + self.timeout_s
        try:
            while self.clock() < deadline:
                try:
                    with self.command_lock:
                        self.publish_zero()
                    self.result["zero_messages"] += 1
                    self.result["successful_message_times_s"].append(self.clock() - start)
                except Exception as exc:
                    self.result["publish_errors"].append(f"{type(exc).__name__}: {exc}")
                remaining = deadline - self.clock()
                if remaining > 0.0:
                    self.sleep(min(self.period_s, remaining))
        finally:
            self.result["actual_duration_s"] = max(0.0, self.clock() - start)
            message_times = self.result["successful_message_times_s"]
            if message_times:
                self.result["published_span_s"] = max(0.0, message_times[-1] - message_times[0])
                if len(message_times) > 1:
                    self.result["max_inter_message_gap_s"] = max(
                        later - earlier for earlier, later in zip(message_times, message_times[1:])
                    )
            self._done.set()
        return dict(self.result)


def run_first_motion_window(
    *,
    sample: Callable[[], Mapping[str, Any]],
    publish: Callable[[float], None],
    registration: Mapping[str, Any],
    set_action: Mapping[str, Any],
    joint_limits: Mapping[str, Mapping[str, float]],
    authorized: bool,
    shutdown_event: threading.Event | None = None,
    on_sample: Callable[[str, Mapping[str, Any]], None] | None = None,
    state_timeout_s: float = 0.5,
    min_root_z: float = 0.52,
    max_tilt_deg: float = 15.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    timer_factory: Callable[[float, Callable[[], None]], Any] | None = None,
) -> dict[str, Any]:
    """One immutable 0.2 m/s, <=1 s, 50 Hz pulse followed by stop and observation."""
    result: dict[str, Any] = {
        "status": "NOT_RUN",
        "source": VELOCITY_SOURCE_NAME,
        "command_forward_m_s": FORWARD_COMMAND_M_S,
        "command_lateral_m_s": 0.0,
        "command_angular_velocity_rad_s": 0.0,
        "pulse_limit_s": PULSE_LIMIT_S,
        "publish_hz": PUBLISH_HZ,
        "source_timeout_ms": SOURCE_TIMEOUT_MS,
        "registration": dict(registration),
        "set_mc_action": dict(set_action),
        "nonzero_messages": 0,
        "samples": [],
        "fault": None,
    }
    if not authorized:
        result["status"] = "NOT_AUTHORIZED"
        return result

    pulse_fault: str | None = None
    if not registration.get("success"):
        pulse_fault = "source_registration_rejected"
    elif not set_action.get("accepted_by_service"):
        pulse_fault = "set_mc_action_rejected"

    command_lock = threading.RLock()
    stop_requested = threading.Event()
    tracker = ActivationTracker()

    def record_sample(phase: str, snapshot: Mapping[str, Any], fault: str | None) -> None:
        row = dict(snapshot)
        row["phase"] = phase
        row["safety_check"] = "FAIL" if fault else "PASS"
        row["safety_fault"] = fault
        result["samples"].append(row)
        if on_sample:
            on_sample(phase, row)

    zero_stop = ZeroStopWindow(
        lambda: publish(0.0),
        timeout_s=max(ZERO_STOP_MIN_S, SOURCE_TIMEOUT_MS / 1000.0) + 1.0 / PUBLISH_HZ,
        publish_hz=PUBLISH_HZ,
        clock=clock,
        sleep=sleep,
        command_lock=command_lock,
    )
    watchdog: IndependentWatchdog | None = None
    pulse_start: float | None = None
    pulse_end: float | None = None
    nonzero_attempt_offsets_s: list[float] = []
    nonzero_success_offsets_s: list[float] = []

    def on_watchdog() -> None:
        nonlocal pulse_fault
        # The timer is also the independent hard-stop at the nominal one-second
        # deadline. Only an early firing is a watchdog fault; a deadline firing
        # still starts the mandatory zero window without turning a normal stop
        # into a false failure.
        if clock() < pulse_deadline - 0.01:
            pulse_fault = pulse_fault or "independent_watchdog_timeout"
        stop_requested.set()
        zero_stop.run_once()

    try:
        if pulse_fault is None:
            initial = sample()
            initial_fault = snapshot_fault(
                initial,
                joint_limits,
                state_timeout_s=state_timeout_s,
                min_root_z=min_root_z,
                max_tilt_deg=max_tilt_deg,
            )
            record_sample("pre-activation", initial, initial_fault)
            if initial_fault:
                pulse_fault = initial_fault
            else:
                tracker.observe(initial)

        if pulse_fault is None:
            pulse_start = clock()
            pulse_deadline = pulse_start + PULSE_LIMIT_S
            next_publish = pulse_start
            watchdog = IndependentWatchdog(
                PULSE_LIMIT_S,
                on_watchdog,
                timer_factory=timer_factory,
            )
            watchdog.start()
            while clock() < pulse_deadline:
                if shutdown_event is not None and shutdown_event.is_set():
                    pulse_fault = "operator_shutdown"
                    stop_requested.set()
                    break
                if watchdog.expired:
                    if clock() < pulse_deadline - 0.01:
                        pulse_fault = pulse_fault or "independent_watchdog_timeout"
                    stop_requested.set()
                    break
                current = sample()
                current_fault = snapshot_fault(
                    current,
                    joint_limits,
                    state_timeout_s=state_timeout_s,
                    min_root_z=min_root_z,
                    max_tilt_deg=max_tilt_deg,
                )
                if current_fault:
                    record_sample("activation", current, current_fault)
                    pulse_fault = current_fault
                    stop_requested.set()
                    break
                tracker.observe(current)
                record_sample("activation", current, None)
                now = clock()
                if shutdown_event is not None and shutdown_event.is_set():
                    pulse_fault = "operator_shutdown"
                    stop_requested.set()
                    break
                if watchdog.expired or stop_requested.is_set():
                    if watchdog.expired and clock() < pulse_deadline - 0.01:
                        pulse_fault = pulse_fault or "independent_watchdog_timeout"
                    stop_requested.set()
                    break
                if now >= pulse_deadline:
                    break
                if now >= next_publish:
                    should_stop = False
                    with command_lock:
                        # Check again after taking the command lock: the
                        # watchdog may have queued a stop while this thread waited.
                        if stop_requested.is_set() or clock() >= pulse_deadline:
                            should_stop = True
                        else:
                            nonzero_attempt_offsets_s.append(clock() - pulse_start)
                            publish(FORWARD_COMMAND_M_S)
                            published_offset = clock() - pulse_start
                            nonzero_success_offsets_s.append(published_offset)
                            if published_offset > PULSE_LIMIT_S:
                                pulse_fault = pulse_fault or "nonzero_publish_returned_after_hard_deadline"
                                stop_requested.set()
                    if should_stop:
                        break
                    result["nonzero_messages"] += 1
                    next_publish = now + 1.0 / PUBLISH_HZ
                remaining = pulse_deadline - clock()
                if remaining > 0.0:
                    sleep(min(0.005, remaining))
            pulse_end = clock()
        else:
            pulse_start = clock()
            pulse_end = pulse_start
    except Exception as exc:
        pulse_fault = pulse_fault or f"exception:{type(exc).__name__}:{exc}"
        stop_requested.set()
    finally:
        if pulse_start is not None and pulse_end is None:
            pulse_end = clock()
        stop_requested.set()
        if watchdog is not None:
            watchdog.cancel()
        stop_result = zero_stop.run_once()
        result["zero_stop"] = stop_result
        post_stop_end = clock() + POST_STOP_OBSERVE_S
        post_stop_samples = 0
        post_stop_fault: str | None = None
        while clock() < post_stop_end:
            try:
                current = sample()
                current_fault = snapshot_fault(
                    current,
                    joint_limits,
                    state_timeout_s=state_timeout_s,
                    min_root_z=min_root_z,
                    max_tilt_deg=max_tilt_deg,
                )
                if current_fault:
                    post_stop_fault = current_fault
                record_sample("post-stop", current, current_fault)
                post_stop_samples += 1
            except Exception as exc:
                post_stop_fault = f"post_stop_exception:{type(exc).__name__}:{exc}"
                current_fault = post_stop_fault
            if current_fault:
                emergency_stop = ZeroStopWindow(
                    lambda: publish(0.0),
                    timeout_s=max(ZERO_STOP_MIN_S, SOURCE_TIMEOUT_MS / 1000.0) + 1.0 / PUBLISH_HZ,
                    publish_hz=PUBLISH_HZ,
                    clock=clock,
                    sleep=sleep,
                    command_lock=command_lock,
                )
                result["post_stop_emergency_zero"] = emergency_stop.run_once()
                break
            remaining = post_stop_end - clock()
            if remaining > 0.0:
                sleep(min(0.05, remaining))

    result["pulse_actual_duration_s"] = (
        max(0.0, pulse_end - pulse_start) if pulse_start is not None and pulse_end is not None else 0.0
    )
    # Loop elapsed time includes sensor sampling after the hard command deadline;
    # publish offsets below measure when nonzero sends actually began/returned.
    result["pulse_loop_elapsed_s"] = result.pop("pulse_actual_duration_s")
    result["nonzero_command_attempt_offsets_s"] = nonzero_attempt_offsets_s
    result["nonzero_command_success_offsets_s"] = nonzero_success_offsets_s
    result["nonzero_command_span_s"] = (
        max(0.0, nonzero_success_offsets_s[-1] - nonzero_success_offsets_s[0])
        if len(nonzero_success_offsets_s) > 1 else 0.0
    )
    result["last_nonzero_command_offset_s"] = (
        nonzero_success_offsets_s[-1] if nonzero_success_offsets_s else None
    )
    result["source_selected_by_get_service"] = tracker.get_source_seen
    result["source_selected_by_common_state"] = tracker.common_source_seen
    result["get_mc_action_locomotion_seen"] = tracker.get_locomotion_seen
    result["common_state_locomotion_seen"] = tracker.common_locomotion_seen
    result["activation_classification"] = tracker.classify()
    result["activation_odometry_delta"] = tracker.odometry_delta()
    result["fault"] = pulse_fault or post_stop_fault
    result["post_stop_observation_s"] = POST_STOP_OBSERVE_S
    result["zero_stop_min_messages"] = ZERO_STOP_MIN_MESSAGES
    result["post_stop_sample_count"] = post_stop_samples
    result["post_stop_fault"] = post_stop_fault
    if pulse_fault:
        result["status"] = "ABORTED"
    elif stop_result["publish_errors"]:
        result["status"] = "ZERO_STOP_FAILED"
    elif stop_result["actual_duration_s"] < SOURCE_TIMEOUT_MS / 1000.0:
        result["status"] = "ZERO_STOP_TOO_SHORT"
    elif stop_result["published_span_s"] < SOURCE_TIMEOUT_MS / 1000.0:
        result["status"] = "ZERO_STOP_DELIVERY_SPAN_TOO_SHORT"
    elif stop_result["zero_messages"] < ZERO_STOP_MIN_MESSAGES:
        result["status"] = "ZERO_STOP_DELIVERY_COUNT_TOO_LOW"
    elif (stop_result["max_inter_message_gap_s"] or 0.0) >= SOURCE_TIMEOUT_MS / 1000.0:
        result["status"] = "ZERO_STOP_DELIVERY_GAP_EXCEEDED_SOURCE_TIMEOUT"
    elif post_stop_fault:
        result["status"] = "POST_STOP_SAFETY_FAILED"
    elif post_stop_samples == 0:
        result["status"] = "POST_STOP_OBSERVATION_MISSING"
    elif result["nonzero_messages"] == 0:
        result["status"] = "NO_ACTIVATION_COMMAND"
    else:
        result["status"] = tracker.classify()
    return result
