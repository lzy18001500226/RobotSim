"""Pure, offline-testable safety logic for the X2 M1 first-motion probe."""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Mapping

from issue80_m1_mode_logic import (
    ACTIVATION_PROBE_DURATION_S,
    ACTIVATION_PROBE_FORWARD_M_S,
    ACTIVATION_PROBE_DURATION_S,
    ACTIVATION_PROBE_FORWARD_M_S,
    LOCOMOTION_CONFIRMED,
    LOCOMOTION_REQUESTED,
    MODE_LD,
    MOTION_ACTIVE,
    STAND_VERIFIED,
    ZERO_AND_STOP_REQUESTED,
    MCLocomotionAbortLifecycle,
    classify_locomotion_activation_readbacks,
)

VELOCITY_SOURCE_NAME = "node.locomotion_velocity"
ACTION_SOURCE_NAME = "rc"  # exact source used by the installed set_mode.py example
SOURCE_PRIORITY = 40
SOURCE_TIMEOUT_MS = 1000
SOURCE_TASK_TERMINAL_FAILURE_STATES = frozenset((2, 3, 4, 5, 6))
SOURCE_TASK_KNOWN_STATES = frozenset((0, 1, 2, 3, 4, 5, 6, 100, 200, 300, 400))
MC_SIM_CONFIG_PATH = (
    "/workspace/raicom2026/mc/bin/mc_param/src/robot/"
    "lx2501_3_t2d5/mc_sim.yaml"
)
FORWARD_COMMAND_M_S = 0.2
PULSE_LIMIT_S = 1.0  # conservative default for unextended first-motion probes
MAX_AUTHORIZED_PULSE_S = 10.0  # current maintainer-reviewed simulation-only ceiling
PUBLISH_HZ = 50.0
COMMAND_QOS_RELIABILITY = "reliable"
COMMAND_QOS_DEPTH = 10
COMMAND_QOS_DURABILITY = "volatile"
COMMAND_PUBLISHER_TOPIC = "/aima/mc/locomotion/velocity"
COMMAND_PUBLISHER_MESSAGE_TYPE = "McLocomotionVelocity"
ZERO_STOP_MIN_S = 1.0
ZERO_STOP_MIN_MESSAGES = math.ceil(SOURCE_TIMEOUT_MS * PUBLISH_HZ / 1000.0)
POST_STOP_OBSERVE_S = 2.5
MAX_EXTRA_POST_STOP_OBSERVE_S = 7.0
MODE_SOURCE_READBACK_SKEW_MAX_S = 0.2  # the configured 5 Hz readback polling interval
BASELINE_WINDOW_S = 1.0
BASELINE_WINDOW_COUNT = 3
BASELINE_DURATION_S = BASELINE_WINDOW_S * BASELINE_WINDOW_COUNT
BASELINE_SAMPLE_PERIOD_S = 1.0 / PUBLISH_HZ

# A second, physics-based stop check. Keep the empirical legacy envelope above
# unchanged; this gate evaluates a final 2 s MuJoCo-state hold after MC readback.
PHYSICAL_STOP_WINDOW_S = 2.0
PHYSICAL_STOP_ROLLING_STEP_S = 0.5
PHYSICAL_STOP_MIN_WINDOW_S = 1.90
PHYSICAL_STOP_MIN_SAMPLES = 20
PHYSICAL_STOP_MAX_SAMPLE_GAP_S = 0.15
PHYSICAL_STOP_MAX_STATE_AGE_S = 0.5
PHYSICAL_STOP_MAX_PLANAR_SPEED_M_S = 0.005
PHYSICAL_STOP_MAX_VERTICAL_SPEED_M_S = 0.005
PHYSICAL_STOP_MAX_PLANAR_DRIFT_M = 0.005
PHYSICAL_STOP_MAX_PLANAR_PATH_M = 0.010
PHYSICAL_STOP_MAX_YAW_DRIFT_RAD = math.radians(2.0)
PHYSICAL_STOP_MAX_YAW_RATE_RAD_S = math.radians(1.0)
PHYSICAL_STOP_MAX_HEIGHT_RANGE_M = 0.005
PHYSICAL_STOP_MAX_TILT_RAD = math.radians(15.0)  # unchanged M1 fall gate
PHYSICAL_STOP_MAX_ROLL_PITCH_RANGE_RAD = math.radians(2.0)
PHYSICAL_STOP_MAX_JOINT_SPEED_RAD_S = 0.05
PHYSICAL_STOP_MAX_JOINT_RANGE_RAD = 0.02
PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD = 0.02
PHYSICAL_STOP_TARGET_STABILITY_WINDOW_S = 1.0
PHYSICAL_STOP_TARGET_STABILITY_RANGE_RAD = 0.01


def validate_b_post_locomotion_observation(
    extra_observation_s: float,
    *,
    extended_b: bool,
    convergence_diagnostic: bool,
) -> dict[str, Any]:
    """Keep legacy B at 5 s; allow the named convergence diagnostic only at the existing 7 s cap."""
    if convergence_diagnostic and not extended_b:
        raise ValueError("post-locomotion convergence observation requires extended waypoint B")
    expected = MAX_EXTRA_POST_STOP_OBSERVE_S if convergence_diagnostic else 5.0
    if not math.isfinite(float(extra_observation_s)) or abs(float(extra_observation_s) - expected) > 1e-9:
        raise ValueError(f"waypoint-B post-stop observation must be exactly {expected:g} seconds")
    return {
        "classification": (
            "SIMULATION_ONLY_POST_LOCOMOTION_STAND_CONVERGENCE_OBSERVATION"
            if convergence_diagnostic else "STANDARD_EXTENDED_B_STOP_OBSERVATION"
        ),
        "extra_zero_observation_s": expected,
        "max_extra_zero_observation_s": MAX_EXTRA_POST_STOP_OBSERVE_S if convergence_diagnostic else 5.0,
        "stop_thresholds_changed": False,
        "composite_rearm_gate_unchanged": True,
    }
# Run64's left arc measured 0.0911 rad at 0.07 rad/s for 4.75 s. A 0.075
# rad/s, 4.5 s arc preserves the measured heading while a 0.5 s warmup gives
# both directions time to confirm source/mode; each side remains 5 s total.
TURN_COMMAND_RAD_S = 0.075
TURN_PULSE_S = 4.5
TURN_RAMP_S = 0.25
TURN_MAX_SUSTAINED_YAW_RATE_RAD_S = 0.28
TURN_MIN_HEADING_CHANGE_RAD = math.radians(5.0)
TURN_MAX_PLANAR_DISPLACEMENT_M = 0.05
TURN_FORWARD_COMMAND_M_S = 0.12
# Run63 confirmed same-sample source+LOCOMOTION at 0.201 s; retain 50 ms margin.
TURN_FORWARD_WARMUP_S = 0.5
TURN_MAX_COMMAND_WINDOW_S = 5.0
TURN_YAW_RATE_WINDOW_S = 0.4

MODE_STAND = "STAND_DEFAULT"
MODE_LOCOMOTION = "LOCOMOTION_DEFAULT"
ALLOWED_MODES = frozenset((MODE_STAND, MODE_LOCOMOTION))
MC_ACTION_STATUSES = frozenset((0, 100, 200))  # IDLE, RUNNING, TRANSITION


def common_action_status_diagnostic(value: Any) -> dict[str, Any]:
    """Classify the CommonState action status without borrowing unrelated enums."""
    try:
        raw_value = int(value)
    except (TypeError, ValueError):
        raw_value = value
    names = {0: "IDLE", 100: "RUNNING", 200: "TRANSITION"}
    if isinstance(raw_value, int) and raw_value in names:
        return {
            "classification": names[raw_value],
            "field": "McCommonState.action_info.status",
            "raw_value": raw_value,
            "installed_message_constant_declared": True,
            "installed_documentation_version_match": None,
            "motion_rearm_allowed": False,
        }
    if raw_value == 300:
        return {
            "classification": "UNINTERRUPTIBLE_TRANSITION",
            "field": "McCommonState.action_info.status",
            "raw_value": raw_value,
            "source_semantics": "AimDK_X2_1.1.0 mc_status documentation",
            "installed_message_constant_declared": False,
            "installed_documentation_version_match": False,
            "runtime_behavior_corroborated": True,
            "corroborating_observation": (
                "2026-10-24 run emitted 300 during LOCOMOTION-to-STAND; a later CommonState and "
                "GetMcAction reported STAND_DEFAULT/status 100, FSM 4, body 1"
            ),
            "motion_rearm_allowed": False,
        }
    return {
        "classification": "UNKNOWN_RUNTIME_STATUS",
        "field": "McCommonState.action_info.status",
        "raw_value": raw_value,
        "installed_message_constant_declared": False,
        "installed_documentation_version_match": False,
        "motion_rearm_allowed": False,
    }


class McFsmStateCode(IntEnum):
    """Wire codes verified in the pinned MC state library, not stale SDK constants."""

    UNKNOWN = 0
    PREPARE = 1
    STARTING = 2
    FINISH = 3
    STABLE = 4
    MOVING = 5
    SAFE = 6
    SPECIAL = 7
    TEST = 8


# SDK message constants disagree with this MC build; its uint32 wire field
# remains compatible. StateManager::RunLoop maps state-action categories to
# 1..8 at offsets 0xd646e..0xd6570; GetCurrentState returns that integer and
# CommonStatePublisher publishes it directly. Do not use this mapping on
# another binary build. Original interface provenance is retained below.
MC_FSM_INTERFACE_SOURCE_PATH = "interface/robot/mc/data/msg/common/McFsmState.msg"
MC_FSM_INTERFACE_SOURCE_SHA256 = (
    "da8a0370cba82d5d2955004a9b68489ee150f557e4c520dcd085bf58600cf6cf"
)
MC_FSM_RUNTIME_LIBRARY_SHA256 = {
    "libmc_manager_state.so":
        "554e3e58752f3c640fa0074d7121663ebeda33037acd73a8d9234e94174c33aa",
    "libmc_module_main_publisher.so":
        "c159d55d6cf1c3217dc9fb4f655c25125b33204ee36432d98e5cc180653e72b9",
}
MC_FSM_ACTIVE_STATES = frozenset((
    int(McFsmStateCode.STARTING),
    int(McFsmStateCode.STABLE),
    int(McFsmStateCode.MOVING),
))
MC_FSM_STABLE = int(McFsmStateCode.STABLE)
MC_FSM_SAFE = int(McFsmStateCode.SAFE)


def verify_mc_fsm_runtime_contract(library_dir: str | Path) -> dict[str, str]:
    """Fail closed before motion if the binary-backed decoder no longer applies."""
    actual = {
        name: hashlib.sha256((Path(library_dir) / name).read_bytes()).hexdigest()
        for name in MC_FSM_RUNTIME_LIBRARY_SHA256
    }
    if actual != MC_FSM_RUNTIME_LIBRARY_SHA256:
        raise ValueError("MC FSM binary contract mismatch; motion prohibited")
    return actual


BODY_POSE_STAND = 1
REQUIRED_STATE_KEYS = ("odom", "imu", "common", "leg_state", "leg_command", "get_mode", "get_source")
ZERO_STOP_INTERPRETATION = (
    "Publisher calls returned without exception; this does not confirm MC receipt or that MC stopped."
)
WATCHDOG_LIMITATION = (
    "threading.Timer runs in this Python process and can request zero while the process is alive; "
    "it cannot publish after this process crashes or is killed."
)
MC_ROS2_NODE_BASE_NAME = "mc_ros2_node"
MC_RUNTIME_EXECUTABLE = "aima-mc-app-main"
MC_VELOCITY_SUBSCRIBER_NAME = "locomotion_velocity"
MC_VELOCITY_MESSAGE_TYPE = "aimdk_msgs/msg/McLocomotionVelocity"
MC_VELOCITY_TOPIC = "/aima/mc/locomotion/velocity"
DDS_QOS_OVERRIDE_ENV_KEYS = (
    "FASTDDS_DEFAULT_PROFILES_FILE",
    "FASTRTPS_DEFAULT_PROFILES_FILE",
    "RMW_FASTRTPS_USE_QOS_FROM_XML",
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class VerifiedMcRuntimeIdentity:
    """MC ROS node identity tied to its configured name and live runtime PID."""

    configured_node_name: str
    process_pid: int
    process_executable: str
    mc_config_sha256: str
    startup_log_sha256: str

    @property
    def node_name(self) -> str:
        return f"{self.configured_node_name}{self.process_pid}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured_node_name": self.configured_node_name,
            "process_pid": self.process_pid,
            "process_executable": self.process_executable,
            "graph_node_name": self.node_name,
            "mc_config_sha256": self.mc_config_sha256,
            "startup_log_sha256": self.startup_log_sha256,
            "provenance": (
                "configured AimRT ros2_plugin base name + Process PID from this run's MC startup log, "
                "cross-checked against the live /proc executable and config argv"
            ),
        }


def load_verified_mc_runtime_identity(
    mc_config_path: str | Path,
    mc_startup_log_path: str | Path,
    *,
    proc_root: str | Path = "/proc",
) -> VerifiedMcRuntimeIdentity:
    """Derive the MC graph name only from the isolated runtime's own process evidence.

    The pinned runtime config names its ROS 2 node ``mc_ros2_node``. In the
    reviewed run, AimRT reported process PID 245 and the ROS graph exposed
    ``mc_ros2_node245``. Require that exact configured-name + live-process-PID
    relationship; never accept arbitrary prefix matches.
    """
    import yaml

    config_path = Path(mc_config_path).resolve(strict=True)
    config_bytes = config_path.read_bytes()
    try:
        config = yaml.safe_load(config_bytes)
    except yaml.YAMLError as exc:
        raise ValueError("MC config is not valid YAML") from exc
    try:
        plugins = config["aimrt"]["plugin"]["plugins"]
    except (TypeError, KeyError) as exc:
        raise ValueError("MC config lacks aimrt.plugin.plugins") from exc
    ros2_plugins = [
        plugin for plugin in plugins
        if isinstance(plugin, Mapping) and plugin.get("name") == "ros2_plugin"
    ] if isinstance(plugins, list) else []
    if len(ros2_plugins) != 1:
        raise ValueError(f"expected one configured MC ros2_plugin, found {len(ros2_plugins)}")
    options = ros2_plugins[0].get("options")
    configured_node_name = options.get("node_name") if isinstance(options, Mapping) else None
    if configured_node_name != MC_ROS2_NODE_BASE_NAME:
        raise ValueError(
            f"unexpected MC ros2_plugin node_name: {configured_node_name!r}"
        )

    startup_log_path = Path(mc_startup_log_path)
    startup_log_bytes = startup_log_path.read_bytes()
    startup_log = _ANSI_ESCAPE_RE.sub(
        "", startup_log_bytes.decode("utf-8", errors="replace")
    )
    runtime_node_names = set(re.findall(
        r"^[ \t]*node_name:[ \t]*([A-Za-z0-9_-]+)[ \t]*$",
        startup_log,
        flags=re.MULTILINE,
    ))
    if configured_node_name not in runtime_node_names:
        raise ValueError("MC startup log does not report the configured ros2_plugin node name")
    pids = set(re.findall(
        r"^[ \t]*\|[ \t]*Process PID[ \t]*\|[ \t]*(\d+)[ \t]*\|[ \t]*$",
        startup_log,
        flags=re.MULTILINE,
    ))
    executables = set(re.findall(
        r"^[ \t]*\|[ \t]*Executable Path[ \t]*\|[ \t]*(.*?)\s*\|[ \t]*$",
        startup_log,
        flags=re.MULTILINE,
    ))
    if len(pids) != 1 or len(executables) != 1:
        raise ValueError(
            f"MC startup log must identify one process PID and executable; got {pids!r}, {executables!r}"
        )
    process_pid = int(next(iter(pids)))
    if process_pid <= 1:
        raise ValueError("MC startup log contains an invalid process PID")
    logged_executable = next(iter(executables))
    if Path(logged_executable).name != MC_RUNTIME_EXECUTABLE:
        raise ValueError(f"unexpected MC executable in startup log: {logged_executable!r}")

    proc_dir = Path(proc_root) / str(process_pid)
    try:
        process_executable = os.readlink(proc_dir / "exe")
        argv = [os.fsdecode(item) for item in (proc_dir / "cmdline").read_bytes().split(b"\0") if item]
    except OSError as exc:
        raise ValueError(f"MC startup PID {process_pid} is not a readable live process") from exc
    if Path(process_executable).name != MC_RUNTIME_EXECUTABLE:
        raise ValueError(f"PID {process_pid} is not the installed MC executable")
    if not argv or Path(argv[0]).name != MC_RUNTIME_EXECUTABLE:
        raise ValueError(f"PID {process_pid} argv does not identify the installed MC executable")
    config_args = [
        arg.split("=", 1)[1]
        for arg in argv
        if arg.startswith("--cfg_file_path=")
    ]
    if len(config_args) != 1:
        raise ValueError("live MC process has no unique config path argument")
    process_config_path = Path(config_args[0]).resolve(strict=False)
    if process_config_path != config_path:
        raise ValueError("live MC process is not running with the verified MC config")

    return VerifiedMcRuntimeIdentity(
        configured_node_name=str(configured_node_name),
        process_pid=process_pid,
        process_executable=process_executable,
        mc_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        startup_log_sha256=hashlib.sha256(startup_log_bytes).hexdigest(),
    )


def _verified_mc_identity_error(identity: VerifiedMcRuntimeIdentity | None) -> str | None:
    if not isinstance(identity, VerifiedMcRuntimeIdentity):
        return "mc_runtime_identity_unverified"
    if (
        identity.configured_node_name != MC_ROS2_NODE_BASE_NAME
        or isinstance(identity.process_pid, bool)
        or not isinstance(identity.process_pid, int)
        or identity.process_pid <= 1
        or not isinstance(identity.process_executable, str)
    ):
        return "mc_runtime_identity_invalid"
    if identity.node_name != f"{MC_ROS2_NODE_BASE_NAME}{identity.process_pid}":
        return "mc_runtime_node_suffix_not_process_pid"
    if Path(identity.process_executable).name != MC_RUNTIME_EXECUTABLE:
        return "mc_runtime_executable_mismatch"
    if not all(
        re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in (identity.mc_config_sha256, identity.startup_log_sha256)
    ):
        return "mc_runtime_provenance_digest_missing"
    return None


def capture_dds_qos_override_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Record only whether known Fast DDS XML QoS override variables are active."""
    source = os.environ if environment is None else environment
    profile_status = {
        key: "set" if str(source.get(key, "")).strip() else "unset"
        for key in DDS_QOS_OVERRIDE_ENV_KEYS[:2]
    }
    xml_setting = str(source.get(DDS_QOS_OVERRIDE_ENV_KEYS[2], "")).strip().lower()
    profile_status[DDS_QOS_OVERRIDE_ENV_KEYS[2]] = (
        "unset" if not xml_setting
        else "false" if xml_setting in ("0", "false", "no", "off")
        else "true" if xml_setting in ("1", "true", "yes", "on")
        else "invalid"
    )
    return profile_status


def _dds_qos_override_environment_error(
    evidence: Mapping[str, Any] | None,
) -> str | None:
    if not isinstance(evidence, Mapping) or set(evidence) != set(DDS_QOS_OVERRIDE_ENV_KEYS):
        return "dds_qos_override_environment_unverified"
    if any(evidence.get(key) != "unset" for key in DDS_QOS_OVERRIDE_ENV_KEYS[:2]):
        return "dds_qos_xml_profile_override_active"
    if evidence.get(DDS_QOS_OVERRIDE_ENV_KEYS[2]) not in ("unset", "false"):
        return "dds_qos_xml_override_active_or_unknown"
    return None


def load_verified_mc_velocity_subscriber_provenance(
    mc_config_path: str | Path,
    mc_startup_log_path: str | Path,
    mc_identity: VerifiedMcRuntimeIdentity,
) -> dict[str, Any]:
    """Verify the one configured MC command subscription and its startup registration.

    This proves which route and message type the pinned MC process configured.
    It deliberately does not infer the effective DataReader history or depth.
    """
    import yaml

    identity_fault = _verified_mc_identity_error(mc_identity)
    if identity_fault:
        raise ValueError(identity_fault)
    config_path = Path(mc_config_path).resolve(strict=True)
    config_bytes = config_path.read_bytes()
    startup_path = Path(mc_startup_log_path).resolve(strict=True)
    startup_bytes = startup_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != mc_identity.mc_config_sha256:
        raise ValueError("MC subscriber config hash differs from verified live process identity")
    if hashlib.sha256(startup_bytes).hexdigest() != mc_identity.startup_log_sha256:
        raise ValueError("MC subscriber startup log hash differs from verified live process identity")
    try:
        config = yaml.safe_load(config_bytes)
    except yaml.YAMLError as exc:
        raise ValueError("MC subscriber config is not valid YAML") from exc
    try:
        aimrt = config["aimrt"]
        plugin_list = aimrt["plugin"]["plugins"]
        channel = aimrt["channel"]
        backend_list = channel["backends"]
        route_list = channel["sub_topics_options"]
        subscriber_list = config["MotionControlModule"]["SubscriberManager"]["subscriber_cfg_list"]
    except (TypeError, KeyError) as exc:
        raise ValueError("MC config lacks the required plugin/channel/subscriber provenance") from exc
    if not all(isinstance(value, list) for value in (plugin_list, backend_list, route_list, subscriber_list)):
        raise ValueError("MC plugin, backend, route, and subscriber entries must be lists")

    ros2_plugins = [
        item for item in plugin_list
        if isinstance(item, Mapping) and item.get("name") == "ros2_plugin"
    ]
    if len(ros2_plugins) != 1:
        raise ValueError("MC config must contain exactly one ros2_plugin")
    plugin_options = ros2_plugins[0].get("options")
    if not isinstance(plugin_options, Mapping):
        raise ValueError("MC ros2_plugin has no options mapping")
    if plugin_options.get("node_name") != mc_identity.configured_node_name:
        raise ValueError("MC ros2_plugin node name differs from verified process identity")
    if plugin_options.get("namespace") not in (None, "") or plugin_options.get("node_namespace") not in (None, ""):
        raise ValueError("MC ROS node namespace is explicitly configured and unsupported here")

    ros2_backends = [
        item for item in backend_list
        if isinstance(item, Mapping) and item.get("type") == "ros2"
    ]
    if len(ros2_backends) != 1:
        raise ValueError("MC config must contain exactly one ROS 2 channel backend")
    backend_options = ros2_backends[0].get("options", {})
    if not isinstance(backend_options, Mapping):
        raise ValueError("MC ROS 2 backend options are malformed")
    if any("qos" in str(key).lower() and "pub" not in str(key).lower() for key in backend_options):
        raise ValueError("MC ROS 2 backend contains an unsupported subscriber QoS override")
    backend_sub_routes = backend_options.get("sub_topics_options", [])
    if backend_sub_routes is None:
        backend_sub_routes = []
    if not isinstance(backend_sub_routes, list):
        raise ValueError("MC ROS 2 backend subscriber routes are malformed")
    matching_backend_routes = [
        item for item in backend_sub_routes
        if isinstance(item, Mapping) and item.get("topic_name") == MC_VELOCITY_TOPIC
    ]
    if len(matching_backend_routes) > 1 or any("qos" in item for item in matching_backend_routes):
        raise ValueError("MC velocity subscriber has a duplicate or explicit backend QoS override")

    matching_routes = [
        item for item in route_list
        if isinstance(item, Mapping) and item.get("topic_name") == MC_VELOCITY_TOPIC
    ]
    if len(matching_routes) != 1:
        raise ValueError("MC config must define exactly one velocity subscriber ROS 2 route")
    route = matching_routes[0]
    if route.get("enable_backends") != ["ros2"] or "qos" in route:
        raise ValueError("MC velocity subscriber route is disabled, ambiguous, or has a QoS override")

    matching_subscribers = [
        item for item in subscriber_list
        if isinstance(item, Mapping)
        and (item.get("name") == MC_VELOCITY_SUBSCRIBER_NAME or item.get("topic") == MC_VELOCITY_TOPIC)
    ]
    if len(matching_subscribers) != 1:
        raise ValueError("MC config must define exactly one velocity subscriber entry")
    subscriber = matching_subscribers[0]
    if (
        subscriber.get("name") != MC_VELOCITY_SUBSCRIBER_NAME
        or subscriber.get("topic") != MC_VELOCITY_TOPIC
        or subscriber.get("enable") is not True
        or "qos" in subscriber
    ):
        raise ValueError("MC velocity subscriber entry is disabled, mismatched, or has a QoS override")

    startup_log = _ANSI_ESCAPE_RE.sub("", startup_bytes.decode("utf-8", errors="replace"))
    created_lines = [
        line for line in startup_log.splitlines()
        if f"Create subscriber: {MC_VELOCITY_SUBSCRIBER_NAME} succeeded, topic: {MC_VELOCITY_TOPIC}" in line
    ]
    type_row = re.compile(
        rf"^\|\s*{re.escape(MC_VELOCITY_TOPIC)}\s*\|\s*ros2:{re.escape(MC_VELOCITY_MESSAGE_TYPE)}\s*"
        r"\|\s*MotionControlModule\s*\|\s*ros2\s*\|\s*monitor\s*\|$"
    )
    type_rows = [line for line in startup_log.splitlines() if type_row.search(line.strip())]
    if len(created_lines) != 1 or len(type_rows) != 1:
        raise ValueError("MC startup log does not uniquely confirm the velocity subscriber and message type")

    return {
        "subscriber_name": MC_VELOCITY_SUBSCRIBER_NAME,
        "node_name": mc_identity.node_name,
        "topic_name": MC_VELOCITY_TOPIC,
        "message_type": MC_VELOCITY_MESSAGE_TYPE,
        "subscriber_enabled": True,
        "ros2_route_enabled": True,
        "startup_subscriber_creation_confirmed": True,
        "startup_type_registration_confirmed": True,
        "config_namespace_option": None,
        "qos_override_present": False,
        "configured_history_depth": "not specified in audited subscriber configuration",
        "actual_reader_history_depth": "UNVERIFIED",
        "mc_config_sha256": mc_identity.mc_config_sha256,
        "startup_log_sha256": mc_identity.startup_log_sha256,
        "evidence_basis": "verified MC process identity, exact config route, startup subscriber log, and type registry row",
    }


def _verified_mc_velocity_subscriber_provenance_error(
    provenance: Mapping[str, Any] | None,
    mc_identity: VerifiedMcRuntimeIdentity,
) -> str | None:
    if not isinstance(provenance, Mapping):
        return "mc_velocity_subscriber_configuration_provenance_missing"
    required = {
        "subscriber_name": MC_VELOCITY_SUBSCRIBER_NAME,
        "node_name": mc_identity.node_name,
        "topic_name": MC_VELOCITY_TOPIC,
        "message_type": MC_VELOCITY_MESSAGE_TYPE,
        "subscriber_enabled": True,
        "ros2_route_enabled": True,
        "startup_subscriber_creation_confirmed": True,
        "startup_type_registration_confirmed": True,
        "config_namespace_option": None,
        "qos_override_present": False,
        "configured_history_depth": "not specified in audited subscriber configuration",
        "actual_reader_history_depth": "UNVERIFIED",
        "mc_config_sha256": mc_identity.mc_config_sha256,
        "startup_log_sha256": mc_identity.startup_log_sha256,
    }
    if any(provenance.get(key) != value for key, value in required.items()):
        return "mc_velocity_subscriber_configuration_provenance_mismatch"
    return None


def _mc_velocity_endpoint_metadata_error(
    endpoint_metadata: list[Mapping[str, Any]] | None,
    mc_identity: VerifiedMcRuntimeIdentity,
) -> str | None:
    if not isinstance(endpoint_metadata, list) or len(endpoint_metadata) != 1:
        return "mc_velocity_subscriber_endpoint_metadata_missing_or_ambiguous"
    endpoint = endpoint_metadata[0]
    if not isinstance(endpoint, Mapping):
        return "mc_velocity_subscriber_endpoint_metadata_malformed"
    if endpoint.get("node_name") != mc_identity.node_name:
        return "mc_velocity_subscriber_endpoint_identity_mismatch"
    if endpoint.get("queried_topic_name") != MC_VELOCITY_TOPIC:
        return "mc_velocity_subscriber_endpoint_topic_mismatch"
    if endpoint.get("node_namespace") != "/":
        return "mc_velocity_subscriber_endpoint_namespace_mismatch"
    if endpoint.get("topic_type") != MC_VELOCITY_MESSAGE_TYPE:
        return "mc_velocity_subscriber_endpoint_message_type_mismatch"
    return None


def mc_subscriber_history_depth_evidence(
    history: str,
    depth: int,
    *,
    rmw_implementation: str,
    ros_distro: str,
) -> dict[str, Any]:
    """Describe discovery metadata without claiming to query the remote DataReader."""
    is_discovery_gap = history == "unknown" and depth == 0
    return {
        "discovery_reported": {"history": history, "depth": depth},
        "classification": (
            "FASTDDS_HUMBLE_DISCOVERY_HISTORY_DEPTH_GAP"
            if is_discovery_gap and rmw_implementation == "rmw_fastrtps_cpp" and ros_distro == "humble"
            else "GRAPH_HISTORY_DEPTH_REPORTED"
        ),
        "actual_mc_datareader_history_depth": "UNVERIFIED",
        "exception_scope": "one verified MC locomotion_velocity endpoint only" if is_discovery_gap else None,
        "command_delivery_proven": False,
    }


def _local_command_publisher_error(
    *,
    publisher_local_identity: Mapping[str, Any] | None,
    publisher_configured_qos: Mapping[str, Any] | None,
    publisher_api_qos: Mapping[str, Any] | None,
    own_node_name: str,
) -> str | None:
    """Verify this runner's command publisher independently of DDS graph metadata."""
    if not isinstance(publisher_local_identity, Mapping):
        return "local_command_publisher_identity_unverified"
    if (
        publisher_local_identity.get("node_name") != own_node_name
        or publisher_local_identity.get("topic_name") != COMMAND_PUBLISHER_TOPIC
        or publisher_local_identity.get("message_type") != COMMAND_PUBLISHER_MESSAGE_TYPE
        or publisher_local_identity.get("is_expected_message_type") is not True
        or publisher_local_identity.get("is_owned_publisher") is not True
        or publisher_local_identity.get("node_topic_publisher_count") != 1
    ):
        return "local_command_publisher_identity_mismatch"

    if not isinstance(publisher_configured_qos, Mapping) or not isinstance(publisher_api_qos, Mapping):
        return "local_command_publisher_qos_unverified"
    expected = {
        "history": "keep_last",
        "depth": COMMAND_QOS_DEPTH,
        "reliability": COMMAND_QOS_RELIABILITY,
        "durability": COMMAND_QOS_DURABILITY,
    }
    for source_name, profile in (
        ("configured", publisher_configured_qos),
        ("publisher_api", publisher_api_qos),
    ):
        if any(profile.get(key) != value for key, value in expected.items()):
            return f"local_command_publisher_{source_name}_qos_mismatch:{dict(profile)!r}"
    if any(publisher_configured_qos.get(key) != publisher_api_qos.get(key) for key in expected):
        return "local_command_publisher_qos_profile_disagreement"
    return None


def forward_projected_displacement(start_pose: Mapping[str, Any], end_pose: Mapping[str, Any]) -> float:
    """Project world-frame odometry displacement onto the start heading."""
    try:
        x0, y0, yaw0 = (float(start_pose[key]) for key in ("x", "y", "yaw"))
        x1, y1 = float(end_pose["x"]), float(end_pose["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("odometry pose must contain numeric x, y, and yaw") from exc
    if not all(math.isfinite(value) for value in (x0, y0, yaw0, x1, y1)):
        raise ValueError("odometry pose contains a nonfinite coordinate")
    dx, dy = x1 - x0, y1 - y0
    return dx * math.cos(yaw0) + dy * math.sin(yaw0)


def point_goal_forward_speed_m_s(pose: Mapping[str, Any], *, start_yaw_rad: float) -> float:
    """Project odometry twist into the goal's along-track axis with its declared frame."""
    try:
        vx, vy = float(pose["vx"]), float(pose["vy"])
        yaw = float(pose["yaw"])
        start_yaw = float(start_yaw_rad)
        twist_frame = str(pose["twist_frame_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("point-goal speed requires vx, vy, yaw, and twist_frame_id") from exc
    if not all(math.isfinite(value) for value in (vx, vy, yaw, start_yaw)):
        raise ValueError("point-goal speed pose values must be finite")
    if twist_frame == "base_link":
        relative_yaw = yaw - start_yaw
        return vx * math.cos(relative_yaw) - vy * math.sin(relative_yaw)
    if twist_frame in ("odom", "map", "world"):
        return vx * math.cos(start_yaw) + vy * math.sin(start_yaw)
    raise ValueError(f"unsupported point-goal twist frame: {twist_frame!r}")


def point_goal_heading_error_rad(
    *,
    current_yaw_rad: float,
    goal_dx_m: float,
    goal_dy_m: float,
) -> float:
    """Return the wrapped bearing error from the current pose to the goal."""
    values = (current_yaw_rad, goal_dx_m, goal_dy_m)
    try:
        current_yaw, dx, dy = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("point-goal heading inputs must be numeric") from exc
    if not all(math.isfinite(value) for value in (current_yaw, dx, dy)):
        raise ValueError("point-goal heading inputs must be finite")
    bearing = math.atan2(dy, dx)
    return math.atan2(math.sin(bearing - current_yaw), math.cos(bearing - current_yaw))


def point_goal_pose_heading_error_rad(
    *,
    current_yaw_rad: float,
    goal_dx_m: float,
    goal_dy_m: float,
    target_yaw_rad: float | None = None,
) -> float:
    """Use position bearing for point goals and explicit yaw error for pose goals."""
    if target_yaw_rad is None:
        return point_goal_heading_error_rad(
            current_yaw_rad=current_yaw_rad,
            goal_dx_m=goal_dx_m,
            goal_dy_m=goal_dy_m,
        )
    try:
        current_yaw = float(current_yaw_rad)
        target_yaw = float(target_yaw_rad)
    except (TypeError, ValueError) as exc:
        raise ValueError("point-goal pose heading inputs must be numeric") from exc
    if not math.isfinite(current_yaw) or not math.isfinite(target_yaw):
        raise ValueError("point-goal pose heading inputs must be finite")
    return math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))


def classify_heading_goal_measurement(
    *,
    start_yaw_rad: float,
    final_yaw_rad: float,
    target_yaw_rad: float,
    requested_heading_change_rad: float,
    minimum_measured_change_rad: float,
    final_yaw_tolerance_rad: float,
) -> dict[str, Any]:
    """Classify measured final yaw against a bounded odom-frame pose goal."""
    values = (
        start_yaw_rad, final_yaw_rad, target_yaw_rad, requested_heading_change_rad,
        minimum_measured_change_rad, final_yaw_tolerance_rad,
    )
    try:
        start, final, target, requested, minimum, tolerance = tuple(
            float(value) for value in values
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("heading-goal values must be numeric") from exc
    if not all(math.isfinite(value) for value in (start, final, target, requested, minimum, tolerance)):
        raise ValueError("heading-goal values must be finite")
    if minimum < 0.0 or tolerance <= 0.0:
        raise ValueError("heading-goal acceptance tolerances are invalid")

    measured_change = math.atan2(math.sin(final - start), math.cos(final - start))
    final_yaw_error = math.atan2(math.sin(target - final), math.cos(target - final))
    required = abs(requested) > 1e-9
    measured_in_direction = (
        not required
        or (abs(measured_change) >= minimum and measured_change * requested > 0.0)
    )
    final_yaw_within_tolerance = not required or abs(final_yaw_error) <= tolerance
    return {
        "required": required,
        "requested_heading_change_rad": requested,
        "measured_heading_change_rad": measured_change,
        "target_yaw_rad": target,
        "final_yaw_error_rad": final_yaw_error,
        "minimum_measured_change_rad": minimum,
        "final_yaw_tolerance_rad": tolerance,
        "measured_heading_change_in_requested_direction": measured_in_direction,
        "final_yaw_within_tolerance": final_yaw_within_tolerance,
        "status": "PASS" if measured_in_direction and final_yaw_within_tolerance else "FAIL",
    }


def point_goal_command_from_bearing_error(
    heading_error_rad: float,
    *,
    cruise_forward_m_s: float = 0.14,
    steering_forward_m_s: float = 0.14,
    heading_gain: float = 0.8,
    maximum_angular_rad_s: float = 0.075,
    steering_threshold_rad: float = 0.005,
    turn_warmup_active: bool = False,
    warmup_forward_m_s: float = TURN_FORWARD_COMMAND_M_S,
) -> tuple[float, float]:
    """Apply the bounded point-goal speed profile and optional measured turn warmup."""
    values = (
        heading_error_rad,
        cruise_forward_m_s,
        steering_forward_m_s,
        heading_gain,
        maximum_angular_rad_s,
        steering_threshold_rad,
        warmup_forward_m_s,
    )
    try:
        error, cruise, steering, gain, maximum_angular, threshold, warmup_forward = tuple(
            float(value) for value in values
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("point-goal command inputs must be numeric") from exc
    if not all(math.isfinite(value) for value in (error, cruise, steering, gain, maximum_angular, threshold, warmup_forward)):
        raise ValueError("point-goal command inputs must be finite")
    if (cruise <= 0.0 or steering <= 0.0 or gain < 0.0 or maximum_angular <= 0.0
            or threshold < 0.0 or warmup_forward <= 0.0 or warmup_forward > FORWARD_COMMAND_M_S):
        raise ValueError("point-goal command bounds are invalid")
    if turn_warmup_active:
        return warmup_forward, 0.0
    forward = steering if abs(error) > threshold else cruise
    angular = max(-maximum_angular, min(maximum_angular, gain * error))
    return forward, angular


def point_goal_angular_taper_factor(
    predicted_settled_error_m: float,
    *,
    taper_start_error_m: float = 0.25,
    final_tolerance_m: float = 0.05,
) -> float:
    """Fade angular input over the terminal approach when final yaw is unconstrained."""
    try:
        error, taper_start, tolerance = tuple(
            float(value) for value in (
                predicted_settled_error_m, taper_start_error_m, final_tolerance_m,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("point-goal angular taper inputs must be numeric") from exc
    if not all(math.isfinite(value) for value in (error, taper_start, tolerance)):
        raise ValueError("point-goal angular taper inputs must be finite")
    if tolerance <= 0.0 or taper_start <= tolerance:
        raise ValueError("point-goal angular taper must begin outside the final tolerance")
    if error >= taper_start:
        return 1.0
    if error <= tolerance:
        return 0.0
    return (error - tolerance) / (taper_start - tolerance)


def point_goal_pose_angular_taper_factor(
    predicted_settled_error_m: float,
    *,
    has_heading_target: bool,
    taper_start_error_m: float = 0.25,
    final_tolerance_m: float = 0.05,
) -> float:
    """Keep yaw authority for pose goals; taper only position-only point goals."""
    if has_heading_target:
        return 1.0
    return point_goal_angular_taper_factor(
        predicted_settled_error_m,
        taper_start_error_m=taper_start_error_m,
        final_tolerance_m=final_tolerance_m,
    )


def build_forward_point_goal(
    start_pose: Mapping[str, Any], *, distance_m: float, heading_change_rad: float = 0.0,
) -> dict[str, Any]:
    """Build an odom-frame point/yaw target from one measured start pose."""
    if start_pose.get("frame_id") != "odom" or start_pose.get("child_frame_id") != "base_link":
        raise ValueError("point-goal start pose must be odom -> base_link")
    try:
        x, y, yaw = (float(start_pose[key]) for key in ("x", "y", "yaw"))
        distance = float(distance_m)
        heading_change = float(heading_change_rad)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("point-goal inputs must contain numeric pose and distance") from exc
    if not all(math.isfinite(value) for value in (x, y, yaw, distance, heading_change)):
        raise ValueError("point-goal inputs must be finite")
    if distance <= 0.0:
        raise ValueError("point-goal distance must be positive")
    path_yaw = math.atan2(math.sin(yaw + heading_change), math.cos(yaw + heading_change))
    heading_x, heading_y = math.cos(path_yaw), math.sin(path_yaw)
    goal = {
        "frame_id": "odom",
        "x": x + distance * heading_x,
        "y": y + distance * heading_y,
        "yaw": path_yaw,
        "heading_change_rad": heading_change,
    }
    return goal


def speed_scaled_point_goal_stop_advance_m(
    measured_forward_speed_m_s: float,
    *,
    endpoint_response_intercept_m: float,
    endpoint_response_time_s: float,
    maximum_forward_speed_m_s: float = 0.2,
) -> float:
    """Estimate a bounded stop advance from measured speed and observed endpoint response.

    The affine calibration is experiment-specific, not a general braking
    model. The speed cap must remain the existing validated motion envelope.
    """
    try:
        speed = float(measured_forward_speed_m_s)
        response_intercept = float(endpoint_response_intercept_m)
        response_time = float(endpoint_response_time_s)
        maximum_speed = float(maximum_forward_speed_m_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("point-goal stop inputs must be numeric") from exc
    if not all(math.isfinite(value) for value in (speed, response_intercept, response_time, maximum_speed)):
        raise ValueError("point-goal stop inputs must be finite")
    if response_intercept < 0.0 or response_time < 0.0 or maximum_speed <= 0.0:
        raise ValueError("point-goal response time and speed envelope are invalid")
    if abs(speed) > maximum_speed:
        raise ValueError("measured forward speed exceeds the existing point-goal envelope")
    maximum_advance = response_intercept + maximum_speed * response_time
    if maximum_advance > 0.05:
        raise ValueError("derived point-goal stop advance exceeds the existing 50 mm tolerance")
    return response_intercept + max(0.0, speed) * response_time


def point_goal_stop_reason(
    *,
    along_track_progress_m: float,
    lateral_error_m: float,
    goal_distance_m: float,
    dynamic_stop_progress_m: float,
    final_tolerance_m: float,
    predicted_settled_tolerance_m: float | None = None,
    maximum_lateral_error_m: float,
    maximum_overshoot_m: float,
) -> str | None:
    """Stop on a crossed line or prediction, retaining separate safety tolerances."""
    values = (
        along_track_progress_m,
        lateral_error_m,
        goal_distance_m,
        dynamic_stop_progress_m,
        final_tolerance_m,
        maximum_lateral_error_m,
        maximum_overshoot_m,
    )
    try:
        numbers = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("point-goal stop decision inputs must be numeric") from exc
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("point-goal stop decision inputs must be finite")
    progress, lateral, goal, stop_progress, final_tolerance, lateral_limit, overshoot_limit = numbers
    prediction_tolerance = (
        final_tolerance if predicted_settled_tolerance_m is None
        else float(predicted_settled_tolerance_m)
    )
    if (
        goal <= 0.0 or stop_progress < goal or final_tolerance <= 0.0
        or not math.isfinite(prediction_tolerance) or prediction_tolerance <= 0.0
        or lateral_limit <= 0.0 or overshoot_limit < 0.0
    ):
        raise ValueError("point-goal stop decision limits are invalid")
    if abs(lateral) > lateral_limit:
        return "lateral_error_limit"
    if progress > goal + overshoot_limit:
        return "hard_overshoot_guard"
    if progress >= stop_progress:
        return "speed_scaled_stop_line_crossed"
    predicted_settled_error = math.hypot(stop_progress - progress, lateral)
    if predicted_settled_error <= prediction_tolerance:
        return "goal_approach_predicted_settle_within_tolerance"
    return None


def stale_readback_keys(snapshot: Mapping[str, Any], max_age_s: float) -> tuple[str, ...]:
    """List stale or absent asynchronous MC readbacks without relaxing their age limit."""
    try:
        max_age = float(max_age_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("readback maximum age must be numeric") from exc
    if not math.isfinite(max_age) or max_age < 0.0:
        raise ValueError("readback maximum age must be finite and nonnegative")
    ages = snapshot.get("state_age_s")
    if not isinstance(ages, Mapping):
        return ("get_mode", "get_source")
    stale: list[str] = []
    for key in ("get_mode", "get_source"):
        age = ages.get(key)
        if (
            not isinstance(age, (int, float))
            or not math.isfinite(float(age))
            or not 0.0 <= float(age) <= max_age
        ):
            stale.append(key)
    return tuple(stale)


def readback_refresh_needed(snapshot: Mapping[str, Any], max_age_s: float) -> bool:
    """Request another bounded ROS service spin after stale or transient readbacks."""
    return bool(stale_readback_keys(snapshot, max_age_s) or snapshot.get("readback_fault"))


def derive_stationary_odom_baseline(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Build three 1-second forward-drift windows from the no-command baseline.

    The resulting envelope is empirical for this run: maximum absolute drift
    observed in the baseline windows. It is not a vendor threshold or a
    statistical confidence bound.
    """
    if len(samples) < BASELINE_WINDOW_COUNT + 1:
        raise ValueError("stationary odometry baseline is incomplete")
    parsed: list[tuple[float, Mapping[str, Any]]] = []
    for sample in samples:
        try:
            timestamp = float(sample["sample_monotonic_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("baseline sample lacks a monotonic timestamp") from exc
        if not math.isfinite(timestamp) or not isinstance(sample.get("pose"), Mapping):
            raise ValueError("baseline sample has invalid timestamp or pose")
        parsed.append((timestamp, sample["pose"]))
    parsed.sort(key=lambda item: item[0])
    baseline_start = parsed[0][0]
    baseline_end = baseline_start + BASELINE_DURATION_S
    if parsed[-1][0] < baseline_end - BASELINE_SAMPLE_PERIOD_S:
        raise ValueError("baseline does not cover three complete 1-second windows")

    windows: list[dict[str, float]] = []
    for index in range(BASELINE_WINDOW_COUNT):
        target_start = baseline_start + index * BASELINE_WINDOW_S
        target_end = target_start + BASELINE_WINDOW_S
        start_sample = min(parsed, key=lambda item: abs(item[0] - target_start))
        end_sample = min(parsed, key=lambda item: abs(item[0] - target_end))
        if (
            abs(start_sample[0] - target_start) > BASELINE_SAMPLE_PERIOD_S
            or abs(end_sample[0] - target_end) > BASELINE_SAMPLE_PERIOD_S
        ):
            raise ValueError("baseline has a gap at a 1-second window boundary")
        duration = end_sample[0] - start_sample[0]
        if duration <= 0.0:
            raise ValueError("baseline window has nonpositive duration")
        windows.append({
            "start_monotonic_s": start_sample[0],
            "end_monotonic_s": end_sample[0],
            "duration_s": duration,
            "forward_displacement_m": forward_projected_displacement(start_sample[1], end_sample[1]),
        })
    noise_bound = max(abs(window["forward_displacement_m"]) for window in windows)
    return {
        "method": "maximum absolute forward displacement in three no-command 1-second windows",
        "window_duration_s": BASELINE_WINDOW_S,
        "window_count": len(windows),
        "sample_count": len(parsed),
        "windows": windows,
        "noise_bound_m": noise_bound,
        "statistical_confidence_bound": False,
    }


def classify_physical_motion(
    *,
    control_handshake_confirmed: bool,
    pulse_forward_displacement_m: float | None,
    baseline_noise_bound_m: float | None,
) -> str:
    """Require measured forward movement beyond this run's empirical baseline envelope."""
    if baseline_noise_bound_m is None or not math.isfinite(float(baseline_noise_bound_m)):
        return "BASELINE_UNAVAILABLE"
    if float(baseline_noise_bound_m) <= 0.0:
        return "BASELINE_NOISE_UNRESOLVED"
    if pulse_forward_displacement_m is None or not math.isfinite(float(pulse_forward_displacement_m)):
        return "PULSE_ODOMETRY_UNAVAILABLE"
    noise = float(baseline_noise_bound_m)
    displacement = float(pulse_forward_displacement_m)
    if displacement < -noise:
        return "MOTION_OPPOSITE_COMMAND"
    if displacement <= noise:
        return "CONTROL_HANDSHAKE_ONLY" if control_handshake_confirmed else "NO_MOTION_CONFIRMED"
    if not control_handshake_confirmed:
        return "UNEXPECTED_MOTION_WITHOUT_CONTROL_HANDSHAKE"
    return "PHYSICAL_MOTION_CONFIRMED"


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


def load_mc_input_source_timeouts(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Read the installed MC sim source-timeout settings; this is config evidence only."""
    import yaml

    source = Path(path)
    payload = source.read_bytes()
    document = yaml.safe_load(payload)
    try:
        input_sources = document["MotionControlModule"]["InputManager"]["input_sources"]
    except (TypeError, KeyError) as exc:
        raise ValueError("mc_sim.yaml has no MotionControlModule.InputManager.input_sources") from exc
    if not isinstance(input_sources, list) or not input_sources:
        raise ValueError("mc_sim.yaml input_sources must be a nonempty list")
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in input_sources:
        if not isinstance(item, Mapping):
            raise ValueError("mc_sim.yaml has a malformed input source")
        name = item.get("name")
        try:
            priority = int(item["priority"])
            timeout_ms = int(item["timeout"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("mc_sim.yaml input source needs integer priority and timeout") from exc
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("mc_sim.yaml input source name is missing or duplicated")
        if not 0 <= priority <= 100 or timeout_ms <= 0:
            raise ValueError(f"mc_sim.yaml input source has invalid priority or timeout: {name}")
        names.add(name)
        records.append({"name": name, "priority": priority, "timeout_ms": timeout_ms})
    return records, hashlib.sha256(payload).hexdigest()


def validate_runtime_scope(
    *,
    rmw_implementation: str | None,
    ros_distro: str | None,
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
    publisher_local_identity: Mapping[str, Any] | None = None,
    publisher_configured_qos: Mapping[str, Any] | None = None,
    publisher_api_qos: Mapping[str, Any] | None = None,
    subscriber_histories: list[str],
    subscriber_depths: list[int],
    subscriber_endpoint_metadata: list[Mapping[str, Any]] | None = None,
    mc_subscriber_provenance: Mapping[str, Any] | None = None,
    dds_qos_override_environment: Mapping[str, Any] | None = None,
    mc_identity: VerifiedMcRuntimeIdentity | None,
) -> str | None:
    """Fail closed unless this is the isolated local MC/simulator graph."""
    identity_fault = _verified_mc_identity_error(mc_identity)
    if identity_fault:
        return identity_fault
    assert mc_identity is not None
    if rmw_implementation != "rmw_fastrtps_cpp":
        return "unexpected_rmw_implementation"
    if ros_distro != "humble":
        return f"unexpected_ros_distro:{ros_distro!r}"
    if ros_domain_id != "20":
        return "unexpected_ros_domain_id"
    if docker_network_mode != "bridge":
        return "container_network_not_isolated_bridge"
    if publisher_names != [own_node_name]:
        return f"velocity_publisher_conflict:{publisher_names!r}"
    local_publisher_fault = _local_command_publisher_error(
        publisher_local_identity=publisher_local_identity,
        publisher_configured_qos=publisher_configured_qos,
        publisher_api_qos=publisher_api_qos,
        own_node_name=own_node_name,
    )
    if local_publisher_fault:
        return local_publisher_fault
    expected_node_name = mc_identity.node_name
    if subscriber_names != [expected_node_name]:
        return f"expected_only_verified_mc_subscriber:{expected_node_name!r}:{subscriber_names!r}"
    subscriber_provenance_fault = _verified_mc_velocity_subscriber_provenance_error(
        mc_subscriber_provenance, mc_identity,
    )
    if subscriber_provenance_fault:
        return subscriber_provenance_fault
    endpoint_metadata_fault = _mc_velocity_endpoint_metadata_error(
        subscriber_endpoint_metadata, mc_identity,
    )
    if endpoint_metadata_fault:
        return endpoint_metadata_fault
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
    # rmw_fastrtps may omit writer history/depth from discovery and expose
    # exactly unknown/0. Accept only that paired metadata gap after the local
    # Publisher API profile above has independently matched the configured
    # KEEP_LAST/10 contract. Any known or partial mismatch remains a failure.
    if not (publisher_history == "unknown" and publisher_depth == 0) and (
        publisher_history != "keep_last" or publisher_depth != COMMAND_QOS_DEPTH
    ):
        return f"command_publisher_history_depth_mismatch:{publisher_history}:{publisher_depth}"
    if len(subscriber_histories) != len(subscriber_names) or len(subscriber_depths) != len(subscriber_names):
        return "subscriber_history_depth_endpoint_count_mismatch"
    history, depth = subscriber_histories[0], subscriber_depths[0]
    if history == "unknown" and isinstance(depth, int) and not isinstance(depth, bool) and depth == 0:
        # In the observed Humble + Fast DDS runtime, the graph's unknown/0 pair is
        # missing discovery metadata, not a DataReader QoS query (upstream: rmw_fastrtps
        # issue 684 and the Humble qos.hpp conversion). Accept only this exact pair
        # after MC identity/config/type/namespace and compatibility
        # checks above, with no active XML QoS overrides. The reader depth stays
        # explicitly unverified and this preflight does not prove command receipt.
        override_fault = _dds_qos_override_environment_error(dds_qos_override_environment)
        if override_fault:
            return override_fault
        return None
    if history != "keep_last" or isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        return f"mc_subscriber_history_depth_unsupported:{subscriber_histories!r}:{subscriber_depths!r}"
    return None


def validate_local_sim_publishers(
    topic_publishers: Mapping[str, list[str]],
    *,
    mc_identity: VerifiedMcRuntimeIdentity | None,
) -> str | None:
    """Require one state publisher per simulator stream and the verified MC command publisher."""
    identity_fault = _verified_mc_identity_error(mc_identity)
    if identity_fault:
        return identity_fault
    assert mc_identity is not None
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
    if topic_publishers["/aima/hal/joint/leg/command"] != [mc_identity.node_name]:
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


def _source_only_readback_fault(value: Any) -> bool:
    if not value:
        return False
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    return bool(parts) and all(part.startswith("GetCurrentInputSource") for part in parts)


def source_readback_gap(snapshot: Mapping[str, Any], *, state_timeout_s: float) -> str | None:
    """Describe an unavailable source readback without classifying it as physical-state loss."""
    source_fault = snapshot.get("get_source_readback_fault")
    if source_fault:
        return str(source_fault)
    # Compatibility with captured snapshots created before readback faults
    # were separated by service.
    generic_fault = snapshot.get("readback_fault")
    if _source_only_readback_fault(generic_fault):
        return str(generic_fault)
    ages = snapshot.get("state_age_s")
    age = ages.get("get_source") if isinstance(ages, Mapping) else None
    if (
        not isinstance(age, (int, float)) or not math.isfinite(float(age))
        or not 0.0 <= float(age) <= state_timeout_s
    ):
        return "GetCurrentInputSource readback is missing or stale"
    if snapshot.get("get_source_header_code") != 0:
        return "GetCurrentInputSource response header is not successful"
    return None


def snapshot_fault(
    snapshot: Mapping[str, Any],
    joint_limits: Mapping[str, Mapping[str, float]],
    *,
    state_timeout_s: float,
    min_root_z: float,
    max_tilt_deg: float,
    expected_source: str = VELOCITY_SOURCE_NAME,
    required_forward_mps: float = FORWARD_COMMAND_M_S,
    max_observed_planar_speed_m_s: float | None = None,
    allow_source_readback_gap: bool = False,
    allow_common_uninterruptible_transition_during_zero_observation: bool = False,
) -> str | None:
    if allow_source_readback_gap:
        mode_readback_fault = snapshot.get("get_mode_readback_fault")
        if mode_readback_fault:
            return f"readback_fault:{mode_readback_fault}"
        generic_readback_fault = snapshot.get("readback_fault")
        if generic_readback_fault and not _source_only_readback_fault(generic_readback_fault):
            return f"readback_fault:{generic_readback_fault}"
    elif snapshot.get("readback_fault"):
        return f"readback_fault:{snapshot['readback_fault']}"
    ages = snapshot.get("state_age_s")
    if not isinstance(ages, Mapping):
        return "missing_state_ages"
    required_state_keys = (
        tuple(key for key in REQUIRED_STATE_KEYS if key != "get_source")
        if allow_source_readback_gap else REQUIRED_STATE_KEYS
    )
    for key in required_state_keys:
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
    common_status = snapshot.get("common_mode_status_value")
    common_status_diagnostic = common_action_status_diagnostic(common_status)
    if common_status_diagnostic["classification"] == "UNINTERRUPTIBLE_TRANSITION":
        if not allow_common_uninterruptible_transition_during_zero_observation:
            return "common_action_uninterruptible_transition_not_allowed_in_this_phase"
    elif common_status_diagnostic["classification"] == "UNKNOWN_RUNTIME_STATUS":
        return (
            "UNKNOWN_RUNTIME_STATUS:McCommonState.action_info.status="
            f"{common_status_diagnostic['raw_value']!r}"
        )

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
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or not math.isfinite(required_forward_mps)
        or not minimum <= required_forward_mps <= maximum
    ):
        return "activation_speed_outside_live_mc_bounds"

    source_names = (
        snapshot.get("get_source_name", ""),
        snapshot.get("common_source_name", ""),
    )
    if not allow_source_readback_gap and snapshot.get("get_source_header_code") != 0:
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
    if max_observed_planar_speed_m_s is not None:
        try:
            vx = float(pose["vx"])
            vy = float(pose["vy"])
            observed_planar_speed = math.hypot(vx, vy)
            maximum_planar_speed = float(max_observed_planar_speed_m_s)
        except (KeyError, TypeError, ValueError):
            return "malformed_odom_planar_velocity"
        if (
            not math.isfinite(observed_planar_speed)
            or not math.isfinite(maximum_planar_speed)
            or maximum_planar_speed <= 0.0
        ):
            return "nonfinite_or_invalid_odom_speed_gate"
        if observed_planar_speed > maximum_planar_speed:
            return (
                "measured_root_planar_speed_limit:"
                f"{observed_planar_speed:.6f}>{maximum_planar_speed:.6f}"
            )
    return None


class ActivationTracker:
    """Keep source acquisition, LD observation, and same-sample handshake distinct."""

    def __init__(self, expected_source: str = VELOCITY_SOURCE_NAME) -> None:
        self.expected_source = expected_source
        self.get_source_seen = False
        self.common_source_seen = False
        self.get_locomotion_seen = False
        self.common_locomotion_seen = False
        self.source_acquired = False
        self.ld_observed = False
        self.control_handshake_confirmed = False
        self.last_control_handshake_observed = False
        self.samples: list[dict[str, Any]] = []
        self.first_pose: dict[str, float] | None = None
        self.first_pose_monotonic_s: float | None = None
        self.last_activation_pose: dict[str, float] | None = None
        self.last_activation_monotonic_s: float | None = None
        self.handshake_sample_indices: list[int] = []

    def observe(self, snapshot: Mapping[str, Any], *, phase: str) -> None:
        row = {
            "get_mode_desc": snapshot.get("get_mode_desc"),
            "common_mode_desc": snapshot.get("common_mode_desc"),
            "get_source_name": snapshot.get("get_source_name", ""),
            "common_source_name": snapshot.get("common_source_name", ""),
            "fsm_state": snapshot.get("fsm_state"),
            "pose": dict(snapshot.get("pose", {})) if isinstance(snapshot.get("pose"), Mapping) else None,
            "sample_monotonic_s": snapshot.get("sample_monotonic_s"),
            "readback_monotonic_s": dict(snapshot.get("readback_monotonic_s", {})),
            "get_mode_header_code": snapshot.get("get_mode_header_code"),
            "get_source_header_code": snapshot.get("get_source_header_code"),
            "get_mode_status_value": snapshot.get("get_mode_status_value"),
            "common_mode_status_value": snapshot.get("common_mode_status_value"),
            "phase": phase,
        }
        self.samples.append(row)
        if isinstance(row["pose"], Mapping):
            pose = {key: float(value) for key, value in row["pose"].items() if isinstance(value, (int, float))}
            if self.first_pose is None:
                self.first_pose = pose
                self.first_pose_monotonic_s = row["sample_monotonic_s"]
            if phase == "activation":
                self.last_activation_pose = pose
                self.last_activation_monotonic_s = row["sample_monotonic_s"]

        source_get = row["get_source_header_code"] == 0 and row["get_source_name"] == self.expected_source
        source_common = row["common_source_name"] == self.expected_source
        mode_get_seen = row["get_mode_header_code"] == 0 and row["get_mode_desc"] == MODE_LOCOMOTION
        mode_common_seen = row["common_mode_desc"] == MODE_LOCOMOTION
        mode_get = mode_get_seen and row.get("get_mode_status_value") == 100
        mode_common = mode_common_seen and row.get("common_mode_status_value") == 100
        source_pair = source_get and source_common
        mode_pair = mode_get_seen and mode_common_seen
        self.get_source_seen |= source_get
        self.common_source_seen |= source_common
        self.get_locomotion_seen |= mode_get_seen
        self.common_locomotion_seen |= mode_common_seen
        self.source_acquired |= source_pair
        self.ld_observed |= mode_pair

        readbacks = row["readback_monotonic_s"]
        try:
            mode_time = float(readbacks["get_mode"])
            source_time = float(readbacks["get_source"])
            readback_skew = abs(mode_time - source_time)
            if not math.isfinite(readback_skew):
                readback_skew = None
        except (KeyError, TypeError, ValueError):
            mode_time = source_time = None
            readback_skew = None
        common_skew = None
        try:
            sample_time = float(row["sample_monotonic_s"])
            common_age = float(snapshot["state_age_s"]["common"])
            common_time = sample_time - common_age
            if mode_time is not None and source_time is not None:
                common_skew = max(abs(mode_time - common_time), abs(source_time - common_time))
                if not math.isfinite(common_skew):
                    common_skew = None
        except (KeyError, TypeError, ValueError):
            common_skew = None
        row["mode_source_readback_skew_s"] = readback_skew
        row["mode_source_common_skew_max_s"] = common_skew
        readback_ages = snapshot.get("state_age_s", {})
        try:
            get_mode_age = float(readback_ages["get_mode"])
            get_source_age = float(readback_ages["get_source"])
            common_state_age = float(readback_ages["common"])
            maximum_readback_age_s = max(get_mode_age, get_source_age, common_state_age)
            fresh_readbacks = (
                all(math.isfinite(value) and value >= 0.0 for value in (
                    get_mode_age, get_source_age, common_state_age,
                ))
                and maximum_readback_age_s <= 0.5
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            get_mode_age = get_source_age = common_state_age = None
            maximum_readback_age_s = None
            fresh_readbacks = False
        row["mode_readback_age_s"] = get_mode_age
        row["source_readback_age_s"] = get_source_age
        row["common_state_age_s"] = common_state_age
        row["fresh_mode_source_readbacks"] = fresh_readbacks
        row["mode_source_readbacks_contemporaneous"] = (
            readback_skew is not None
            and readback_skew <= MODE_SOURCE_READBACK_SKEW_MAX_S
            and common_skew is not None
            and common_skew <= MODE_SOURCE_READBACK_SKEW_MAX_S
            and fresh_readbacks
        )
        row["source_pair_observed"] = source_pair
        row["ld_pair_observed"] = mode_pair
        row["control_handshake_observed"] = (
            source_pair and mode_get and mode_common
            and row["mode_source_readbacks_contemporaneous"]
        )
        self.last_control_handshake_observed = bool(row["control_handshake_observed"])
        if self.last_control_handshake_observed:
            self.control_handshake_confirmed = True
            self.handshake_sample_indices.append(len(self.samples) - 1)

    def classify_control_handshake(self) -> str:
        if not self.source_acquired:
            return "SOURCE_NOT_ACQUIRED"
        if not self.ld_observed:
            return "LD_NOT_OBSERVED"
        if not self.control_handshake_confirmed:
            return "CONTROL_HANDSHAKE_NOT_CONFIRMED"
        return "CONTROL_HANDSHAKE_CONFIRMED"

    def odometry_delta(self) -> dict[str, float | int] | None:
        """Return raw world delta and forward projection through the pulse endpoint."""
        if self.first_pose is None or self.last_activation_pose is None:
            return None
        delta_yaw = self.last_activation_pose.get("yaw", 0.0) - self.first_pose.get("yaw", 0.0)
        return {
            "dx_m": self.last_activation_pose.get("x", 0.0) - self.first_pose.get("x", 0.0),
            "dy_m": self.last_activation_pose.get("y", 0.0) - self.first_pose.get("y", 0.0),
            "dyaw_rad": math.atan2(math.sin(delta_yaw), math.cos(delta_yaw)),
            "forward_projected_m": forward_projected_displacement(self.first_pose, self.last_activation_pose),
            "start_heading_rad": self.first_pose.get("yaw", 0.0),
            "start_monotonic_s": self.first_pose_monotonic_s,
            "end_monotonic_s": self.last_activation_monotonic_s,
            "sample_count": len(self.samples),
        }

    def post_stop_odometry_delta(self, post_stop_pose: Mapping[str, Any] | None) -> float | None:
        if self.last_activation_pose is None or not isinstance(post_stop_pose, Mapping):
            return None
        return forward_projected_displacement(self.last_activation_pose, post_stop_pose)


class ProcessLocalTimerWatchdog:
    """A threading.Timer in this process; not a process-external crash watchdog."""

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
    """One idempotent burst of local zero-publish calls spanning the source timeout.

    A Python publisher call returning is not an MC receipt or stop acknowledgement.
    """

    def __init__(
        self,
        publish_zero: Callable[[], None],
        *,
        timeout_s: float = ZERO_STOP_MIN_S,
        publish_hz: float = PUBLISH_HZ,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        command_lock: Any | None = None,
        during_zero_callback: Callable[[], None] | None = None,
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
        self.during_zero_callback = during_zero_callback
        self._start_lock = threading.Lock()
        self._started = threading.Event()
        self._done = threading.Event()
        self.result: dict[str, Any] = {
            "requested_duration_s": timeout_s,
            "start_monotonic_s": None,
            "first_publish_attempt_monotonic_s": None,
            "end_monotonic_s": None,
            "loop_duration_s": 0.0,
            "zero_publish_calls_started": 0,
            "zero_publish_calls_returned": 0,
            "during_zero_callback_count": 0,
            "during_zero_callback_errors": [],
            "publish_call_return_times_s": [],
            "publish_call_return_span_s": 0.0,
            "max_gap_between_publish_call_returns_s": None,
            "publish_exceptions": [],
            "mc_receipt_confirmed": False,
            "mc_stop_confirmed": False,
            "interpretation": ZERO_STOP_INTERPRETATION,
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
        self.result["start_monotonic_s"] = start
        deadline = start + self.timeout_s
        hard_deadline = deadline + 5.0 * self.period_s
        next_publish = start
        try:
            # A wall-time burst alone can finish with the first/last returned
            # calls less than one source timeout apart under ROS scheduling
            # jitter. Permit only a bounded tail of ZERO calls to finish the
            # original >=1 s / >=50 calls gate; never extend nonzero motion.
            while self.clock() < hard_deadline:
                times = self.result["publish_call_return_times_s"]
                complete = (
                    len(times) >= ZERO_STOP_MIN_MESSAGES
                    and times[-1] - times[0] >= ZERO_STOP_MIN_S
                )
                if self.clock() >= deadline and complete:
                    break
                try:
                    with self.command_lock:
                        self.result["zero_publish_calls_started"] += 1
                        if self.result["first_publish_attempt_monotonic_s"] is None:
                            self.result["first_publish_attempt_monotonic_s"] = self.clock()
                        self.publish_zero()
                    self.result["zero_publish_calls_returned"] += 1
                    self.result["publish_call_return_times_s"].append(self.clock() - start)
                except Exception as exc:
                    self.result["publish_exceptions"].append(f"{type(exc).__name__}: {exc}")
                if self.during_zero_callback is not None:
                    try:
                        self.during_zero_callback()
                        self.result["during_zero_callback_count"] += 1
                    except Exception as exc:
                        self.result["during_zero_callback_errors"].append(
                            f"{type(exc).__name__}: {exc}"
                        )
                    next_publish += self.period_s
                    now = self.clock()
                    if next_publish < now:
                        next_publish = now + self.period_s
                    remaining = min(next_publish, hard_deadline) - now
                else:
                    remaining = hard_deadline - self.clock()
                    remaining = min(self.period_s, remaining)
                if remaining > 0.0:
                    self.sleep(remaining)
        finally:
            end = self.clock()
            self.result["end_monotonic_s"] = end
            self.result["loop_duration_s"] = max(0.0, end - start)
            return_times = self.result["publish_call_return_times_s"]
            if return_times:
                self.result["publish_call_return_span_s"] = max(0.0, return_times[-1] - return_times[0])
                if len(return_times) > 1:
                    self.result["max_gap_between_publish_call_returns_s"] = max(
                        later - earlier for earlier, later in zip(return_times, return_times[1:])
                    )
            self._done.set()
        return dict(self.result)


def measure_post_stop_forward_tail(
    samples: list[Mapping[str, Any]], baseline_noise_bound_m: float | None,
) -> dict[str, Any]:
    """Separate braking travel from motion over a complete final baseline-length window.

    Keep the original pulse-endpoint displacement as braking evidence. A
    stationary final window need not coincide with the moving pulse endpoint.
    Use maximum excursion, so returning to the starting position cannot fake
    stationary behavior. This remains an empirical odometry test, not MC ACK.
    """
    rows = [s for s in samples if s.get("phase") == "post-stop"
            and isinstance(s.get("pose"), Mapping)]
    result: dict[str, Any] = {"status": "INCOMPLETE_FINAL_WINDOW", "mc_stop_confirmed": False}
    if not rows:
        return result
    end = float(rows[-1]["sample_monotonic_s"])
    eligible = [i for i, s in enumerate(rows)
                if float(s["sample_monotonic_s"]) <= end - BASELINE_WINDOW_S]
    if not eligible:
        return result
    tail = rows[eligible[-1]:]
    excursion = max(abs(forward_projected_displacement(tail[0]["pose"], s["pose"]))
                    for s in tail)
    result.update({
        "duration_s": end - float(tail[0]["sample_monotonic_s"]),
        "sample_count": len(tail),
        "max_forward_excursion_m": excursion,
        "baseline_noise_bound_m": baseline_noise_bound_m,
        "start_pose": dict(tail[0]["pose"]), "end_pose": dict(tail[-1]["pose"]),
        "status": "ABOVE_BASELINE_ENVELOPE",
        "scope": "forward odometry stationary relative to the same-run baseline, not an MC stop acknowledgement",
    })
    if (baseline_noise_bound_m is not None and baseline_noise_bound_m > 0.0
            and excursion <= baseline_noise_bound_m
            and all(not s.get("safety_fault") for s in tail)
            and all(s.get("common_mode_desc") == MODE_STAND
                    and s.get("get_mode_desc") == MODE_STAND
                    and s.get("fsm_state") == MC_FSM_STABLE for s in tail)):
        result["status"] = "WITHIN_BASELINE_ENVELOPE"
    return result


def _unwrap_yaw(values: list[float]) -> list[float]:
    if not values:
        return []
    result = [values[0]]
    for value in values[1:]:
        delta = (value - result[-1] + math.pi) % (2.0 * math.pi) - math.pi
        result.append(result[-1] + delta)
    return result


def _least_squares_slope(times: list[float], values: list[float]) -> float | None:
    if len(times) != len(values) or len(times) < 2:
        return None
    mean_t = sum(times) / len(times)
    mean_v = sum(values) / len(values)
    denominator = sum((value - mean_t) ** 2 for value in times)
    if denominator <= 0.0:
        return None
    return sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values)) / denominator


def _physical_window_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one time-aligned window without treating mode readback as physics."""
    if len(rows) < 2:
        raise ValueError("physical-state window requires at least two samples")
    ordered = sorted(rows, key=lambda row: float(row["sample_monotonic_s"]))
    times = [float(row["sample_monotonic_s"]) for row in ordered]
    if not all(math.isfinite(value) for value in times):
        raise ValueError("physical-state window has invalid monotonic timestamps")
    poses = [row.get("pose") for row in ordered]
    required_pose_fields = ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz")
    if any(not isinstance(pose, Mapping) for pose in poses):
        raise ValueError("physical-state window has no complete odometry pose")
    numeric_poses: list[dict[str, float]] = []
    for pose in poses:
        parsed = {name: float(pose[name]) for name in required_pose_fields}
        if not all(math.isfinite(value) for value in parsed.values()):
            raise ValueError("physical-state window contains nonfinite odometry")
        numeric_poses.append(parsed)
    x0, y0 = numeric_poses[0]["x"], numeric_poses[0]["y"]
    planar_excursions = [math.hypot(pose["x"] - x0, pose["y"] - y0) for pose in numeric_poses]
    planar_path = sum(
        math.hypot(later["x"] - earlier["x"], later["y"] - earlier["y"])
        for earlier, later in zip(numeric_poses, numeric_poses[1:])
    )
    yaws = _unwrap_yaw([pose["yaw"] for pose in numeric_poses])
    yaw_rate = _least_squares_slope(times, yaws)

    joint_samples: dict[str, list[tuple[float, float, int]]] = {}
    command_samples: dict[str, list[float]] = {}
    for row in ordered:
        joints = row.get("leg_states")
        if not isinstance(joints, list) or not joints:
            raise ValueError("physical-state window is missing measured leg joints")
        for joint in joints:
            name = str(joint["name"])
            position, velocity = float(joint["position"]), float(joint["velocity"])
            error_code = int(joint.get("error_code", 0))
            if not math.isfinite(position) or not math.isfinite(velocity):
                raise ValueError("physical-state window contains nonfinite joint state")
            joint_samples.setdefault(name, []).append((position, velocity, error_code))
        command = row.get("mc_leg_command")
        command_joints = command.get("joints") if isinstance(command, Mapping) else None
        if not isinstance(command_joints, list) or not command_joints:
            raise ValueError("physical-state window is missing MC leg-command targets")
        for joint in command_joints:
            name = str(joint["name"])
            position = float(joint["position"])
            if not math.isfinite(position):
                raise ValueError("physical-state window contains nonfinite MC leg target")
            command_samples.setdefault(name, []).append(position)
    expected_names = set(joint_samples)
    if not expected_names or any(len(values) != len(ordered) for values in joint_samples.values()):
        raise ValueError("measured leg-joint coverage changes inside stop window")
    if set(command_samples) != expected_names or any(
        len(values) != len(ordered) for values in command_samples.values()
    ):
        raise ValueError("MC leg-command coverage does not match measured leg-joint coverage")

    ages: list[float] = []
    age_coverage = True
    for row in ordered:
        state_ages = row.get("state_age_s")
        if not isinstance(state_ages, Mapping):
            age_coverage = False
            continue
        for key in ("odom", "imu", "common", "leg_state", "leg_command", "get_mode"):
            value = state_ages.get(key)
            if value is None:
                age_coverage = False
                continue
            age = float(value)
            if not math.isfinite(age) or age < 0.0:
                age_coverage = False
            else:
                ages.append(age)

    frame_pairs = {
        (str(pose.get("frame_id", "")), str(pose.get("child_frame_id", "")))
        for pose in poses if isinstance(pose, Mapping)
    }
    frame_pair_available = bool(frame_pairs) and all(all(pair) for pair in frame_pairs)
    errors = [
        str(row.get("safety_fault")) for row in ordered
        if row.get("safety_fault") or row.get("safety_check") == "FAIL"
    ]
    return {
        "window_start_monotonic_s": times[0],
        "window_end_monotonic_s": times[-1],
        "duration_s": times[-1] - times[0],
        "sample_count": len(ordered),
        "max_sample_gap_s": max(later - earlier for earlier, later in zip(times, times[1:])),
        "max_state_age_s": max(ages) if ages else None,
        "state_age_coverage_complete": age_coverage,
        "frame_pairs": [list(pair) for pair in sorted(frame_pairs)],
        "frame_pair_available": frame_pair_available,
        "frame_pair_consistent": len(frame_pairs) == 1 if frame_pair_available else None,
        "max_planar_speed_m_s": max(math.hypot(pose["vx"], pose["vy"]) for pose in numeric_poses),
        "max_abs_vertical_speed_m_s": max(abs(pose["vz"]) for pose in numeric_poses),
        "planar_endpoint_drift_m": math.hypot(
            numeric_poses[-1]["x"] - x0, numeric_poses[-1]["y"] - y0
        ),
        "max_planar_excursion_m": max(planar_excursions),
        "planar_path_length_m": planar_path,
        "yaw_drift_rad": yaws[-1] - yaws[0],
        "max_abs_yaw_excursion_rad": max(abs(value - yaws[0]) for value in yaws),
        "yaw_rate_fit_rad_s": yaw_rate,
        "z_range_m": max(pose["z"] for pose in numeric_poses) - min(pose["z"] for pose in numeric_poses),
        "max_abs_tilt_rad": max(math.hypot(pose["roll"], pose["pitch"]) for pose in numeric_poses),
        "roll_range_rad": max(pose["roll"] for pose in numeric_poses) - min(pose["roll"] for pose in numeric_poses),
        "pitch_range_rad": max(pose["pitch"] for pose in numeric_poses) - min(pose["pitch"] for pose in numeric_poses),
        "max_abs_joint_velocity_rad_s": max(
            abs(velocity) for values in joint_samples.values() for _, velocity, _ in values
        ),
        "max_joint_position_range_rad": max(
            max(position for position, _, _ in values) - min(position for position, _, _ in values)
            for values in joint_samples.values()
        ),
        "max_joint_command_range_rad": max(
            max(values) - min(values) for values in command_samples.values()
        ),
        "joint_error_codes": sorted({
            error for values in joint_samples.values() for _, _, error in values
        }),
        "safety_faults": errors,
        "mode_desc_values": sorted({str(row.get("get_mode_desc")) for row in ordered}),
        "common_mode_desc_values": sorted({str(row.get("common_mode_desc")) for row in ordered}),
        "fsm_state_values": sorted({row.get("fsm_state") for row in ordered}, key=str),
        "body_status_values": sorted({row.get("body_status") for row in ordered}, key=str),
        "get_mode_header_codes": sorted({row.get("get_mode_header_code") for row in ordered}, key=str),
        "get_mode_status_values": sorted({row.get("get_mode_status_value") for row in ordered}, key=str),
        "common_mode_status_values": sorted({row.get("common_mode_status_value") for row in ordered}, key=str),
        "readback_faults": sorted({str(row.get("readback_fault")) for row in ordered if row.get("readback_fault")}),
    }


def summarize_stationary_physical_state(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Record per-second simulator-state noise to justify the separate physics gate."""
    if len(samples) < BASELINE_WINDOW_COUNT + 1:
        raise ValueError("stationary physical-state baseline is incomplete")
    rows = sorted(samples, key=lambda row: float(row["sample_monotonic_s"]))
    start = float(rows[0]["sample_monotonic_s"])
    summaries = []
    for index in range(BASELINE_WINDOW_COUNT):
        begin = start + index * BASELINE_WINDOW_S
        end = begin + BASELINE_WINDOW_S
        window = [row for row in rows if begin <= float(row["sample_monotonic_s"]) <= end]
        if len(window) < 2 or float(window[-1]["sample_monotonic_s"]) - float(window[0]["sample_monotonic_s"]) < 0.9:
            raise ValueError("stationary physical-state baseline lacks a complete one-second window")
        summary = _physical_window_metrics(window)
        if summary["duration_s"] > 1.1:
            raise ValueError("stationary physical-state baseline window exceeds bounded duration")
        summaries.append(summary)
    numeric_keys = (
        "max_planar_speed_m_s", "max_abs_vertical_speed_m_s", "planar_endpoint_drift_m",
        "planar_path_length_m", "max_abs_yaw_excursion_rad", "yaw_rate_fit_rad_s",
        "z_range_m", "max_abs_joint_velocity_rad_s", "max_joint_position_range_rad",
        "max_joint_command_range_rad",
    )
    return {
        "method": "three no-command one-second windows from simulator odometry and leg state/command telemetry",
        "window_count": len(summaries),
        "window_duration_s": BASELINE_WINDOW_S,
        "body_status": rows[0].get("body_status"),
        "windows": summaries,
        "maxima": {key: max(float(window[key]) for window in summaries) for key in numeric_keys},
        "window_indices": list(range(BASELINE_WINDOW_COUNT)),
        "limitations": [
            "MuJoCo foot contact forces and tangential slip are not exported by the installed simulator interface.",
            "Pose/twist are read from the simulator-owned /aima/hal/odom/state stream; the vendor binary does not expose a second raw-qpos telemetry channel.",
        ],
    }


def _physical_stop_violations(
    metrics: Mapping[str, Any], *, baseline_body_status: Any = None,
) -> list[str]:
    violations: list[str] = []
    if metrics.get("duration_s", 0.0) < PHYSICAL_STOP_MIN_WINDOW_S:
        violations.append("stable_window_too_short")
    if metrics.get("sample_count", 0) < PHYSICAL_STOP_MIN_SAMPLES:
        violations.append("too_few_state_samples")
    if metrics.get("max_sample_gap_s", math.inf) > PHYSICAL_STOP_MAX_SAMPLE_GAP_S:
        violations.append("state_sample_gap_exceeded")
    if not metrics.get("state_age_coverage_complete") or metrics.get("max_state_age_s") is None:
        violations.append("state_freshness_unverified")
    elif metrics["max_state_age_s"] > PHYSICAL_STOP_MAX_STATE_AGE_S:
        violations.append("state_stale")
    if metrics.get("safety_faults"):
        violations.append("existing_safety_gate_fault")
    if metrics.get("readback_faults"):
        violations.append("readback_fault")
    if metrics.get("mode_desc_values") != [MODE_STAND]:
        violations.append("get_mc_action_not_stand_throughout_window")
    if metrics.get("common_mode_desc_values") != [MODE_STAND]:
        violations.append("common_state_not_stand_throughout_window")
    if metrics.get("fsm_state_values") != [MC_FSM_STABLE]:
        violations.append("raw_fsm_not_stable_throughout_window")
    if baseline_body_status is not None and metrics.get("body_status_values") != [baseline_body_status]:
        violations.append("raw_body_status_changed_from_standing_baseline")
    if metrics.get("get_mode_header_codes") != [0]:
        violations.append("get_mc_action_readback_failed")
    if metrics.get("get_mode_status_values") != [100] or metrics.get("common_mode_status_values") != [100]:
        violations.append("mode_status_not_idle_throughout_window")
    if metrics.get("joint_error_codes") != [0]:
        violations.append("joint_error_code_present")
    limits = (
        ("max_planar_speed_m_s", PHYSICAL_STOP_MAX_PLANAR_SPEED_M_S, "planar_speed_exceeded"),
        ("max_abs_vertical_speed_m_s", PHYSICAL_STOP_MAX_VERTICAL_SPEED_M_S, "vertical_speed_exceeded"),
        ("planar_endpoint_drift_m", PHYSICAL_STOP_MAX_PLANAR_DRIFT_M, "planar_endpoint_drift_exceeded"),
        ("max_planar_excursion_m", PHYSICAL_STOP_MAX_PLANAR_DRIFT_M, "planar_excursion_exceeded"),
        ("planar_path_length_m", PHYSICAL_STOP_MAX_PLANAR_PATH_M, "planar_path_exceeded"),
        ("max_abs_yaw_excursion_rad", PHYSICAL_STOP_MAX_YAW_DRIFT_RAD, "heading_excursion_exceeded"),
        ("yaw_rate_fit_rad_s", PHYSICAL_STOP_MAX_YAW_RATE_RAD_S, "heading_rate_exceeded"),
        ("z_range_m", PHYSICAL_STOP_MAX_HEIGHT_RANGE_M, "root_height_variation_exceeded"),
        ("max_abs_tilt_rad", PHYSICAL_STOP_MAX_TILT_RAD, "existing_tilt_gate_exceeded"),
        ("roll_range_rad", PHYSICAL_STOP_MAX_ROLL_PITCH_RANGE_RAD, "roll_variation_exceeded"),
        ("pitch_range_rad", PHYSICAL_STOP_MAX_ROLL_PITCH_RANGE_RAD, "pitch_variation_exceeded"),
        ("max_abs_joint_velocity_rad_s", PHYSICAL_STOP_MAX_JOINT_SPEED_RAD_S, "joint_velocity_not_settled"),
        ("max_joint_position_range_rad", PHYSICAL_STOP_MAX_JOINT_RANGE_RAD, "joint_position_not_settled"),
        ("max_joint_command_range_rad", PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD, "joint_target_not_settled"),
    )
    for key, limit, fault in limits:
        value = metrics.get(key)
        if value is None or not math.isfinite(float(value)) or abs(float(value)) > limit:
            violations.append(fault)
    if metrics.get("frame_pair_consistent") is False:
        violations.append("odometry_frame_pair_changed_or_missing")
    return violations


def _joint_stop_window_diagnostics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe measured joints and command targets independently for one window."""
    ordered = sorted(rows, key=lambda row: float(row["sample_monotonic_s"]))
    per_joint: dict[str, dict[str, list[float]]] = {}
    for row in ordered:
        states = {str(joint["name"]): joint for joint in row["leg_states"]}
        commands = {
            str(joint["name"]): joint
            for joint in row["mc_leg_command"]["joints"]
        }
        if set(states) != set(commands):
            raise ValueError("measured joints and MC command targets differ in stop window")
        for name in states:
            state = states[name]
            command = commands[name]
            entry = per_joint.setdefault(name, {
                "position_rad": [], "velocity_rad_s": [], "target_rad": [],
                "command_minus_state_rad": [],
            })
            position = float(state["position"])
            target = float(command["position"])
            velocity = float(state["velocity"])
            if not all(math.isfinite(value) for value in (position, target, velocity)):
                raise ValueError("nonfinite joint value in stop-window diagnostics")
            entry["position_rad"].append(position)
            entry["velocity_rad_s"].append(velocity)
            entry["target_rad"].append(target)
            entry["command_minus_state_rad"].append(target - position)

    diagnostics = []
    for name, values in per_joint.items():
        positions = values["position_rad"]
        targets = values["target_rad"]
        deltas = values["command_minus_state_rad"]
        diagnostics.append({
            "name": name,
            "sample_count": len(positions),
            "actual_position_start_rad": positions[0],
            "actual_position_end_rad": positions[-1],
            "actual_position_range_rad": max(positions) - min(positions),
            "actual_velocity_abs_max_rad_s": max(abs(value) for value in values["velocity_rad_s"]),
            "target_start_rad": targets[0],
            "target_end_rad": targets[-1],
            "target_delta_rad": targets[-1] - targets[0],
            "target_range_rad": max(targets) - min(targets),
            "samplewise_command_minus_state_start_rad": deltas[0],
            "samplewise_command_minus_state_end_rad": deltas[-1],
            "samplewise_command_minus_state_abs_max_rad": max(abs(value) for value in deltas),
        })
    diagnostics.sort(key=lambda item: (-item["target_range_rad"], item["name"]))
    return {
        "window_start_monotonic_s": float(ordered[0]["sample_monotonic_s"]),
        "window_end_monotonic_s": float(ordered[-1]["sample_monotonic_s"]),
        "sample_count": len(ordered),
        "samplewise_command_minus_state_interpretation": (
            "raw synchronized telemetry difference; not a calibrated actuator tracking error"
        ),
        "per_joint": diagnostics,
    }


def _rolling_physical_stop_windows(
    rows: list[Mapping[str, Any]], *, zero_time: float, baseline_body_status: Any = None,
) -> list[dict[str, Any]]:
    """Record unchanged 2 s physical/target gates at 0.5 s spacing after STAND."""
    if len(rows) < 2:
        return []
    ordered = sorted(rows, key=lambda row: float(row["sample_monotonic_s"]))
    first = float(ordered[0]["sample_monotonic_s"])
    last = float(ordered[-1]["sample_monotonic_s"])
    final_start = last - PHYSICAL_STOP_WINDOW_S
    starts: list[float] = []
    start = first
    while start <= final_start + 1e-9:
        starts.append(start)
        start += PHYSICAL_STOP_ROLLING_STEP_S
    if final_start >= first and (not starts or abs(starts[-1] - final_start) > 1e-6):
        starts.append(final_start)

    windows: list[dict[str, Any]] = []
    for index, requested_start in enumerate(starts):
        window_rows = [
            row for row in ordered
            if requested_start <= float(row["sample_monotonic_s"])
            <= requested_start + PHYSICAL_STOP_WINDOW_S
        ]
        entry: dict[str, Any] = {
            "window_index": index,
            "requested_start_monotonic_s": requested_start,
            "requested_start_after_zero_s": requested_start - zero_time,
            "window_duration_s": PHYSICAL_STOP_WINDOW_S,
            "sample_count": len(window_rows),
            "evaluation": "NOT_EVALUATED",
        }
        try:
            metrics = _physical_window_metrics(window_rows)
            violations = _physical_stop_violations(
                metrics, baseline_body_status=baseline_body_status,
            )
            physical_violations = [
                item for item in violations if item != "joint_target_not_settled"
            ]
            joint = _joint_stop_window_diagnostics(window_rows)
            right_ankle = next(
                (item for item in joint["per_joint"] if item["name"] == "right_ankle_pitch_joint"),
                None,
            )
            target_status = "FAIL" if "joint_target_not_settled" in violations else "PASS"
            physical_status = "FAIL" if physical_violations else "PASS"
            entry.update({
                "evaluation": "COMPLETE",
                "window_start_monotonic_s": metrics["window_start_monotonic_s"],
                "window_end_monotonic_s": metrics["window_end_monotonic_s"],
                "physical_state_gate": {"status": physical_status, "violations": physical_violations},
                "controller_target_stability_gate": {
                    "status": target_status,
                    "threshold_rad": PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD,
                    "max_joint_target_range_rad": metrics["max_joint_command_range_rad"],
                    "violations": [item for item in violations if item == "joint_target_not_settled"],
                },
                "composite_gate": {"status": "PASS" if not violations else "FAIL", "violations": violations},
                "raw_mode_fsm_body_values": {
                    "get_mode": metrics["mode_desc_values"],
                    "common_mode": metrics["common_mode_desc_values"],
                    "fsm": metrics["fsm_state_values"],
                    "body_status": metrics["body_status_values"],
                },
                "right_ankle_pitch_joint": right_ankle,
                "per_joint": joint["per_joint"],
                "metrics": metrics,
            })
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            entry["reason"] = f"{type(exc).__name__}: {exc}"
        windows.append(entry)
    return windows


def _first_stable_joint_target_window(
    samples: list[Mapping[str, Any]], *, after_monotonic_s: float,
) -> dict[str, Any]:
    rows = sorted(
        (row for row in samples if float(row.get("sample_monotonic_s", -math.inf)) >= after_monotonic_s),
        key=lambda row: float(row["sample_monotonic_s"]),
    )
    for start_index, start in enumerate(rows):
        start_time = float(start["sample_monotonic_s"])
        window = [
            row for row in rows[start_index:]
            if float(row["sample_monotonic_s"]) <= start_time + PHYSICAL_STOP_TARGET_STABILITY_WINDOW_S
        ]
        if len(window) < 2 or float(window[-1]["sample_monotonic_s"]) - start_time < 0.95:
            continue
        try:
            maps = [
                {str(joint["name"]): float(joint["position"])
                 for joint in row["mc_leg_command"]["joints"]}
                for row in window
            ]
            if not maps[0] or any(set(values) != set(maps[0]) for values in maps):
                continue
            max_range = max(
                max(values[name] for values in maps) - min(values[name] for values in maps)
                for name in maps[0]
            )
        except (KeyError, TypeError, ValueError):
            continue
        max_gap = max(
            float(later["sample_monotonic_s"]) - float(earlier["sample_monotonic_s"])
            for earlier, later in zip(window, window[1:])
        )
        if max_gap <= PHYSICAL_STOP_MAX_SAMPLE_GAP_S and max_range <= PHYSICAL_STOP_TARGET_STABILITY_RANGE_RAD:
            return {
                "status": "OBSERVED",
                "start_monotonic_s": start_time,
                "end_monotonic_s": float(window[-1]["sample_monotonic_s"]),
                "delay_from_zero_start_s": start_time - after_monotonic_s,
                "duration_s": float(window[-1]["sample_monotonic_s"]) - start_time,
                "sample_count": len(window),
                "max_per_joint_target_range_rad": max_range,
                "threshold_rad": PHYSICAL_STOP_TARGET_STABILITY_RANGE_RAD,
                "window_s": PHYSICAL_STOP_TARGET_STABILITY_WINDOW_S,
            }
    return {
        "status": "NOT_OBSERVED",
        "threshold_rad": PHYSICAL_STOP_TARGET_STABILITY_RANGE_RAD,
        "window_s": PHYSICAL_STOP_TARGET_STABILITY_WINDOW_S,
    }


def evaluate_physical_stop_gate(
    samples: list[Mapping[str, Any]], *, zero_start_monotonic_s: float,
    stand_confirmed_monotonic_s: float | None,
    baseline_state_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require a fresh two-second hold and report physical/target gates separately.

    The final observed two-second window is the acceptance window. The legacy
    one-second empirical forward-drift gate remains separate. Top-level status
    stays composite for existing callers: both physical state and controller
    target stability must pass.
    """
    criteria = {
        "window_s": PHYSICAL_STOP_WINDOW_S,
        "minimum_observed_window_s": PHYSICAL_STOP_MIN_WINDOW_S,
        "minimum_state_samples": PHYSICAL_STOP_MIN_SAMPLES,
        "max_sample_gap_s": PHYSICAL_STOP_MAX_SAMPLE_GAP_S,
        "max_state_age_s": PHYSICAL_STOP_MAX_STATE_AGE_S,
        "max_planar_speed_m_s": PHYSICAL_STOP_MAX_PLANAR_SPEED_M_S,
        "max_vertical_speed_m_s": PHYSICAL_STOP_MAX_VERTICAL_SPEED_M_S,
        "max_planar_endpoint_and_peak_excursion_m": PHYSICAL_STOP_MAX_PLANAR_DRIFT_M,
        "max_planar_path_m": PHYSICAL_STOP_MAX_PLANAR_PATH_M,
        "max_yaw_excursion_deg": math.degrees(PHYSICAL_STOP_MAX_YAW_DRIFT_RAD),
        "max_yaw_fit_rate_deg_s": math.degrees(PHYSICAL_STOP_MAX_YAW_RATE_RAD_S),
        "max_root_height_range_m": PHYSICAL_STOP_MAX_HEIGHT_RANGE_M,
        "max_tilt_deg": math.degrees(PHYSICAL_STOP_MAX_TILT_RAD),
        "max_roll_pitch_range_deg": math.degrees(PHYSICAL_STOP_MAX_ROLL_PITCH_RANGE_RAD),
        "max_leg_joint_speed_rad_s": PHYSICAL_STOP_MAX_JOINT_SPEED_RAD_S,
        "max_leg_joint_position_range_rad": PHYSICAL_STOP_MAX_JOINT_RANGE_RAD,
        "max_leg_target_range_rad": PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD,
    }
    zero_time = float(zero_start_monotonic_s)
    result: dict[str, Any] = {
        "status": "NOT_PASSED",
        "name": "PHYSICAL_STOP_GATE",
        "zero_start_monotonic_s": zero_time,
        "stand_confirmed_monotonic_s": stand_confirmed_monotonic_s,
        "stand_readback_delay_s": (
            float(stand_confirmed_monotonic_s) - zero_time
            if stand_confirmed_monotonic_s is not None else None
        ),
        "criteria": criteria,
        "criteria_rationale": {
            "xy_position": "5 mm is 4.2% of the offline point-goal adapter's 0.12 m position tolerance; same-run stationary one-second XY endpoint noise is also reported below.",
            "planar_velocity": "5 mm/s limits additional travel to 10 mm over the 2 s hold, 8.3% of the same point-goal tolerance.",
            "heading": "2 deg excursion is 29% of the offline adapter's 0.12 rad yaw tolerance; the 1 deg/s fit bound constrains drift over the complete hold.",
            "joint_and_pose": "Limits are independent physical-settling acceptance bounds, calibrated against the same-run no-command noise; existing fall, joint-position, and joint-velocity safety checks remain active and unchanged.",
            "sampling": "MuJoCo odometry is configured at 500 Hz; this observer samples around 17-50 Hz. The 2 s window and 20-sample minimum span many simulator updates and reject stale/gapped telemetry.",
            "navigation_reference": "The 0.12 m / 0.12 rad values are defaults in the proposal-only Issue 80 point-goal adapter, not an integrated or manufacturer acceptance contract.",
        },
        "baseline_state_summary": dict(baseline_state_summary or {}),
        "joint_target_stability_after_zero": _first_stable_joint_target_window(
            samples, after_monotonic_s=zero_time,
        ),
        "legacy_baseline_stop_gate": "unchanged; see post_stop_forward_tail.status",
        "physical_state_gate": {"status": "NOT_EVALUATED"},
        "controller_target_stability_gate": {"status": "NOT_EVALUATED"},
        "joint_window_diagnostics": {"status": "NOT_EVALUATED"},
        "mc_zero_receipt_ack": False,
        "mc_stop_ack": False,
        "acknowledgement_interpretation": "No installed per-command MC receipt/stop acknowledgement is exposed; this gate uses observed simulator physics and does not fabricate an ACK.",
        "foot_contact_force_or_slip_telemetry": "not exported by the installed simulator ROS interface; see the separate offline collision-geometry replay and its limitations.",
    }
    if stand_confirmed_monotonic_s is None or not math.isfinite(float(stand_confirmed_monotonic_s)):
        result["reason"] = "stable_STAND_FSM_readback_not_confirmed"
        return result

    post_rows = sorted(
        (row for row in samples if row.get("phase") == "post-stop"
         and isinstance(row.get("pose"), Mapping)),
        key=lambda row: float(row.get("sample_monotonic_s", -math.inf)),
    )
    # A transient source-service outage is observable but cannot contribute to
    # acceptance. If service readback later recovers, only one uninterrupted
    # readback epoch can satisfy the complete post-STAND physical hold.
    eligible_epochs = [
        int(row.get("post_stop_readback_epoch", 0))
        for row in post_rows
        if row.get("post_stop_gate_eligible", True)
    ]
    latest_epoch = max(eligible_epochs) if eligible_epochs else None
    post_rows = [
        row for row in post_rows
        if row.get("post_stop_gate_eligible", True)
        and (latest_epoch is None or int(row.get("post_stop_readback_epoch", 0)) == latest_epoch)
    ]
    result["readback_epoch_used"] = latest_epoch
    result["readback_gap_samples_excluded"] = sum(
        1 for row in samples
        if row.get("phase") == "post-stop" and not row.get("post_stop_gate_eligible", True)
    )
    after_stand = [
        row for row in post_rows
        if float(row.get("sample_monotonic_s", -math.inf)) >= float(stand_confirmed_monotonic_s)
    ]
    if not after_stand:
        result["reason"] = "no_post_readback_physics_samples"
        return result
    end_time = float(after_stand[-1]["sample_monotonic_s"])
    final_start = end_time - PHYSICAL_STOP_WINDOW_S
    final_window = [
        row for row in after_stand
        if final_start <= float(row["sample_monotonic_s"]) <= end_time
    ]
    try:
        final_metrics = _physical_window_metrics(final_window)
    except (KeyError, TypeError, ValueError) as exc:
        result["reason"] = f"final_physical_window_invalid:{type(exc).__name__}:{exc}"
        result["final_window"] = {
            "start_monotonic_s": final_start,
            "end_monotonic_s": end_time,
            "sample_count": len(final_window),
        }
        return result
    baseline_body_status = (
        baseline_state_summary.get("body_status")
        if isinstance(baseline_state_summary, Mapping) else None
    )
    rolling_windows = _rolling_physical_stop_windows(
        after_stand,
        zero_time=zero_time,
        baseline_body_status=baseline_body_status,
    )
    result["rolling_post_stand_windows"] = rolling_windows
    result["rolling_post_stand_window_s"] = PHYSICAL_STOP_WINDOW_S
    result["rolling_post_stand_step_s"] = PHYSICAL_STOP_ROLLING_STEP_S
    result["first_target_stable_window"] = next(
        (window for window in rolling_windows
         if window.get("controller_target_stability_gate", {}).get("status") == "PASS"),
        None,
    )
    result["first_composite_stop_window"] = next(
        (window for window in rolling_windows
         if window.get("composite_gate", {}).get("status") == "PASS"),
        None,
    )
    final_violations = _physical_stop_violations(
        final_metrics, baseline_body_status=baseline_body_status,
    )
    result["final_window"] = final_metrics
    result["final_window_violations"] = final_violations
    physical_violations = [
        violation for violation in final_violations
        if violation != "joint_target_not_settled"
    ]
    target_violation = "joint_target_not_settled" in final_violations
    joint_diagnostics = _joint_stop_window_diagnostics(final_window)
    result["physical_state_gate"] = {
        "status": "FAIL" if physical_violations else "PASS",
        "violations": physical_violations,
        "includes": "confirmed STAND/FSM/body status, fresh telemetry, base pose/twist, and actual joint state",
    }
    target_limit = PHYSICAL_STOP_MAX_COMMAND_RANGE_RAD
    violating_targets = [
        item for item in joint_diagnostics["per_joint"]
        if item["target_range_rad"] > target_limit
    ]
    result["controller_target_stability_gate"] = {
        "status": "FAIL" if target_violation else "PASS",
        "threshold_rad": target_limit,
        "window_s": PHYSICAL_STOP_WINDOW_S,
        "observed_max_target_range_rad": final_metrics["max_joint_command_range_rad"],
        "violating_joints": [item["name"] for item in violating_targets],
        "one_second_tighter_target_window_is_diagnostic_only": True,
        "one_second_tighter_target_window": result["joint_target_stability_after_zero"],
    }
    result["joint_window_diagnostics"] = joint_diagnostics
    result["blocking_components"] = [
        name for name, status in (
            ("physical_state_gate", result["physical_state_gate"]["status"]),
            ("controller_target_stability_gate", result["controller_target_stability_gate"]["status"]),
        ) if status != "PASS"
    ]

    first_passing: dict[str, Any] | None = None
    first_passing_violations: list[str] = []
    for start_index, start_row in enumerate(after_stand):
        start = float(start_row["sample_monotonic_s"])
        candidate = [
            row for row in after_stand[start_index:]
            if float(row["sample_monotonic_s"]) <= start + PHYSICAL_STOP_WINDOW_S
        ]
        try:
            metrics = _physical_window_metrics(candidate)
        except (KeyError, TypeError, ValueError):
            continue
        violations = _physical_stop_violations(metrics, baseline_body_status=baseline_body_status)
        if not violations:
            first_passing = metrics
            break
        if not first_passing_violations:
            first_passing_violations = violations
    result["first_qualifying_window"] = first_passing
    result["first_qualifying_window_start_delay_from_zero_s"] = (
        first_passing["window_start_monotonic_s"] - zero_time
        if first_passing else None
    )
    result["physical_stop_latency_to_final_hold_start_s"] = final_metrics["window_start_monotonic_s"] - zero_time
    result["physical_stop_latency_to_final_hold_end_s"] = final_metrics["window_end_monotonic_s"] - zero_time

    anchor_row = next(
        (row for row in reversed(samples)
         if row.get("phase") in ("activation", "pulse-deadline-observation")
         and isinstance(row.get("pose"), Mapping)
         and float(row.get("sample_monotonic_s", math.inf)) <= zero_time),
        None,
    )
    if anchor_row is not None:
        anchor = anchor_row["pose"]
        trajectory = [
            row for row in samples
            if isinstance(row.get("pose"), Mapping)
            and float(row.get("sample_monotonic_s", -math.inf)) >= float(anchor_row["sample_monotonic_s"])
            and float(row.get("sample_monotonic_s", math.inf)) <= final_metrics["window_start_monotonic_s"]
        ]
        points = [(float(row["pose"]["x"]), float(row["pose"]["y"])) for row in trajectory]
        path = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
        excursions = [math.hypot(x - float(anchor["x"]), y - float(anchor["y"])) for x, y in points]
        forward = [
            forward_projected_displacement(anchor, row["pose"])
            for row in trajectory
        ]
        result["braking_travel_from_last_in_pulse_pose"] = {
            "anchor_monotonic_s": anchor_row["sample_monotonic_s"],
            "final_hold_start_monotonic_s": final_metrics["window_start_monotonic_s"],
            "elapsed_s": final_metrics["window_start_monotonic_s"] - float(anchor_row["sample_monotonic_s"]),
            "planar_path_length_m": path,
            "net_planar_displacement_m": math.hypot(
                points[-1][0] - points[0][0], points[-1][1] - points[0][1],
            ) if points else None,
            "max_planar_excursion_m": max(excursions) if excursions else None,
            "forward_projection_min_m": min(forward) if forward else None,
            "forward_projection_max_m": max(forward) if forward else None,
            "interpretation": "Braking/settling travel before the final physical hold; separate from residual drift measured inside final_window.",
        }
    else:
        result["braking_travel_from_last_in_pulse_pose"] = {"status": "ANCHOR_UNAVAILABLE"}

    result["status"] = "PASS" if not final_violations else "FAIL"
    if final_violations:
        result["reason"] = "final_two_second_physical_hold_outside_criteria"
    else:
        result["reason"] = "fresh_stand_readback_and_final_two_second_physical_hold_passed"
    return result


def stand_mode_readback_confirmed(snapshot: Mapping[str, Any]) -> bool:
    """Require both applied-mode readbacks and stable FSM before ending zero hold."""
    return (
        snapshot.get("get_mode_desc") == MODE_STAND
        and snapshot.get("common_mode_desc") == MODE_STAND
        and snapshot.get("get_mode_header_code") == 0
        and snapshot.get("get_mode_status_value") == 100
        and snapshot.get("common_mode_status_value") == 100
        and snapshot.get("fsm_state") == MC_FSM_STABLE
        and snapshot.get("body_status") == BODY_POSE_STAND
    )


def classify_turning_rotation(
    *, start_yaw_rad: float | None, end_yaw_rad: float | None,
    commanded_angular_velocity_rad_s: float, baseline_yaw_noise_rad: float | None,
) -> dict[str, Any]:
    """Require measured yaw in the commanded direction beyond baseline noise."""
    if (
        start_yaw_rad is None or end_yaw_rad is None
        or not math.isfinite(float(start_yaw_rad))
        or not math.isfinite(float(end_yaw_rad))
        or not math.isfinite(commanded_angular_velocity_rad_s)
        or commanded_angular_velocity_rad_s == 0.0
    ):
        return {"status": "TURN_ODOMETRY_UNAVAILABLE"}
    delta = math.atan2(
        math.sin(float(end_yaw_rad) - float(start_yaw_rad)),
        math.cos(float(end_yaw_rad) - float(start_yaw_rad)),
    )
    measured_noise = (
        abs(float(baseline_yaw_noise_rad))
        if baseline_yaw_noise_rad is not None and math.isfinite(float(baseline_yaw_noise_rad))
        else 0.0
    )
    threshold = max(TURN_MIN_HEADING_CHANGE_RAD, 5.0 * measured_noise)
    signed_delta = delta * math.copysign(1.0, commanded_angular_velocity_rad_s)
    status = (
        "TURN_DIRECTION_PASS" if signed_delta >= threshold
        else "TURN_OPPOSITE_DIRECTION" if signed_delta <= -threshold
        else "TURN_HEADING_CHANGE_TOO_SMALL"
    )
    return {
        "status": status,
        "start_yaw_rad": float(start_yaw_rad),
        "end_yaw_rad": float(end_yaw_rad),
        "signed_yaw_delta_rad": delta,
        "commanded_sign_yaw_delta_rad": signed_delta,
        "minimum_measured_yaw_delta_rad": threshold,
        "stationary_yaw_noise_bound_rad": measured_noise,
    }


def classify_turn_planar_motion(
    *, start_pose: Mapping[str, Any], end_pose: Mapping[str, Any],
    forward_velocity_m_s: float, pulse_duration_s: float,
    angular_velocity_rad_s: float = 0.0, forward_warmup_s: float = 0.0,
) -> dict[str, Any]:
    """Bound translation by the integrated commanded arc plus the existing 5 cm drift margin."""
    values = (forward_velocity_m_s, pulse_duration_s, angular_velocity_rad_s, forward_warmup_s)
    if not all(math.isfinite(float(value)) for value in values) or pulse_duration_s < 0.0:
        raise ValueError("turn planar-motion inputs must be finite with a non-negative duration")
    if forward_warmup_s < 0.0:
        raise ValueError("forward warmup duration must be non-negative")
    start_yaw = float(start_pose["yaw"])
    dx = float(end_pose["x"]) - float(start_pose["x"])
    dy = float(end_pose["y"]) - float(start_pose["y"])
    forward_projection = dx * math.cos(start_yaw) + dy * math.sin(start_yaw)
    lateral_projection = -dx * math.sin(start_yaw) + dy * math.cos(start_yaw)
    planar_displacement = math.hypot(dx, dy)
    # The measured sample window can exceed the command deadline by a few
    # milliseconds while the first post-deadline snapshot is captured.
    turn_duration_s = min(TURN_PULSE_S, max(0.0, pulse_duration_s - forward_warmup_s))
    expected_forward = forward_velocity_m_s * pulse_duration_s
    expected_lateral = 0.0
    if turn_duration_s > 0.0 and angular_velocity_rad_s != 0.0 and forward_velocity_m_s != 0.0:
        step_count = max(1, math.ceil(turn_duration_s / 0.005))
        step_s = turn_duration_s / step_count
        heading_delta = 0.0
        for index in range(step_count):
            midpoint_s = (index + 0.5) * step_s
            angular_midpoint = smooth_turn_velocity(
                angular_velocity_rad_s, midpoint_s, turn_duration_s,
            )
            expected_lateral += (
                forward_velocity_m_s * math.sin(heading_delta + 0.5 * angular_midpoint * step_s) * step_s
            )
            heading_delta += angular_midpoint * step_s
    allowed_planar = abs(expected_forward) + TURN_MAX_PLANAR_DISPLACEMENT_M
    allowed_lateral = abs(expected_lateral) + TURN_MAX_PLANAR_DISPLACEMENT_M
    unexpected_reverse = (
        abs(forward_velocity_m_s) > 1e-12
        and forward_projection * math.copysign(1.0, forward_velocity_m_s) < -allowed_lateral
    )
    within = (
        planar_displacement <= allowed_planar
        and abs(lateral_projection) <= allowed_lateral
        and not unexpected_reverse
    )
    return {
        "status": "PASS" if within else "PLANAR_MOTION_OUTSIDE_COMMAND_ENVELOPE",
        "forward_velocity_command_m_s": forward_velocity_m_s,
        "pulse_duration_s": pulse_duration_s,
        "expected_forward_displacement_m": expected_forward,
        "expected_lateral_displacement_m": expected_lateral,
        "measured_forward_projection_m": forward_projection,
        "measured_lateral_projection_m": lateral_projection,
        "measured_planar_displacement_m": planar_displacement,
        "allowed_planar_displacement_m": allowed_planar,
        "allowed_lateral_drift_m": allowed_lateral,
        "unexpected_reverse_motion": unexpected_reverse,
        "lateral_direction_matches_expected_arc": (
            expected_lateral == 0.0
            or lateral_projection * expected_lateral >= 0.0
            or abs(lateral_projection) <= TURN_MAX_PLANAR_DISPLACEMENT_M
        ),
        "margin_basis": "integrated commanded-arc envelope plus the unchanged 0.05 m unintended-drift allowance",
    }


class SustainedYawRateWindow:
    """Measure net heading rate on monotonic receive time and retain source-stamp anomalies."""

    def __init__(self, window_s: float = TURN_YAW_RATE_WINDOW_S) -> None:
        if not math.isfinite(window_s) or window_s <= 0.0:
            raise ValueError("yaw-rate window must be finite and positive")
        self.window_s = window_s
        self.samples: list[tuple[float, float]] = []
        self.max_abs_window_rate_rad_s = 0.0
        self.max_abs_raw_twist_rad_s = 0.0
        self.last_source_stamp_s: float | None = None
        self.source_stamp_sample_count = 0
        self.source_stamp_regression_count = 0
        self.source_stamp_duplicate_count = 0
        self.first_source_stamp_regression: dict[str, Any] | None = None
        self.first_source_stamp_duplicate: dict[str, Any] | None = None

    def update(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        pose = snapshot.get("pose")
        if not isinstance(pose, Mapping):
            return {"status": "UNAVAILABLE", "reason": "pose_missing"}
        try:
            yaw = float(pose["yaw"])
            receive_time = float(snapshot["sample_monotonic_s"])
        except (KeyError, TypeError, ValueError):
            return {"status": "UNAVAILABLE", "reason": "yaw_or_timestamp_missing"}
        if not math.isfinite(yaw) or not math.isfinite(receive_time):
            return {"status": "UNAVAILABLE", "reason": "nonfinite_yaw_or_receive_time"}

        source_stamp_s = None
        if "stamp_sec" in pose and "stamp_nanosec" in pose:
            try:
                source_stamp_s = float(pose["stamp_sec"]) + float(pose["stamp_nanosec"]) * 1e-9
            except (TypeError, ValueError):
                source_stamp_s = None
            if source_stamp_s is not None and math.isfinite(source_stamp_s):
                self.source_stamp_sample_count += 1
                if self.last_source_stamp_s is not None:
                    delta = source_stamp_s - self.last_source_stamp_s
                    if delta < -1e-9:
                        self.source_stamp_regression_count += 1
                        if self.first_source_stamp_regression is None:
                            self.first_source_stamp_regression = {
                                "previous_stamp_s": self.last_source_stamp_s,
                                "current_stamp_s": source_stamp_s,
                                "receive_monotonic_s": receive_time,
                            }
                    elif delta <= 1e-9:
                        self.source_stamp_duplicate_count += 1
                        if self.first_source_stamp_duplicate is None:
                            self.first_source_stamp_duplicate = {
                                "previous_stamp_s": self.last_source_stamp_s,
                                "current_stamp_s": source_stamp_s,
                                "receive_monotonic_s": receive_time,
                            }
                self.last_source_stamp_s = source_stamp_s

        raw_twist = pose.get("wz")
        if isinstance(raw_twist, (int, float)) and math.isfinite(float(raw_twist)):
            self.max_abs_raw_twist_rad_s = max(self.max_abs_raw_twist_rad_s, abs(float(raw_twist)))

        if self.samples:
            timestamp_delta = receive_time - self.samples[-1][0]
            if timestamp_delta < -1e-9:
                self.samples.clear()
                self.samples.append((receive_time, yaw))
                return {
                    "status": "HOST_MONOTONIC_RESET",
                    "time_basis": "host_monotonic_receive_time",
                    "window_s": self.window_s,
                    "coverage_s": 0.0,
                    "raw_twist_wz_rad_s": raw_twist,
                    "source_stamp_s": source_stamp_s,
                }
            if timestamp_delta <= 1e-9:
                return {
                    "status": "DUPLICATE_SAMPLE",
                    "time_basis": "host_monotonic_receive_time",
                    "window_s": self.window_s,
                    "coverage_s": max(0.0, receive_time - self.samples[0][0]),
                    "raw_twist_wz_rad_s": raw_twist,
                    "source_stamp_s": source_stamp_s,
                }
        self.samples.append((receive_time, yaw))
        cutoff = receive_time - self.window_s
        while len(self.samples) > 2 and self.samples[1][0] <= cutoff:
            self.samples.pop(0)
        start_time, start_yaw = self.samples[0]
        coverage = receive_time - start_time
        if coverage < self.window_s - 0.01:
            return {
                "status": "WARMING_UP",
                "time_basis": "host_monotonic_receive_time",
                "window_s": self.window_s,
                "coverage_s": max(0.0, coverage),
                "raw_twist_wz_rad_s": raw_twist,
                "source_stamp_s": source_stamp_s,
            }
        yaw_delta = math.atan2(math.sin(yaw - start_yaw), math.cos(yaw - start_yaw))
        rate = yaw_delta / coverage
        self.max_abs_window_rate_rad_s = max(self.max_abs_window_rate_rad_s, abs(rate))
        return {
            "status": "MEASURED",
            "time_basis": "host_monotonic_receive_time",
            "window_s": self.window_s,
            "coverage_s": coverage,
            "yaw_delta_rad": yaw_delta,
            "signed_net_yaw_rate_rad_s": rate,
            "absolute_net_yaw_rate_rad_s": abs(rate),
            "raw_twist_wz_rad_s": raw_twist,
            "source_stamp_s": source_stamp_s,
            "source_stamp_regression_count": self.source_stamp_regression_count,
            "source_stamp_duplicate_count": self.source_stamp_duplicate_count,
        }


def _maximum_turn_origin_displacement(
    samples: list[Mapping[str, Any]],
    turn_start_snapshot: Mapping[str, Any] | None,
    turn_start_monotonic_s: float | None,
) -> float | None:
    """Return the largest observed planar radius through active and stop samples."""
    origin = turn_start_snapshot.get("pose") if isinstance(turn_start_snapshot, Mapping) else None
    if not isinstance(origin, Mapping) or turn_start_monotonic_s is None:
        return None
    maximum = 0.0
    for row in samples:
        if row.get("phase") not in {"turn-activation", "post-stop"}:
            continue
        try:
            sample_time = float(row["sample_monotonic_s"])
            pose = row["pose"]
            if sample_time < turn_start_monotonic_s or not isinstance(pose, Mapping):
                continue
            distance = math.hypot(
                float(pose["x"]) - float(origin["x"]),
                float(pose["y"]) - float(origin["y"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        maximum = max(maximum, distance)
    return maximum


def run_angular_turn_window(
    *, sample: Callable[[], Mapping[str, Any]],
    publish_turn: Callable[[float, float], None], publish_zero: Callable[[], None],
    registration: Mapping[str, Any], set_action: Mapping[str, Any],
    joint_limits: Mapping[str, Mapping[str, float]],
    baseline_forward_displacements_m: list[float] | None,
    baseline_state_summary: Mapping[str, Any] | None,
    authorized: bool, angular_velocity_rad_s: float,
    forward_velocity_m_s: float = 0.0,
    forward_warmup_s: float = 0.0,
    pulse_duration_s: float = TURN_PULSE_S,
    pre_deadline_zero_margin_s: float = 0.0,
    max_observed_planar_speed_m_s: float = 0.2,
    zero_until_stand_timeout_s: float = 3.5,
    extra_post_stop_observe_s: float = 5.0,
    state_timeout_s: float = 0.5,
    min_root_z: float = 0.52,
    max_tilt_deg: float = 15.0,
    max_turn_displacement_m: float | None = None,
    turn_stop_reserve_m: float = 0.0,
    termination_condition: Callable[[Mapping[str, Any]], str | None] | None = None,
    shutdown_event: threading.Event | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    timer_factory: Callable[[float, Callable[[], None]], Any] | None = None,
    on_sample: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one bounded forward-arc turn with measured yaw and physical settling checks."""
    result: dict[str, Any] = {
        "status": "NOT_RUN",
        "command_kind": "YAW_ONLY_TURN" if forward_velocity_m_s == 0.0 else "FORWARD_ARC_TURN",
        "source": VELOCITY_SOURCE_NAME,
        "angular_velocity_rad_s": angular_velocity_rad_s,
        "forward_velocity_m_s": forward_velocity_m_s,
        "forward_warmup_s": forward_warmup_s,
        "lateral_velocity_m_s": 0.0,
        "pulse_duration_s": pulse_duration_s,
        "total_command_duration_s": forward_warmup_s + pulse_duration_s,
        "pre_deadline_zero_margin_s": pre_deadline_zero_margin_s,
        "max_turn_displacement_m": max_turn_displacement_m,
        "turn_stop_reserve_m": turn_stop_reserve_m,
        "turn_stop_trigger_displacement_m": (
            max_turn_displacement_m - turn_stop_reserve_m
            if max_turn_displacement_m is not None else None
        ),
        "ramp_duration_s_each_end": TURN_RAMP_S,
        "publish_hz": PUBLISH_HZ,
        "angular_command_limit_rad_s": TURN_COMMAND_RAD_S,
        "max_sustained_net_yaw_rate_rad_s": TURN_MAX_SUSTAINED_YAW_RATE_RAD_S,
        "yaw_rate_window_s": TURN_YAW_RATE_WINDOW_S,
        "max_observed_planar_speed_m_s": max_observed_planar_speed_m_s,
        "source_timeout_ms": SOURCE_TIMEOUT_MS,
        "registration": dict(registration),
        "set_mc_action": dict(set_action),
        "samples": [],
        "fault": None,
        "mc_zero_receipt_ack": False,
        "mc_stop_ack": False,
    }
    validation_fault = None
    if not authorized:
        result["status"] = "NOT_AUTHORIZED"
        return result
    if not math.isfinite(forward_velocity_m_s) or not 0.0 <= forward_velocity_m_s <= 0.2:
        validation_fault = "turn_forward_command_outside_fixed_0p2_m_s_cap"
    elif not math.isfinite(forward_warmup_s) or not 0.0 <= forward_warmup_s <= 1.0:
        validation_fault = "turn_forward_warmup_outside_fixed_0_to_1_second_cap"
    elif not math.isfinite(angular_velocity_rad_s) or not 0.0 < abs(angular_velocity_rad_s) <= TURN_COMMAND_RAD_S:
        validation_fault = "turn_command_outside_fixed_0p075_rad_s_cap"
    elif not math.isfinite(pulse_duration_s) or not 0.0 < pulse_duration_s <= TURN_PULSE_S:
        validation_fault = "turn_duration_outside_fixed_4p5_second_cap"
    elif forward_warmup_s + pulse_duration_s > TURN_MAX_COMMAND_WINDOW_S:
        validation_fault = "turn_command_window_exceeds_fixed_5_second_per_direction_cap"
    elif (
        not math.isfinite(pre_deadline_zero_margin_s)
        or not 0.0 <= pre_deadline_zero_margin_s <= 0.1
        or pre_deadline_zero_margin_s >= forward_warmup_s + pulse_duration_s
    ):
        validation_fault = "pre_deadline_zero_margin_outside_small_bounded_range"
    elif not math.isfinite(max_observed_planar_speed_m_s) or not 0.0 < max_observed_planar_speed_m_s <= 0.2:
        validation_fault = "planar_speed_guard_outside_existing_0p2_m_s_cap"
    elif not math.isfinite(zero_until_stand_timeout_s) or not 0.0 < zero_until_stand_timeout_s <= 3.5:
        validation_fault = "zero_until_stand_timeout_outside_existing_budget"
    elif not math.isfinite(extra_post_stop_observe_s) or not 0.0 <= extra_post_stop_observe_s <= 5.0:
        validation_fault = "post_stop_observation_outside_existing_budget"
    elif registration.get("success") is not True:
        validation_fault = "source_registration_rejected"
    elif set_action.get("accepted_by_service") is not True:
        validation_fault = "set_mc_action_rejected"
    if not validation_fault and max_turn_displacement_m is not None:
        if (
            not math.isfinite(max_turn_displacement_m)
            or max_turn_displacement_m <= 0.0
            or not math.isfinite(turn_stop_reserve_m)
            or not 0.0 <= turn_stop_reserve_m < max_turn_displacement_m
        ):
            validation_fault = "turn_displacement_budget_or_stop_reserve_invalid"
    elif not validation_fault and turn_stop_reserve_m != 0.0:
        validation_fault = "turn_stop_reserve_requires_a_displacement_budget"
    if validation_fault:
        result["status"] = "ABORTED"
        result["fault"] = validation_fault
        return result

    samples: list[dict[str, Any]] = result["samples"]
    command_lock = threading.RLock()
    stop_requested = threading.Event()
    tracker = ActivationTracker()
    yaw_rate_monitor = SustainedYawRateWindow(TURN_YAW_RATE_WINDOW_S)
    pulse_fault: str | None = None
    post_stop_fault: str | None = None
    active_stop_reason: str | None = None
    maximum_turn_displacement_observed_m = 0.0
    initial: Mapping[str, Any] | None = None
    last_active: Mapping[str, Any] | None = None
    pulse_start: float | None = None
    pulse_end: float | None = None
    turn_start_snapshot: Mapping[str, Any] | None = None
    turn_start_monotonic_s: float | None = None
    nonzero_publish_attempts = 0
    nonzero_publish_calls_returned = 0
    zero_extension_count = 0
    zero_extension_first: float | None = None
    zero_extension_last: float | None = None
    stand_confirmed_s: float | None = None
    stop_result: dict[str, Any] = {}
    zero_stop: ZeroStopWindow

    def annotate_yaw_rate(snapshot: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
        row = dict(snapshot)
        measurement = yaw_rate_monitor.update(row)
        row["turn_yaw_rate_window"] = measurement
        if measurement.get("status") == "HOST_MONOTONIC_RESET":
            return row, "yaw_host_monotonic_discontinuity"
        if measurement.get("status") == "UNAVAILABLE":
            return row, "yaw_measurement_unavailable"
        if (
            measurement.get("status") == "MEASURED"
            and float(measurement["absolute_net_yaw_rate_rad_s"]) > TURN_MAX_SUSTAINED_YAW_RATE_RAD_S
        ):
            return row, "sustained_yaw_rate_limit"
        return row, None

    def record(phase: str, snapshot: Mapping[str, Any], fault: str | None = None) -> None:
        row = dict(snapshot)
        row["phase"] = phase
        row["safety_check"] = "FAIL" if fault else "PASS"
        row["safety_fault"] = fault
        samples.append(row)
        if on_sample:
            on_sample(phase, row)

    def poll_during_zero_stop() -> None:
        """Keep ROS services and sensor callbacks flowing throughout zero publication."""
        nonlocal post_stop_fault
        current, yaw_rate_fault = annotate_yaw_rate(sample())
        current_fault = snapshot_fault(
            current, joint_limits, state_timeout_s=state_timeout_s,
            min_root_z=min_root_z, max_tilt_deg=max_tilt_deg,
            max_observed_planar_speed_m_s=max_observed_planar_speed_m_s,
        )
        if current.get("readback_fault"):
            current_fault = current_fault or "readback_fault:" + str(current.get("readback_fault"))
        if yaw_rate_fault:
            current_fault = current_fault or "post_stop_" + yaw_rate_fault
        if current_fault:
            post_stop_fault = post_stop_fault or current_fault
        tracker.observe(current, phase="post-stop")
        record("post-stop", current, current_fault)

    zero_stop = ZeroStopWindow(
        publish_zero,
        timeout_s=max(ZERO_STOP_MIN_S, SOURCE_TIMEOUT_MS / 1000.0) + 1.0 / PUBLISH_HZ,
        publish_hz=PUBLISH_HZ,
        clock=clock,
        sleep=sleep,
        command_lock=command_lock,
        during_zero_callback=poll_during_zero_stop,
    )
    watchdog: ProcessLocalTimerWatchdog | None = None

    def watchdog_stop() -> None:
        nonlocal active_stop_reason, pulse_fault
        scheduled_deadline = (
            pulse_start + forward_warmup_s + pulse_duration_s - pre_deadline_zero_margin_s
            if pulse_start is not None else None
        )
        if scheduled_deadline is not None and clock() < scheduled_deadline - 0.01:
            pulse_fault = pulse_fault or "independent_watchdog_timeout"
        elif scheduled_deadline is not None:
            active_stop_reason = "pre_deadline_zero_margin" if pre_deadline_zero_margin_s > 0.0 else "hard_deadline_watchdog"
        stop_requested.set()
        zero_stop.run_once()

    try:
        initial, initial_yaw_rate_fault = annotate_yaw_rate(sample())
        preflight_fault = snapshot_fault(
            initial, joint_limits, state_timeout_s=state_timeout_s,
            min_root_z=min_root_z, max_tilt_deg=max_tilt_deg,
            max_observed_planar_speed_m_s=max_observed_planar_speed_m_s,
        )
        if not stand_mode_readback_confirmed(initial):
            preflight_fault = preflight_fault or "turn_requires_fresh_stable_STAND_preflight"
        if initial.get("readback_fault"):
            preflight_fault = preflight_fault or "readback_fault:" + str(initial.get("readback_fault"))
        if initial_yaw_rate_fault:
            preflight_fault = preflight_fault or initial_yaw_rate_fault
        current_source = str(initial.get("common_source_name") or "")
        if current_source not in ("", VELOCITY_SOURCE_NAME):
            preflight_fault = preflight_fault or "conflicting_common_state_input_source"
        speed_status = initial.get("speed_status")
        angular_bounds = speed_status.get("angular_bounds") if isinstance(speed_status, Mapping) else None
        try:
            speed_mode = int(speed_status["speed_mode"])
            bound_min = float(angular_bounds["min_value"])
            bound_max = float(angular_bounds["max_value"])
            if speed_mode == 0 or not bound_min <= angular_velocity_rad_s <= bound_max:
                preflight_fault = preflight_fault or "turn_command_outside_live_angular_bounds"
            forward_bounds = speed_status.get("forward_bounds")
            if forward_velocity_m_s > 0.0:
                if not isinstance(forward_bounds, Mapping):
                    preflight_fault = preflight_fault or "live_forward_bounds_unavailable"
                elif not float(forward_bounds["min_value"]) <= forward_velocity_m_s <= float(forward_bounds["max_value"]):
                    preflight_fault = preflight_fault or "turn_forward_command_outside_live_forward_bounds"
        except (KeyError, TypeError, ValueError):
            speed_mode, bound_min, bound_max = None, None, None
            forward_bounds = None
            preflight_fault = preflight_fault or "live_angular_bounds_unavailable"
        result["live_speed_status_preflight"] = {
            "speed_mode": speed_mode,
            "angular_bounds": {"min_value": bound_min, "max_value": bound_max}
            if bound_min is not None else None,
            "forward_bounds": dict(forward_bounds) if isinstance(forward_bounds, Mapping) else None,
            "requested_angular_velocity_rad_s": angular_velocity_rad_s,
            "requested_forward_velocity_m_s": forward_velocity_m_s,
        }
        if preflight_fault:
            record("turn-preflight", initial, preflight_fault)
            pulse_fault = preflight_fault
        else:
            tracker.observe(initial, phase="pre-activation")
            record("turn-preflight", initial)
            pulse_start = clock()
            hard_deadline = pulse_start + forward_warmup_s + pulse_duration_s
            command_deadline = hard_deadline - pre_deadline_zero_margin_s
            next_publish = pulse_start
            watchdog = ProcessLocalTimerWatchdog(
                forward_warmup_s + pulse_duration_s - pre_deadline_zero_margin_s,
                watchdog_stop,
                timer_factory=timer_factory,
            )
            watchdog.start()
            while clock() < command_deadline:
                if shutdown_event is not None and shutdown_event.is_set():
                    pulse_fault = "operator_shutdown"
                    stop_requested.set()
                    break
                if stop_requested.is_set() or watchdog.expired:
                    break
                current, yaw_rate_fault = annotate_yaw_rate(sample())
                elapsed_s = max(0.0, clock() - pulse_start)
                warmup_active = elapsed_s < forward_warmup_s
                sample_phase = "turn-forward-warmup" if warmup_active else "turn-activation"
                current_fault = snapshot_fault(
                    current, joint_limits, state_timeout_s=state_timeout_s,
                    min_root_z=min_root_z, max_tilt_deg=max_tilt_deg,
                    max_observed_planar_speed_m_s=max_observed_planar_speed_m_s,
                )
                if current.get("readback_fault"):
                    current_fault = current_fault or "readback_fault:" + str(current.get("readback_fault"))
                if yaw_rate_fault:
                    current_fault = current_fault or yaw_rate_fault
                try:
                    current_bounds = current["speed_status"]["angular_bounds"]
                    if not float(current_bounds["min_value"]) <= angular_velocity_rad_s <= float(current_bounds["max_value"]):
                        current_fault = current_fault or "live_angular_bound_changed"
                except (KeyError, TypeError, ValueError):
                    current_fault = current_fault or "turn_angular_state_or_bounds_unavailable"
                if current_fault:
                    record(sample_phase, current, current_fault)
                    pulse_fault = current_fault
                    stop_requested.set()
                    break
                tracker.observe(current, phase="activation")
                if not warmup_active and turn_start_snapshot is None:
                    if forward_warmup_s > 0.0 and not tracker.control_handshake_confirmed:
                        pulse_fault = "turn_control_handshake_not_confirmed_after_forward_warmup"
                        record("turn-activation", current, pulse_fault)
                        stop_requested.set()
                        break
                    turn_start_snapshot = current
                    turn_start_monotonic_s = float(current.get("sample_monotonic_s", clock()))
                current = dict(current)
                if not warmup_active and turn_start_snapshot is not None and max_turn_displacement_m is not None:
                    turn_origin = turn_start_snapshot.get("pose")
                    current_pose = current.get("pose")
                    if not isinstance(turn_origin, Mapping) or not isinstance(current_pose, Mapping):
                        pulse_fault = "turn_displacement_guard_pose_unavailable"
                        record(sample_phase, current, pulse_fault)
                        stop_requested.set()
                        break
                    try:
                        turn_displacement = math.hypot(
                            float(current_pose["x"]) - float(turn_origin["x"]),
                            float(current_pose["y"]) - float(turn_origin["y"]),
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        pulse_fault = f"turn_displacement_guard_pose_invalid:{exc}"
                        record(sample_phase, current, pulse_fault)
                        stop_requested.set()
                        break
                    maximum_turn_displacement_observed_m = max(
                        maximum_turn_displacement_observed_m, turn_displacement
                    )
                    current["turn_origin_displacement_m"] = turn_displacement
                    current["turn_stop_trigger_displacement_m"] = (
                        max_turn_displacement_m - turn_stop_reserve_m
                    )
                    if turn_displacement >= max_turn_displacement_m - turn_stop_reserve_m:
                        active_stop_reason = "measured_turn_displacement_budget_reached"
                        current["controlled_stop_reason"] = active_stop_reason
                        last_active = current
                        record(sample_phase, current)
                        stop_requested.set()
                        pulse_end = clock()
                        break
                if termination_condition is not None:
                    try:
                        termination_reason = termination_condition(current)
                    except Exception as exc:
                        pulse_fault = f"turn_termination_condition_exception:{type(exc).__name__}:{exc}"
                        record(sample_phase, current, pulse_fault)
                        stop_requested.set()
                        break
                    if termination_reason:
                        active_stop_reason = str(termination_reason)
                        current["controlled_stop_reason"] = active_stop_reason
                        last_active = current
                        record(sample_phase, current)
                        stop_requested.set()
                        pulse_end = clock()
                        break
                record(sample_phase, current)
                now = clock()
                if shutdown_event is not None and shutdown_event.is_set():
                    pulse_fault = "operator_shutdown"
                    stop_requested.set()
                    break
                if now >= command_deadline:
                    active_stop_reason = (
                        "pre_deadline_zero_margin"
                        if pre_deadline_zero_margin_s > 0.0 else "hard_command_deadline"
                    )
                    pulse_end = clock()
                    break
                if stop_requested.is_set() or watchdog.expired:
                    break
                if now >= next_publish:
                    turn_elapsed_s = max(0.0, clock() - pulse_start - forward_warmup_s)
                    command = (
                        0.0 if warmup_active
                        else smooth_turn_velocity(angular_velocity_rad_s, turn_elapsed_s, pulse_duration_s)
                    )
                    with command_lock:
                        if stop_requested.is_set() or clock() >= command_deadline:
                            break
                        nonzero_command = forward_velocity_m_s != 0.0 or command != 0.0
                        if nonzero_command:
                            nonzero_publish_attempts += 1
                        publish_turn(forward_velocity_m_s, command)
                        if nonzero_command:
                            nonzero_publish_calls_returned += 1
                            if clock() > command_deadline:
                                pulse_fault = pulse_fault or "nonzero_turn_publish_returned_after_scheduled_zero_deadline"
                                stop_requested.set()
                    if stop_requested.is_set():
                        break
                    next_publish += 1.0 / PUBLISH_HZ
                    if next_publish < now:
                        next_publish = now + 1.0 / PUBLISH_HZ
                last_active = current
                remaining = command_deadline - clock()
                if remaining > 0.0:
                    sleep(min(0.005, remaining))
            if pulse_end is None:
                pulse_end = clock()
                if active_stop_reason is None and pulse_end >= command_deadline:
                    active_stop_reason = (
                        "pre_deadline_zero_margin"
                        if pre_deadline_zero_margin_s > 0.0 else "hard_command_deadline"
                    )
    except Exception as exc:
        pulse_fault = pulse_fault or f"exception:{type(exc).__name__}:{exc}"
        stop_requested.set()
    finally:
        if pulse_start is not None and pulse_end is None:
            pulse_end = clock()
        stop_requested.set()
        stop_result = zero_stop.run_once()
        if watchdog is not None:
            watchdog.cancel()

    zero_start = float(stop_result.get("start_monotonic_s", clock()))
    zero_end = float(stop_result.get("end_monotonic_s", clock()))
    post_end = zero_end + POST_STOP_OBSERVE_S + extra_post_stop_observe_s
    stand_deadline = zero_end + zero_until_stand_timeout_s
    next_zero = zero_end
    while clock() < post_end:
        current, yaw_rate_fault = annotate_yaw_rate(sample())
        current_fault = snapshot_fault(
            current, joint_limits, state_timeout_s=state_timeout_s,
            min_root_z=min_root_z, max_tilt_deg=max_tilt_deg,
            max_observed_planar_speed_m_s=max_observed_planar_speed_m_s,
        )
        if current.get("readback_fault"):
            current_fault = current_fault or "readback_fault:" + str(current.get("readback_fault"))
        if yaw_rate_fault:
            current_fault = current_fault or "post_stop_" + yaw_rate_fault
        now = clock()
        if current_fault:
            post_stop_fault = current_fault
            record("post-stop", current, current_fault)
            ZeroStopWindow(
                publish_zero, timeout_s=ZERO_STOP_MIN_S + 1.0 / PUBLISH_HZ,
                publish_hz=PUBLISH_HZ, clock=clock, sleep=sleep,
                command_lock=command_lock,
                during_zero_callback=poll_during_zero_stop,
            ).run_once()
            break
        tracker.observe(current, phase="post-stop")
        record("post-stop", current)
        if stand_confirmed_s is None and stand_mode_readback_confirmed(current):
            stand_confirmed_s = float(current["sample_monotonic_s"])
        if stand_confirmed_s is None and now >= stand_deadline:
            post_stop_fault = post_stop_fault or "post_stop_stand_readback_timeout"
        if stand_confirmed_s is None and now >= next_zero:
            if zero_extension_first is None:
                zero_extension_first = now
            publish_zero()
            zero_extension_last = clock()
            zero_extension_count += 1
            next_zero = clock() + 1.0 / PUBLISH_HZ
        remaining = post_end - clock()
        if remaining > 0.0:
            sleep(min(0.05, remaining))

    motion_start_pose = initial.get("pose") if isinstance(initial, Mapping) else None
    turn_start_pose = turn_start_snapshot.get("pose") if isinstance(turn_start_snapshot, Mapping) else None
    last_pose = last_active.get("pose") if isinstance(last_active, Mapping) else None
    start_yaw = float(turn_start_pose["yaw"]) if isinstance(turn_start_pose, Mapping) else None
    end_yaw = float(last_pose["yaw"]) if isinstance(last_pose, Mapping) else None
    rotation = classify_turning_rotation(
        start_yaw_rad=start_yaw,
        end_yaw_rad=end_yaw,
        commanded_angular_velocity_rad_s=angular_velocity_rad_s,
        baseline_yaw_noise_rad=(
            baseline_state_summary.get("maxima", {}).get("max_abs_yaw_excursion_rad")
            if isinstance(baseline_state_summary, Mapping) else None
        ),
    )
    planar_displacement = None
    planar_motion: dict[str, Any] = {"status": "TURN_ODOMETRY_UNAVAILABLE"}
    if isinstance(motion_start_pose, Mapping) and isinstance(last_pose, Mapping):
        actual_duration = (
            max(0.0, pulse_end - pulse_start)
            if pulse_start is not None and pulse_end is not None else 0.0
        )
        planar_motion = classify_turn_planar_motion(
            start_pose=motion_start_pose,
            end_pose=last_pose,
            forward_velocity_m_s=forward_velocity_m_s,
            pulse_duration_s=actual_duration,
            angular_velocity_rad_s=angular_velocity_rad_s,
            forward_warmup_s=forward_warmup_s,
        )
        planar_displacement = float(planar_motion["measured_planar_displacement_m"])
    baseline_values = []
    if baseline_forward_displacements_m is not None:
        try:
            baseline_values = [float(value) for value in baseline_forward_displacements_m]
        except (TypeError, ValueError):
            baseline_values = []
    baseline_noise = max((abs(value) for value in baseline_values), default=None)
    legacy_stop = measure_post_stop_forward_tail(samples, baseline_noise)
    physical_stop = evaluate_physical_stop_gate(
        samples,
        zero_start_monotonic_s=float(stop_result.get("start_monotonic_s", zero_start)),
        stand_confirmed_monotonic_s=stand_confirmed_s,
        baseline_state_summary=baseline_state_summary,
    )
    result.update({
        "pulse_start_monotonic_s": pulse_start,
        "pulse_end_monotonic_s": pulse_end,
        "pulse_actual_duration_s": max(0.0, pulse_end - pulse_start)
        if pulse_start is not None and pulse_end is not None else 0.0,
        "turn_start_monotonic_s": turn_start_monotonic_s,
        "active_stop_reason": active_stop_reason or "bounded_command_window_elapsed",
        "turn_displacement_guard": {
            "maximum_turn_displacement_m": max_turn_displacement_m,
            "stop_reserve_m": turn_stop_reserve_m,
            "trigger_displacement_m": (
                max_turn_displacement_m - turn_stop_reserve_m
                if max_turn_displacement_m is not None else None
            ),
            "maximum_observed_turn_displacement_m": maximum_turn_displacement_observed_m,
            "maximum_total_displacement_including_stop_m": _maximum_turn_origin_displacement(
                samples, turn_start_snapshot, turn_start_monotonic_s,
            ),
            "controlled_stop_triggered": active_stop_reason == "measured_turn_displacement_budget_reached",
            "evidence_basis": "measured planar displacement from the first post-warmup turn pose; the stop reserve remains available for zero-command deceleration",
        },
        "nonzero_publish_attempts": nonzero_publish_attempts,
        "nonzero_publish_calls_returned": nonzero_publish_calls_returned,
        "control_handshake_confirmed": tracker.control_handshake_confirmed,
        "control_handshake_status": tracker.classify_control_handshake(),
        "control_handshake_evidence": [tracker.samples[index] for index in tracker.handshake_sample_indices],
        "rotation": rotation,
        "yaw_rate_gate": {
            "metric": "net wrapped yaw displacement over host-monotonic receive time",
            "window_s": TURN_YAW_RATE_WINDOW_S,
            "maximum_sustained_net_yaw_rate_rad_s": TURN_MAX_SUSTAINED_YAW_RATE_RAD_S,
            "maximum_observed_windowed_yaw_rate_rad_s": yaw_rate_monitor.max_abs_window_rate_rad_s,
            "maximum_raw_twist_wz_rad_s": yaw_rate_monitor.max_abs_raw_twist_rad_s,
            "raw_twist_peaks_are_recorded_but_not_used_as_a_single-sample_abort": True,
            "source_odometry_stamp_integrity": {
                "status": (
                    "UNAVAILABLE" if yaw_rate_monitor.source_stamp_sample_count == 0
                    else "NONMONOTONIC"
                    if yaw_rate_monitor.source_stamp_regression_count or yaw_rate_monitor.source_stamp_duplicate_count
                    else "MONOTONIC"
                ),
                "sample_count": yaw_rate_monitor.source_stamp_sample_count,
                "regression_count": yaw_rate_monitor.source_stamp_regression_count,
                "duplicate_count": yaw_rate_monitor.source_stamp_duplicate_count,
                "first_regression": yaw_rate_monitor.first_source_stamp_regression,
                "first_duplicate": yaw_rate_monitor.first_source_stamp_duplicate,
                "source_stamps_are_not_used_for_the_local_physical_rate_gate": True,
            },
        },
        "planar_displacement_during_turn_m": planar_displacement,
        "planar_motion": planar_motion,
        "stand_confirmed_monotonic_s": stand_confirmed_s,
        "zero_stop": stop_result,
        "zero_until_stand": {
            "requested_timeout_s": zero_until_stand_timeout_s,
            "stand_confirmed": stand_confirmed_s is not None,
            "stand_confirmed_monotonic_s": stand_confirmed_s,
            "zero_publish_calls_returned": zero_extension_count,
            "first_publish_monotonic_s": zero_extension_first,
            "last_publish_monotonic_s": zero_extension_last,
            "mc_receipt_confirmed": False,
        },
        "legacy_baseline_stop_gate": {
            "status": legacy_stop.get("status"),
            "tail_window": legacy_stop,
            "interpretation": "Unchanged empirical baseline gate; it remains visible independently of PHYSICAL_STOP_GATE.",
        },
        "physical_stop_gate": physical_stop,
        "post_stop_fault": post_stop_fault,
        "mc_zero_receipt_ack": False,
        "mc_stop_ack": False,
        "fault": pulse_fault or post_stop_fault,
    })
    if pulse_fault or post_stop_fault:
        result["status"] = "ABORTED"
    elif stop_result.get("publish_exceptions"):
        result["status"] = "ZERO_STOP_PUBLISH_EXCEPTION"
    elif stop_result.get("during_zero_callback_errors"):
        result["status"] = "ZERO_STOP_TELEMETRY_PUMP_ERROR"
    elif stop_result.get("zero_publish_calls_returned", 0) < ZERO_STOP_MIN_MESSAGES:
        result["status"] = "ZERO_STOP_CALL_COUNT_TOO_LOW"
    elif not tracker.control_handshake_confirmed:
        result["status"] = "TURN_CONTROL_HANDSHAKE_NOT_CONFIRMED"
    elif rotation.get("status") != "TURN_DIRECTION_PASS":
        result["status"] = rotation.get("status", "TURN_NOT_CONFIRMED")
    elif planar_motion.get("status") != "PASS":
        result["status"] = "TURN_PLANAR_DRIFT_EXCEEDED"
    elif physical_stop.get("status") != "PASS":
        result["status"] = "TURN_PHYSICAL_STOP_NOT_PASSED"
    else:
        result["status"] = "PHYSICAL_TURN_AND_STOP_PASS"
    return result


def smoothstep_velocity(
    start_m_s: float, target_m_s: float, elapsed_s: float, ramp_duration_s: float,
) -> float:
    """C1-continuous monotone velocity ramp, bounded by the tested target cap."""
    values = (start_m_s, target_m_s, elapsed_s, ramp_duration_s)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("velocity ramp values must be finite")
    if not 0.0 < start_m_s <= target_m_s <= FORWARD_COMMAND_M_S:
        raise ValueError("velocity ramp must be positive, monotone, and under the command cap")
    if ramp_duration_s <= 0.0:
        raise ValueError("velocity ramp duration must be positive")
    progress = max(0.0, min(1.0, elapsed_s / ramp_duration_s))
    blend = progress * progress * (3.0 - 2.0 * progress)
    return start_m_s + (target_m_s - start_m_s) * blend


def smoothstep_deceleration_velocity(
    start_m_s: float, elapsed_s: float, deceleration_duration_s: float,
    end_m_s: float = 0.0,
) -> float:
    """C1-continuous one-second-bounded reduction of a positive command to end speed."""
    values = (start_m_s, elapsed_s, deceleration_duration_s, end_m_s)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("deceleration values must be finite")
    if not 0.0 <= end_m_s < start_m_s <= FORWARD_COMMAND_M_S:
        raise ValueError("deceleration start must be positive and under the command cap")
    if not 0.0 < deceleration_duration_s <= 1.0:
        raise ValueError("deceleration duration must be in (0, 1] seconds")
    progress = max(0.0, min(1.0, elapsed_s / deceleration_duration_s))
    blend = progress * progress * (3.0 - 2.0 * progress)
    return end_m_s + (start_m_s - end_m_s) * (1.0 - blend)


def smooth_turn_velocity(angular_velocity_rad_s: float, elapsed_s: float, duration_s: float) -> float:
    """Bounded C1 angular command with short symmetric ramps for one turn probe."""
    if (
        not math.isfinite(angular_velocity_rad_s)
        or not math.isfinite(elapsed_s)
        or not math.isfinite(duration_s)
        or not 0.0 < abs(angular_velocity_rad_s) <= TURN_COMMAND_RAD_S
        or not 0.0 < duration_s <= TURN_PULSE_S
    ):
        raise ValueError("turn command must be within the fixed 0.075 rad/s, 4.5 s profile")
    ramp = min(TURN_RAMP_S, duration_s / 2.0)
    if elapsed_s < 0.0 or elapsed_s >= duration_s:
        return 0.0
    if elapsed_s < ramp:
        phase = elapsed_s / ramp
    elif elapsed_s > duration_s - ramp:
        phase = (duration_s - elapsed_s) / ramp
    else:
        phase = 1.0
    blend = phase * phase * (3.0 - 2.0 * phase)
    return math.copysign(abs(angular_velocity_rad_s) * blend, angular_velocity_rad_s)


def startup_then_cruise_velocity(
    startup_m_s: float,
    cruise_m_s: float,
    forward_displacement_m: float,
    transition_displacement_m: float,
) -> float:
    """Use a bounded start command until measured forward motion, then cruise."""
    values = (startup_m_s, cruise_m_s, forward_displacement_m, transition_displacement_m)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("startup profile values must be finite")
    if not 0.0 < cruise_m_s < startup_m_s <= FORWARD_COMMAND_M_S:
        raise ValueError("startup speed must exceed cruise speed and remain under the command cap")
    if transition_displacement_m <= 0.0:
        raise ValueError("startup transition displacement must be positive")
    return cruise_m_s if forward_displacement_m >= transition_displacement_m else startup_m_s


def joint_command_array_snapshot(message: Any, received_monotonic_s: float | None = None) -> dict[str, Any] | None:
    """Keep the MC-to-simulator joint targets and header stamp in the run trace."""
    if message is None:
        return None

    def field(value: Any, name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    def numeric(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    header = field(message, "header")
    stamp = field(header, "stamp") if header is not None else None
    joints = field(message, "joints", ()) or ()
    records = []
    for joint in joints:
        records.append({
            "name": str(field(joint, "name", "")),
            "position": numeric(field(joint, "position")),
            "velocity": numeric(field(joint, "velocity")),
            "effort": numeric(field(joint, "effort")),
            "stiffness": numeric(field(joint, "stiffness")),
            "damping": numeric(field(joint, "damping")),
        })
    received = numeric(received_monotonic_s)
    return {
        "header_stamp": {
            "sec": int(field(stamp, "sec", 0)) if stamp is not None else None,
            "nanosec": int(field(stamp, "nanosec", 0)) if stamp is not None else None,
        },
        "received_monotonic_s": received,
        "joints": records,
    }


def float64_multi_array_snapshot(
    message: Any,
    received_monotonic_s: float | None = None,
) -> dict[str, Any] | None:
    """Record mc_debug_f64 payload and labels without assigning undocumented meanings."""
    if message is None:
        return None

    def field(value: Any, name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    layout = field(message, "layout")
    dimensions = field(layout, "dim", ()) if layout is not None else ()
    data = field(message, "data", ()) or ()
    values: list[float | None] = []
    nonfinite_indexes: list[int] = []
    for index, raw in enumerate(data):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            values.append(value)
        else:
            values.append(None)
            nonfinite_indexes.append(index)
    receive_time = None
    if received_monotonic_s is not None:
        try:
            parsed_receive_time = float(received_monotonic_s)
        except (TypeError, ValueError):
            parsed_receive_time = math.nan
        if math.isfinite(parsed_receive_time):
            receive_time = parsed_receive_time
    return {
        "received_monotonic_s": receive_time,
        "layout": {
            "data_offset": int(field(layout, "data_offset", 0)) if layout is not None else None,
            "dimensions": [
                {
                    "label": str(field(dimension, "label", "")),
                    "size": int(field(dimension, "size", 0)),
                    "stride": int(field(dimension, "stride", 0)),
                }
                for dimension in dimensions
            ],
        },
        "data": values,
        "nonfinite_indexes": nonfinite_indexes,
        "interpretation": (
            "raw std_msgs/Float64MultiArray values; do not assign field semantics "
            "unless the installed MC build provides matching documentation"
        ),
    }


def mc_rl_debug_snapshot(message: Any, received_monotonic_s: float | None = None) -> dict[str, Any] | None:
    """Keep the version-matched MC observation/action vectors and raw header."""
    if message is None:
        return None

    def field(value: Any, name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    def timestamp(value: Any) -> dict[str, int] | None:
        if value is None:
            return None
        return {
            "sec": int(field(value, "sec", 0)),
            "nanosec": int(field(value, "nanosec", 0)),
        }

    def vector(values: Any) -> tuple[list[float | None], list[int]]:
        result: list[float | None] = []
        invalid: list[int] = []
        for index, raw in enumerate(values or ()):
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                parsed = math.nan
            if math.isfinite(parsed):
                result.append(parsed)
            else:
                result.append(None)
                invalid.append(index)
        return result, invalid

    header = field(message, "header")
    stamp = field(header, "stamp") if header is not None else None
    measured = field(header, "meas_stamp") if header is not None else None
    receive_time = None
    if received_monotonic_s is not None:
        try:
            parsed_receive_time = float(received_monotonic_s)
        except (TypeError, ValueError):
            parsed_receive_time = math.nan
        if math.isfinite(parsed_receive_time):
            receive_time = parsed_receive_time
    obs, obs_invalid = vector(field(message, "obs_vec", ()))
    action, action_invalid = vector(field(message, "action_vec", ()))
    return {
        "received_monotonic_s": receive_time,
        "header": {
            "stamp": timestamp(stamp),
            "frame_id": str(field(header, "frame_id", "")) if header is not None else None,
            "sequence": int(field(header, "sequence", 0)) if header is not None else None,
            "meas_stamp": timestamp(measured),
        },
        "obs_vec": obs,
        "action_vec": action,
        "nonfinite_obs_indexes": obs_invalid,
        "nonfinite_action_indexes": action_invalid,
        "interpretation": (
            "Version-matched fields are named obs_vec/action_vec; vector indices are left raw "
            "until mapped to the matching FoundationWalk configuration."
        ),
    }


def run_first_motion_window(
    *,
    sample: Callable[[], Mapping[str, Any]],
    publish: Callable[[float], None],
    registration: Mapping[str, Any],
    set_action: Mapping[str, Any],
    joint_limits: Mapping[str, Mapping[str, float]],
    baseline_forward_displacements_m: list[float] | None,
    authorized: bool,
    baseline_state_summary: Mapping[str, Any] | None = None,
    forward_command_m_s: float = FORWARD_COMMAND_M_S,
    pulse_duration_s: float = PULSE_LIMIT_S,
    max_pulse_duration_s: float = PULSE_LIMIT_S,
    ramp_start_speed_m_s: float | None = None,
    ramp_duration_s: float = 0.0,
    deceleration_duration_s: float = 0.0,
    deceleration_end_speed_m_s: float = 0.0,
    startup_speed_m_s: float | None = None,
    startup_transition_displacement_m: float | None = None,
    startup_transition_timeout_s: float = 0.0,
    extra_post_stop_observe_s: float = 0.0,
    extra_post_stop_observe_limit_s: float = 5.0,
    after_zero_stop: Callable[[], Mapping[str, Any]] | None = None,
    after_zero_abort: Callable[[], Mapping[str, Any]] | None = None,
    during_zero_callback: Callable[[], None] | None = None,
    shutdown_event: threading.Event | None = None,
    on_sample: Callable[[str, Mapping[str, Any]], None] | None = None,
    termination_condition: Callable[[Mapping[str, Any]], bool] | None = None,
    termination_condition_name: str = "external_condition",
    state_timeout_s: float = 0.5,
    min_root_z: float = 0.52,
    max_tilt_deg: float = 15.0,
    max_observed_planar_speed_m_s: float | None = None,
    zero_until_stand_timeout_s: float = 0.0,
    require_locomotion_confirmation: bool = True,
    pre_motion_abort_reason: str | None = None,
    activation_lifecycle: MCLocomotionAbortLifecycle | None = None,
    activation_probe_only: bool = False,
    source_contract_supported: bool = False,
    source_grounded_route_activation: bool = False,
    source_grounded_activation_timeout_s: float = ACTIVATION_PROBE_DURATION_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    timer_factory: Callable[[float, Callable[[], None]], Any] | None = None,
) -> dict[str, Any]:
    """One bounded <=0.2 m/s, at most the supplied authorized duration, 50 Hz pulse."""
    result: dict[str, Any] = {
        "status": "NOT_RUN",
        "source": VELOCITY_SOURCE_NAME,
        "command_forward_m_s": forward_command_m_s,
        "command_lateral_m_s": 0.0,
        "command_angular_velocity_rad_s": 0.0,
        "pulse_limit_s": max_pulse_duration_s,
        "pulse_duration_s_requested": pulse_duration_s,
        "velocity_profile": {
            "shape": (
                "motion-triggered-startup-then-cruise" if startup_speed_m_s is not None
                else "constant-then-smoothstep-deceleration" if deceleration_duration_s > 0.0
                else "smoothstep" if ramp_duration_s > 0.0 else "constant"
            ),
            "ramp_start_speed_m_s": ramp_start_speed_m_s,
            "target_speed_m_s": forward_command_m_s,
            "ramp_duration_s": ramp_duration_s,
            "deceleration_duration_s": deceleration_duration_s,
            "deceleration_start_s": (
                pulse_duration_s - deceleration_duration_s
                if deceleration_duration_s > 0.0 else None
            ),
            "end_speed_m_s": (
                deceleration_end_speed_m_s if deceleration_duration_s > 0.0 else None
            ),
            "deceleration_end_speed_m_s": (
                deceleration_end_speed_m_s if deceleration_duration_s > 0.0 else None
            ),
            "startup_speed_m_s": startup_speed_m_s,
            "cruise_speed_m_s": forward_command_m_s if startup_speed_m_s is not None else None,
            "transition_displacement_m": startup_transition_displacement_m,
            "startup_timeout_s": startup_transition_timeout_s,
            "transition_observed": False,
            "transition_monotonic_s": None,
            "transition_displacement_observed_m": None,
        },
        "publish_hz": PUBLISH_HZ,
        "source_timeout_ms": SOURCE_TIMEOUT_MS,
        "zero_publish_semantics": ZERO_STOP_INTERPRETATION,
        "watchdog_scope_limitation": WATCHDOG_LIMITATION,
        "max_observed_planar_speed_m_s": max_observed_planar_speed_m_s,
        "zero_until_stand_timeout_s": zero_until_stand_timeout_s,
        "require_locomotion_confirmation": bool(require_locomotion_confirmation),
        "pre_motion_abort_reason": pre_motion_abort_reason,
        "activation_probe_only": bool(activation_probe_only),
        "source_contract_supported": bool(source_contract_supported),
        "source_grounded_route_activation": {
            "enabled": bool(source_grounded_route_activation),
            "maximum_qualification_s": source_grounded_activation_timeout_s
            if source_grounded_route_activation else None,
            "mode_source_confirmed": False,
            "confirmation_monotonic_s": None,
            "first_nonzero_is_part_of_this_window": bool(source_grounded_route_activation),
        },
        "activation_probe": {
            "enabled": bool(activation_probe_only),
            "forward_command_m_s": ACTIVATION_PROBE_FORWARD_M_S if activation_probe_only else None,
            "duration_cap_s": ACTIVATION_PROBE_DURATION_S if activation_probe_only else None,
            "mode_source_confirmed": False,
            "confirmation_monotonic_s": None,
            "terminal_reason": None,
        },
        "registration": dict(registration),
        "set_mc_action": dict(set_action),
        "nonzero_publish_calls_returned": 0,
        "samples": [],
        "fault": None,
        "after_zero_stop_action": None,
        "termination_condition_met": False,
        "termination_condition_name": None,
    }
    if not authorized:
        result["status"] = "NOT_AUTHORIZED"
        return result

    pulse_fault: str | None = pre_motion_abort_reason
    if activation_probe_only and activation_lifecycle is None:
        pulse_fault = pulse_fault or "activation_probe_requires_explicit_lifecycle_state"
    elif activation_probe_only and not source_contract_supported:
        pulse_fault = pulse_fault or "activation_probe_source_contract_not_supported"
    elif activation_probe_only and (
        abs(float(forward_command_m_s) - ACTIVATION_PROBE_FORWARD_M_S) > 1e-9
        or abs(float(pulse_duration_s) - ACTIVATION_PROBE_DURATION_S) > 1e-9
        or abs(float(max_pulse_duration_s) - ACTIVATION_PROBE_DURATION_S) > 1e-9
        or ramp_duration_s != 0.0
        or deceleration_duration_s != 0.0
        or startup_speed_m_s is not None
        or startup_transition_displacement_m is not None
        or termination_condition is not None
    ):
        pulse_fault = pulse_fault or "activation_probe_profile_must_remain_exactly_0p12_mps_for_1p8s"
    elif source_grounded_route_activation and activation_probe_only:
        pulse_fault = pulse_fault or "route_activation_cannot_use_probe_only_mode"
    elif source_grounded_route_activation and (
        activation_lifecycle is None
        or not source_contract_supported
        or abs(float(forward_command_m_s) - ACTIVATION_PROBE_FORWARD_M_S) > 1e-9
        or not math.isfinite(float(source_grounded_activation_timeout_s))
        or abs(float(source_grounded_activation_timeout_s) - ACTIVATION_PROBE_DURATION_S) > 1e-9
        or float(source_grounded_activation_timeout_s) > float(pulse_duration_s)
    ):
        pulse_fault = pulse_fault or "source_grounded_route_activation_profile_invalid"
    elif not activation_probe_only and source_contract_supported and not source_grounded_route_activation:
        pulse_fault = pulse_fault or "source_contract_flag_requires_explicit_activation_probe_mode"
    if (
        not math.isfinite(extra_post_stop_observe_limit_s)
        or not 0.0 <= extra_post_stop_observe_limit_s <= MAX_EXTRA_POST_STOP_OBSERVE_S
    ):
        pulse_fault = "extra_stop_observation_limit_outside_bounded_budget"
    elif (
        not math.isfinite(extra_post_stop_observe_s)
        or not 0.0 <= extra_post_stop_observe_s <= extra_post_stop_observe_limit_s
    ):
        pulse_fault = "extra_stop_observation_outside_bounded_budget"
    elif (
        not math.isfinite(zero_until_stand_timeout_s)
        or not 0.0 <= zero_until_stand_timeout_s <= 3.5
    ):
        pulse_fault = "zero_until_stand_timeout_outside_measured_3p5s_budget"
    elif (
        not math.isfinite(max_pulse_duration_s)
        or not 0.0 < max_pulse_duration_s <= MAX_AUTHORIZED_PULSE_S
        or not math.isfinite(pulse_duration_s)
        or not 0.0 < pulse_duration_s <= max_pulse_duration_s
    ):
        pulse_fault = "pulse_duration_outside_authorized_cap"
    elif not math.isfinite(forward_command_m_s) or not 0.0 < forward_command_m_s <= FORWARD_COMMAND_M_S:
        pulse_fault = "forward_command_outside_authorized_cap"
    elif max_observed_planar_speed_m_s is not None and (
        not math.isfinite(max_observed_planar_speed_m_s)
        or not 0.0 < max_observed_planar_speed_m_s <= FORWARD_COMMAND_M_S
    ):
        pulse_fault = "observed_planar_speed_gate_outside_authorized_cap"
    elif not math.isfinite(ramp_duration_s) or not 0.0 <= ramp_duration_s <= pulse_duration_s:
        pulse_fault = "velocity_ramp_duration_outside_pulse"
    elif (
        not math.isfinite(deceleration_duration_s)
        or not 0.0 <= deceleration_duration_s <= min(1.0, pulse_duration_s)
    ):
        pulse_fault = "velocity_deceleration_duration_outside_one_second_bound"
    elif (
        not math.isfinite(deceleration_end_speed_m_s)
        or not 0.0 <= deceleration_end_speed_m_s < forward_command_m_s
        or (deceleration_duration_s == 0.0 and deceleration_end_speed_m_s != 0.0)
    ):
        pulse_fault = "velocity_deceleration_end_speed_invalid"
    elif deceleration_duration_s > 0.0 and (
        ramp_duration_s > 0.0 or startup_speed_m_s is not None
    ):
        pulse_fault = "velocity_deceleration_profile_conflicts_with_startup_or_ramp"
    elif ramp_duration_s > 0.0 and (
        ramp_start_speed_m_s is None
        or not math.isfinite(ramp_start_speed_m_s)
        or not 0.0 < ramp_start_speed_m_s <= forward_command_m_s
    ):
        pulse_fault = "velocity_ramp_start_outside_command_range"
    elif startup_speed_m_s is None and (
        startup_transition_displacement_m is not None or startup_transition_timeout_s != 0.0
    ):
        pulse_fault = "startup_profile_parameters_incomplete"
    elif startup_speed_m_s is not None and (
        not math.isfinite(startup_speed_m_s)
        or not 0.0 < forward_command_m_s < startup_speed_m_s <= FORWARD_COMMAND_M_S
    ):
        pulse_fault = "startup_profile_speed_outside_command_range"
    elif startup_speed_m_s is not None and (
        ramp_duration_s > 0.0
        or startup_transition_displacement_m is None
        or not math.isfinite(startup_transition_displacement_m)
        or not 0.0 < startup_transition_displacement_m <= 0.10
        or not math.isfinite(startup_transition_timeout_s)
        or not 0.0 < startup_transition_timeout_s <= min(0.75, pulse_duration_s)
    ):
        pulse_fault = "startup_profile_transition_gate_invalid"
    elif not registration.get("success"):
        pulse_fault = "source_registration_rejected"
    elif not set_action.get("accepted_by_service"):
        pulse_fault = "set_mc_action_rejected"
    baseline_values: list[float] = []
    if baseline_forward_displacements_m is not None:
        try:
            baseline_values = [float(value) for value in baseline_forward_displacements_m]
        except (TypeError, ValueError):
            baseline_values = []
    if len(baseline_values) != BASELINE_WINDOW_COUNT or not all(math.isfinite(value) for value in baseline_values):
        pulse_fault = pulse_fault or "stationary_odom_baseline_unavailable"
    baseline_noise_bound_m = (
        max(abs(value) for value in baseline_values)
        if len(baseline_values) == BASELINE_WINDOW_COUNT
        and all(math.isfinite(value) for value in baseline_values)
        else None
    )
    result["baseline_forward_displacements_m"] = baseline_values
    result["baseline_noise_bound_m"] = baseline_noise_bound_m
    result["baseline_criterion"] = (
        "positive forward-projected pulse displacement must exceed the maximum absolute displacement "
        "of three no-command 1-second windows measured in this run; empirical envelope, not a vendor threshold or confidence bound"
    )
    result["mode_source_readback_skew_limit_s"] = MODE_SOURCE_READBACK_SKEW_MAX_S

    command_lock = threading.RLock()
    stop_requested = threading.Event()
    tracker = ActivationTracker()
    post_stop_first_pose: Mapping[str, Any] | None = None
    post_stop_last_pose: Mapping[str, Any] | None = None
    post_stop_first_observation: dict[str, Any] | None = None
    post_stop_fault: str | None = None
    post_stop_readback_epoch = 0
    post_stop_readback_gap_active = False
    post_stop_readback_events: list[dict[str, Any]] = []
    post_stop_readback_valid_monotonic_s: float | None = None
    post_stop_acceptance_gap_active = False
    post_stop_acceptance_gap_events: list[dict[str, Any]] = []

    def record_sample(phase: str, snapshot: Mapping[str, Any], fault: str | None) -> None:
        row = dict(snapshot)
        row["phase"] = phase
        if (
            activation_lifecycle is not None
            and activation_lifecycle.state == ZERO_AND_STOP_REQUESTED
            and not activation_lifecycle.locomotion_confirmed_before_stop
            and phase.startswith("post-stop")
            and (row.get("get_mode_desc") == MODE_LD or row.get("common_mode_desc") == MODE_LD)
        ):
            activation_lifecycle.observe_locomotion_readback(
                {
                    "classification": "LATE_LOCOMOTION_MODE_OBSERVED",
                    "reason": "GetMcAction or CommonState still reports LOCOMOTION after the abort/STAND recovery request",
                    "get_mode_desc": row.get("get_mode_desc"),
                    "common_mode_desc": row.get("common_mode_desc"),
                    "get_mode_status_value": row.get("get_mode_status_value"),
                    "common_mode_status_value": row.get("common_mode_status_value"),
                },
                now_s=float(row.get("sample_monotonic_s", clock())),
            )
        common_status_diagnostic = common_action_status_diagnostic(
            row.get("common_mode_status_value")
        )
        if common_status_diagnostic["classification"] in {
            "UNINTERRUPTIBLE_TRANSITION", "UNKNOWN_RUNTIME_STATUS",
        }:
            row["common_action_status_diagnostic"] = common_status_diagnostic
            if phase.startswith("post-stop"):
                row["post_stop_status_policy"] = (
                    "ZERO_ONLY_OBSERVATION; transition/unknown status never confirms STAND or rearms"
                )
                result.setdefault("post_stop_nonterminal_status_observations", []).append({
                    "sample_monotonic_s": row.get("sample_monotonic_s"),
                    "raw_value": common_status_diagnostic["raw_value"],
                    "classification": common_status_diagnostic["classification"],
                    "motion_rearm_allowed": False,
                })
        readback_gap = row.get("post_stop_source_readback_gap") if phase == "post-stop" else None
        row["safety_check"] = (
            "FAIL" if fault else "INCOMPLETE"
            if readback_gap
            or common_status_diagnostic["classification"] in {
                "UNINTERRUPTIBLE_TRANSITION", "UNKNOWN_RUNTIME_STATUS",
            }
            or (phase.startswith("post-stop") and not stand_mode_readback_confirmed(row))
            else "PASS"
        )
        row["safety_fault"] = fault or (
            f"required service readback unavailable: {readback_gap}" if readback_gap else None
        )
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
        during_zero_callback=during_zero_callback,
    )
    watchdog: ProcessLocalTimerWatchdog | None = None
    pulse_start: float | None = None
    pulse_end: float | None = None
    startup_origin_pose: Mapping[str, Any] | None = None
    startup_transition_observed = False
    nonzero_attempt_offsets_s: list[float] = []
    nonzero_success_offsets_s: list[float] = []

    def on_watchdog() -> None:
        nonlocal pulse_fault
        # This process-local timer thread is the hard stop only while this
        # Python process remains alive. An on-deadline firing is normal; an
        # early firing is a watchdog fault. Neither proves MC receipt or stop.
        if clock() < pulse_deadline - 0.01:
            pulse_fault = pulse_fault or "independent_watchdog_timeout"
        stop_requested.set()
        zero_stop.run_once()

    try:
        if activation_lifecycle is not None:
            if pulse_fault is None:
                if activation_probe_only or source_grounded_route_activation:
                    activation_lifecycle.begin_activation_probe(
                        source_contract_supported=source_contract_supported,
                        forward_command_m_s=forward_command_m_s,
                        duration_s=(
                            pulse_duration_s if activation_probe_only
                            else source_grounded_activation_timeout_s
                        ),
                        now_s=clock(),
                    )
                else:
                    activation_lifecycle.begin_motion(now_s=clock())
            else:
                activation_lifecycle.request_zero_and_stop(
                    now_s=clock(), reason=str(pulse_fault),
                )
        if pulse_fault is None:
            initial = sample()
            initial_fault = snapshot_fault(
                initial,
                joint_limits,
                state_timeout_s=state_timeout_s,
                min_root_z=min_root_z,
                max_tilt_deg=max_tilt_deg,
                max_observed_planar_speed_m_s=max_observed_planar_speed_m_s,
            )
            record_sample("pre-activation", initial, initial_fault)
            if initial_fault:
                pulse_fault = initial_fault
            else:
                tracker.observe(initial, phase="pre-activation")
                pose = initial.get("pose")
                startup_origin_pose = dict(pose) if isinstance(pose, Mapping) else None

        if pulse_fault is None:
            pulse_start = clock()
            pulse_deadline = pulse_start + pulse_duration_s
            next_publish = pulse_start
            watchdog = ProcessLocalTimerWatchdog(
                pulse_duration_s,
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
                    max_observed_planar_speed_m_s=max_observed_planar_speed_m_s,
                )
                if current_fault:
                    record_sample("activation", current, current_fault)
                    pulse_fault = current_fault
                    stop_requested.set()
                    break
                if clock() >= pulse_deadline:
                    record_sample("pulse-deadline-observation", current, None)
                    stop_requested.set()
                    break
                command_offset = clock() - pulse_start
                startup_displacement = None
                if startup_speed_m_s is not None:
                    try:
                        if startup_origin_pose is None or not isinstance(current.get("pose"), Mapping):
                            raise ValueError("startup profile requires odometry pose")
                        startup_displacement = forward_projected_displacement(
                            startup_origin_pose, current["pose"]
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        pulse_fault = f"startup_profile_odometry_unavailable:{exc}"
                        stop_requested.set()
                        record_sample("activation", current, None)
                        break
                    if (
                        not startup_transition_observed
                        and startup_displacement >= float(startup_transition_displacement_m)
                    ):
                        startup_transition_observed = True
                        result["velocity_profile"].update({
                            "transition_observed": True,
                            "transition_monotonic_s": clock(),
                            "transition_displacement_observed_m": startup_displacement,
                        })
                    if not startup_transition_observed and command_offset >= startup_transition_timeout_s:
                        row = dict(current)
                        row["command_profile_stage"] = "startup-timeout"
                        row["forward_displacement_since_start_m"] = startup_displacement
                        row["command_profile_fault"] = "startup_displacement_not_reached_before_timeout"
                        record_sample("activation", row, None)
                        pulse_fault = "startup_displacement_not_reached_before_timeout"
                        stop_requested.set()
                        break
                    current = dict(current)
                    current["command_profile_stage"] = "cruise" if startup_transition_observed else "startup"
                    current["forward_displacement_since_start_m"] = startup_displacement
                tracker.observe(current, phase="activation")
                record_sample("activation", current, None)
                if activation_probe_only or source_grounded_route_activation:
                    probe_classification = classify_locomotion_activation_readbacks(
                        current,
                        expected_source=VELOCITY_SOURCE_NAME,
                        maximum_readback_age_s=state_timeout_s,
                        maximum_sync_skew_s=MODE_SOURCE_READBACK_SKEW_MAX_S,
                    )
                    activation_lifecycle.observe_locomotion_readback(
                        probe_classification, now_s=clock(),
                    )
                    activation_record = (
                        result["activation_probe"] if activation_probe_only
                        else result["source_grounded_route_activation"]
                    )
                    activation_record["last_readback_classification"] = probe_classification
                    if probe_classification.get("classification") == "LOCOMOTION_CONFIRMED":
                        if source_grounded_route_activation:
                            if not activation_lifecycle.nonzero_motion_allowed:
                                activation_lifecycle.begin_motion(now_s=clock())
                            if not activation_record["mode_source_confirmed"]:
                                activation_record.update({
                                    "mode_source_confirmed": True,
                                    "confirmation_monotonic_s": clock(),
                                    "terminal_reason": "fresh synchronized LOCO/source handshake observed; continue the same bounded route window",
                                })
                        else:
                            activation_record.update({
                                "mode_source_confirmed": True,
                                "confirmation_monotonic_s": clock(),
                                "terminal_reason": "fresh synchronized LOCO/source handshake observed; aborting probe to STAND",
                            })
                            pulse_fault = "activation_probe_complete_abort_to_stand"
                            stop_requested.set()
                            break
                    if str(probe_classification.get("classification", "")).startswith("ERROR_"):
                        pulse_fault = "activation_probe_readback_error:" + str(
                            probe_classification.get("reason") or probe_classification.get("classification")
                        )
                        activation_record["terminal_reason"] = pulse_fault
                        stop_requested.set()
                        break
                    if (
                        source_grounded_route_activation
                        and activation_lifecycle.activation_probe_command_allowed
                        and command_offset >= source_grounded_activation_timeout_s
                    ):
                        pulse_fault = "source_grounded_route_activation_timeout"
                        activation_record["terminal_reason"] = pulse_fault
                        stop_requested.set()
                        break
                if (
                    not activation_probe_only
                    and not source_grounded_route_activation
                    and require_locomotion_confirmation
                    and not tracker.last_control_handshake_observed
                ):
                    pulse_fault = "fresh_locomotion_mode_source_handshake_not_confirmed_before_nonzero"
                    stop_requested.set()
                    break
                if termination_condition is not None:
                    try:
                        reached = bool(termination_condition(current))
                    except Exception as exc:
                        pulse_fault = f"termination_condition_exception:{type(exc).__name__}:{exc}"
                        stop_requested.set()
                        break
                    if reached:
                        result["termination_condition_met"] = True
                        result["termination_condition_name"] = termination_condition_name
                        stop_requested.set()
                        break
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
                if activation_probe_only and not activation_lifecycle.activation_probe_command_allowed:
                    pulse_fault = "activation_probe_state_no_longer_permits_a_nonzero_probe_command"
                    result["activation_probe"]["terminal_reason"] = pulse_fault
                    stop_requested.set()
                    break
                if (
                    source_grounded_route_activation
                    and not (
                        activation_lifecycle.activation_probe_command_allowed
                        or activation_lifecycle.nonzero_motion_allowed
                    )
                ):
                    pulse_fault = "source_grounded_route_activation_state_no_longer_permits_nonzero"
                    result["source_grounded_route_activation"]["terminal_reason"] = pulse_fault
                    stop_requested.set()
                    break
                if now >= next_publish:
                    should_stop = False
                    with command_lock:
                        # Check again after taking the command lock: the
                        # watchdog may have queued a stop while this thread waited.
                        if stop_requested.is_set() or clock() >= pulse_deadline:
                            should_stop = True
                        else:
                            command_offset = clock() - pulse_start
                            nonzero_attempt_offsets_s.append(command_offset)
                            if (
                                deceleration_duration_s > 0.0
                                and command_offset >= pulse_duration_s - deceleration_duration_s
                            ):
                                command_speed = smoothstep_deceleration_velocity(
                                    forward_command_m_s,
                                    command_offset - (pulse_duration_s - deceleration_duration_s),
                                    deceleration_duration_s,
                                    deceleration_end_speed_m_s,
                                )
                            elif activation_probe_only:
                                command_speed = ACTIVATION_PROBE_FORWARD_M_S
                            elif startup_speed_m_s is not None:
                                if startup_origin_pose is None:
                                    raise RuntimeError("startup origin pose missing")
                                command_speed = startup_then_cruise_velocity(
                                    startup_speed_m_s,
                                    forward_command_m_s,
                                    startup_displacement if startup_displacement is not None else 0.0,
                                    float(startup_transition_displacement_m),
                                )
                            else:
                                command_speed = (
                                    smoothstep_velocity(
                                        float(ramp_start_speed_m_s),
                                        forward_command_m_s,
                                        command_offset,
                                        ramp_duration_s,
                                    )
                                    if ramp_duration_s > 0.0 else forward_command_m_s
                                )
                            publish(command_speed)
                            published_offset = clock() - pulse_start
                            nonzero_success_offsets_s.append(published_offset)
                            if published_offset > pulse_duration_s:
                                pulse_fault = pulse_fault or "nonzero_publish_returned_after_hard_deadline"
                                stop_requested.set()
                    if should_stop:
                        break
                    result["nonzero_publish_calls_returned"] += 1
                    next_publish += 1.0 / PUBLISH_HZ
                    if next_publish < now:
                        next_publish = now + 1.0 / PUBLISH_HZ
                remaining = pulse_deadline - clock()
                if remaining > 0.0:
                    sleep(min(0.005, remaining))
            if activation_probe_only:
                if pulse_fault is None:
                    pulse_fault = "activation_probe_deadline_without_mode_source_confirmation"
                    stop_requested.set()
                result["activation_probe"]["terminal_reason"] = (
                    result["activation_probe"].get("terminal_reason") or pulse_fault
                )
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
        if activation_lifecycle is not None:
            if pulse_fault is not None:
                activation_lifecycle.rearm_forbidden = True
            activation_lifecycle.request_zero_and_stop(
                now_s=clock(),
                reason=pulse_fault or "bounded motion window ended; zero and STAND recovery required",
            )
        recovery_callback = after_zero_stop
        if pulse_fault is not None or (
            require_locomotion_confirmation and not tracker.last_control_handshake_observed
        ):
            recovery_callback = after_zero_abort or after_zero_stop
        # Start and complete the bounded zero-command window before any
        # telemetry callback can stall. The deadline watchdog remains armed
        # until the stop path has been initiated by run_once().
        stop_result = zero_stop.run_once()
        result["zero_stop"] = stop_result
        zero_window_callback_errors = list(stop_result.get("during_zero_callback_errors", []))
        if zero_window_callback_errors:
            post_stop_fault = "zero_stop_telemetry_callback_error:" + zero_window_callback_errors[0]
        result["zero_stop_callback_errors"] = zero_window_callback_errors
        if watchdog is not None:
            watchdog.cancel()
        stop_transition_confirmation: dict[str, Any] = {
            "status": "NOT_REQUESTED" if zero_until_stand_timeout_s == 0.0 else "WAITING_FOR_STAND_READBACK",
            "requested_timeout_s": zero_until_stand_timeout_s,
            "stand_action_requested": bool(
                recovery_callback is not None
                and (
                    bool(set_action.get("request_sent"))
                    or bool(set_action.get("accepted_by_service"))
                )
                and zero_until_stand_timeout_s > 0.0
            ),
            "request_result": None,
            "started_monotonic_s": None,
            "deadline_monotonic_s": None,
            "confirmed_monotonic_s": None,
            "stand_confirmed": False,
        }
        stop_mode_confirmed = False
        post_stop_zero_extension = {
            "status": "ZERO_CONTINUES_THROUGH_POST_STOP_OBSERVATION",
            "requested_timeout_s": zero_until_stand_timeout_s,
            "zero_publish_calls_returned": 0,
            "first_publish_monotonic_s": None,
            "last_publish_monotonic_s": None,
            "max_gap_between_publish_call_returns_s": 0.0,
            "publish_exceptions": [],
            "mc_receipt_confirmed": False,
        }

        def publish_post_stop_zero(now_s: float) -> str | None:
            try:
                with command_lock:
                    publish(0.0)
                returned = clock()
                previous = post_stop_zero_extension["last_publish_monotonic_s"]
                if previous is not None:
                    gap = max(0.0, returned - float(previous))
                    post_stop_zero_extension["max_gap_between_publish_call_returns_s"] = max(
                        float(post_stop_zero_extension["max_gap_between_publish_call_returns_s"]),
                        gap,
                    )
                    if gap >= SOURCE_TIMEOUT_MS / 1000.0:
                        return "post_stop_zero_publish_gap_exceeded_source_timeout"
                if post_stop_zero_extension["first_publish_monotonic_s"] is None:
                    post_stop_zero_extension["first_publish_monotonic_s"] = float(now_s)
                post_stop_zero_extension["last_publish_monotonic_s"] = returned
                post_stop_zero_extension["zero_publish_calls_returned"] += 1
                return None
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                post_stop_zero_extension["publish_exceptions"].append(error)
                post_stop_zero_extension["status"] = "ZERO_PUBLISH_FAILED"
                return "post_stop_zero_publish_exception:" + error

        # Continue zero immediately after the initial window, before a
        # potentially blocking post-walk SetMcAction/readback call.
        initial_post_stop_zero_fault = publish_post_stop_zero(clock())
        if initial_post_stop_zero_fault:
            post_stop_fault = initial_post_stop_zero_fault
        initial_zero_callback_error: str | None = None
        if during_zero_callback is not None:
            try:
                during_zero_callback()
            except Exception as exc:
                initial_zero_callback_error = f"{type(exc).__name__}: {exc}"
                post_stop_fault = post_stop_fault or (
                    "post_stop_initial_zero_callback_error:" + initial_zero_callback_error
                )
        post_stop_zero_extension["initial_zero_callback_error"] = initial_zero_callback_error
        stand_request_was_eligible = bool(stop_transition_confirmation["stand_action_requested"])
        if recovery_callback is not None and stand_request_was_eligible:
            callback_zero_stop = threading.Event()
            callback_zero_metrics: dict[str, Any] = {
                "started": False,
                "publish_calls_returned": 0,
                "publish_faults": [],
                "max_inter_publish_gap_s": 0.0,
                "start_wall_monotonic_s": None,
                "end_wall_monotonic_s": None,
                "thread_joined": True,
            }

            def callback_zero_pump() -> None:
                period = 1.0 / PUBLISH_HZ
                next_send = time.monotonic()
                last_return: float | None = None
                callback_zero_metrics["started"] = True
                callback_zero_metrics["start_wall_monotonic_s"] = next_send
                while not callback_zero_stop.is_set():
                    now_wall = time.monotonic()
                    if now_wall < next_send:
                        callback_zero_stop.wait(next_send - now_wall)
                        continue
                    fault = publish_post_stop_zero(clock())
                    returned_wall = time.monotonic()
                    callback_zero_metrics["publish_calls_returned"] += int(fault is None)
                    if last_return is not None:
                        callback_zero_metrics["max_inter_publish_gap_s"] = max(
                            float(callback_zero_metrics["max_inter_publish_gap_s"]),
                            returned_wall - last_return,
                        )
                    last_return = returned_wall
                    if fault:
                        callback_zero_metrics["publish_faults"].append(fault)
                    next_send = returned_wall + period
                callback_zero_metrics["end_wall_monotonic_s"] = time.monotonic()

            callback_zero_thread = threading.Thread(
                target=callback_zero_pump,
                name="issue80-post-stop-zero-pump",
                daemon=True,
            )
            callback_zero_thread.start()
            try:
                if (
                    initial_post_stop_zero_fault is None
                    and initial_zero_callback_error is None
                    and not zero_window_callback_errors
                ):
                    if activation_lifecycle is not None and not activation_lifecycle.begin_stand_request_once(now_s=clock()):
                        raise RuntimeError("STAND recovery request is not permitted in the current activation lifecycle")
                    stop_transition_confirmation["stand_action_requested"] = True
                    result["after_zero_stop_action"] = dict(recovery_callback())
                    immediate_mode = result["after_zero_stop_action"].get("mode_readback_after_request")
                    if (
                        activation_lifecycle is not None
                        and activation_lifecycle.state == ZERO_AND_STOP_REQUESTED
                        and not activation_lifecycle.locomotion_confirmed_before_stop
                        and isinstance(immediate_mode, Mapping)
                        and immediate_mode.get("action_desc") == MODE_LD
                    ):
                        activation_lifecycle.observe_locomotion_readback(
                            {
                                "classification": "LATE_LOCOMOTION_MODE_OBSERVED",
                                "reason": "GetMcAction immediately after the one STAND recovery request still reports LOCOMOTION",
                                "get_mode_desc": immediate_mode.get("action_desc"),
                                "get_mode_status_value": immediate_mode.get("status_value"),
                            },
                            now_s=clock(),
                        )
                else:
                    result["after_zero_stop_action"] = {
                        "success": False,
                        "error": (
                            initial_post_stop_zero_fault
                            or ("post_stop_initial_zero_callback_error:" + initial_zero_callback_error
                                if initial_zero_callback_error else None)
                            or ("zero_stop_telemetry_callback_error:" + zero_window_callback_errors[0]
                                if zero_window_callback_errors else None)
                        ),
                    }
            except Exception as exc:
                result["after_zero_stop_action"] = {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                callback_zero_stop.set()
                callback_zero_thread.join(timeout=0.5)
                callback_zero_metrics["thread_joined"] = not callback_zero_thread.is_alive()
                if not callback_zero_metrics["thread_joined"]:
                    callback_zero_metrics["publish_faults"].append(
                        "post_stop_zero_pump_thread_join_timeout"
                    )
                if callback_zero_metrics["publish_faults"]:
                    post_stop_fault = post_stop_fault or str(
                        callback_zero_metrics["publish_faults"][0]
                    )
            post_stop_zero_extension["blocking_action_zero_pump"] = callback_zero_metrics
        if zero_until_stand_timeout_s > 0.0:
            stop_transition_confirmation["request_result"] = result.get("after_zero_stop_action")
            stop_transition_confirmation["started_monotonic_s"] = clock()
            stop_transition_confirmation["deadline_monotonic_s"] = (
                float(stop_transition_confirmation["started_monotonic_s"])
                + zero_until_stand_timeout_s
            )
            next_post_stop_zero_publish = float(stop_transition_confirmation["started_monotonic_s"])
        else:
            next_post_stop_zero_publish = clock()
        first_post_stop_sample_attempt_s = clock()
        try:
            first_post_stop = sample()
            immediate_fault = snapshot_fault(
                first_post_stop,
                joint_limits,
                state_timeout_s=state_timeout_s,
                min_root_z=min_root_z,
                max_tilt_deg=max_tilt_deg,
                max_observed_planar_speed_m_s=max_observed_planar_speed_m_s,
                allow_source_readback_gap=True,
                allow_common_uninterruptible_transition_during_zero_observation=True,
            )
            first_source_readback_gap = source_readback_gap(
                first_post_stop, state_timeout_s=state_timeout_s,
            )
            first_sample_time = float(first_post_stop.get("sample_monotonic_s", clock()))
            if first_source_readback_gap:
                post_stop_readback_epoch += 1
                post_stop_readback_gap_active = True
                post_stop_readback_events.append({
                    "start_monotonic_s": first_sample_time,
                    "end_monotonic_s": None,
                    "reason": first_source_readback_gap,
                    "recovered": False,
                    "readback_epoch": post_stop_readback_epoch,
                })
            else:
                post_stop_readback_valid_monotonic_s = first_sample_time
            record_sample("post-stop-first-observation", first_post_stop, immediate_fault)
            first_post_stop_pose = first_post_stop.get("pose")
            post_stop_first_observation = {
                "pose": dict(first_post_stop_pose) if isinstance(first_post_stop_pose, Mapping) else None,
                "sample_attempt_monotonic_s": first_post_stop_sample_attempt_s,
                "sample_monotonic_s": first_post_stop.get("sample_monotonic_s"),
                "safety_fault": immediate_fault,
                "source_readback_gap": first_source_readback_gap,
                "physical_state_observation_valid": immediate_fault is None,
                "timing_relation": "sampled_after_zero_stop_window",
                "zero_window_complete_before_sample": (
                    stop_result["end_monotonic_s"] is not None
                    and stop_result["end_monotonic_s"] <= first_post_stop_sample_attempt_s
                ),
            }
            if isinstance(first_post_stop_pose, Mapping):
                post_stop_first_pose = dict(first_post_stop_pose)
                post_stop_last_pose = dict(first_post_stop_pose)
            if zero_until_stand_timeout_s > 0.0 and stand_mode_readback_confirmed(first_post_stop):
                stop_mode_confirmed = True
                stop_transition_confirmation["status"] = (
                    "STAND_READBACK_CONFIRMED"
                    if stop_transition_confirmation["stand_action_requested"]
                    else "AUTO_STAND_READBACK_CONFIRMED"
                )
                stop_transition_confirmation["stand_confirmed"] = True
                stop_transition_confirmation["confirmed_monotonic_s"] = float(
                    first_post_stop.get("sample_monotonic_s", clock())
                )
                post_stop_zero_extension["status"] = "STAND_READBACK_CONFIRMED"
            if immediate_fault:
                post_stop_fault = immediate_fault
        except Exception as exc:
            post_stop_fault = f"post_stop_first_observation_exception:{type(exc).__name__}:{exc}"
            post_stop_first_observation = {
                "pose": None,
                "sample_attempt_monotonic_s": first_post_stop_sample_attempt_s,
                "safety_fault": post_stop_fault,
                "timing_relation": "sampled_after_zero_stop_window",
                "zero_window_complete_before_sample": (
                    stop_result["end_monotonic_s"] is not None
                    and stop_result["end_monotonic_s"] <= first_post_stop_sample_attempt_s
                ),
            }
        result["post_stop_first_observation"] = post_stop_first_observation
        # Completing this qualification intentionally sets pulse_fault to enter
        # the abort path. It must not suppress the already authorized bounded
        # observation tail: STAND may arrive late, and the strict physical-stop
        # evaluator needs a complete fresh hold after that readback.
        intentional_activation_probe_abort = (
            activation_probe_only
            and result["activation_probe"].get("mode_source_confirmed") is True
            and pulse_fault == "activation_probe_complete_abort_to_stand"
        )
        post_stop_extra_applied_s = (
            extra_post_stop_observe_s
            if pulse_fault is None or intentional_activation_probe_abort
            else 0.0
        )
        result["post_stop_extra_observation_applied_s"] = post_stop_extra_applied_s
        post_stop_end = clock() + POST_STOP_OBSERVE_S + post_stop_extra_applied_s
        post_stop_samples = 0
        while clock() < post_stop_end:
            now = clock()
            stop_extension_publish_fault: str | None = None
            if zero_until_stand_timeout_s > 0.0 and not stop_mode_confirmed:
                deadline = float(stop_transition_confirmation["deadline_monotonic_s"])
                if now >= deadline and stop_transition_confirmation["status"] == "WAITING_FOR_STAND_READBACK":
                    transition_timeout = (
                        "STAND_READBACK_TIMEOUT"
                        if stop_transition_confirmation["stand_action_requested"]
                        else "AUTO_STAND_READBACK_TIMEOUT"
                    )
                    stop_transition_confirmation["status"] = transition_timeout
                    post_stop_zero_extension["status"] = transition_timeout + "_ZERO_CONTINUES"
                    timeout_fault = (
                        "post_stop_stand_readback_timeout"
                        if stop_transition_confirmation["stand_action_requested"]
                        else "post_stop_auto_stand_transition_timeout"
                    )
                    post_stop_fault = post_stop_fault or timeout_fault
            if now >= next_post_stop_zero_publish:
                stop_extension_publish_fault = publish_post_stop_zero(now)
                if stop_extension_publish_fault:
                    post_stop_fault = post_stop_fault or stop_extension_publish_fault
                next_post_stop_zero_publish = clock() + 1.0 / PUBLISH_HZ
            try:
                current = sample()
                current_fault = snapshot_fault(
                    current,
                    joint_limits,
                    state_timeout_s=state_timeout_s,
                    min_root_z=min_root_z,
                    max_tilt_deg=max_tilt_deg,
                    max_observed_planar_speed_m_s=max_observed_planar_speed_m_s,
                    allow_source_readback_gap=True,
                    allow_common_uninterruptible_transition_during_zero_observation=True,
                )
                if stop_extension_publish_fault and not current_fault:
                    current_fault = stop_extension_publish_fault
                if current_fault:
                    post_stop_fault = current_fault
                if zero_until_stand_timeout_s > 0.0 and not stop_mode_confirmed:
                    if stand_mode_readback_confirmed(current):
                        stop_mode_confirmed = True
                        prior_status = str(stop_transition_confirmation["status"])
                        stop_transition_confirmation["status"] = (
                            "LATE_STAND_READBACK_CONFIRMED"
                            if prior_status == "STAND_READBACK_TIMEOUT"
                            else "LATE_AUTO_STAND_READBACK_CONFIRMED"
                            if prior_status == "AUTO_STAND_READBACK_TIMEOUT"
                            else "STAND_READBACK_CONFIRMED"
                            if stop_transition_confirmation["stand_action_requested"]
                            else "AUTO_STAND_READBACK_CONFIRMED"
                        )
                        stop_transition_confirmation["stand_confirmed"] = True
                        stop_transition_confirmation["confirmed_monotonic_s"] = float(
                            current.get("sample_monotonic_s", clock())
                        )
                        post_stop_zero_extension["status"] = stop_transition_confirmation["status"]
                current_sample_time = float(current.get("sample_monotonic_s", clock()))
                current_source_readback_gap = source_readback_gap(
                    current, state_timeout_s=state_timeout_s,
                )
                if current_source_readback_gap:
                    if not post_stop_readback_gap_active:
                        post_stop_readback_epoch += 1
                        post_stop_readback_events.append({
                            "start_monotonic_s": current_sample_time,
                            "end_monotonic_s": None,
                            "reason": current_source_readback_gap,
                            "recovered": False,
                            "readback_epoch": post_stop_readback_epoch,
                        })
                    post_stop_readback_gap_active = True
                    post_stop_readback_valid_monotonic_s = None
                else:
                    if post_stop_readback_gap_active:
                        event = post_stop_readback_events[-1]
                        event["end_monotonic_s"] = current_sample_time
                        event["duration_s"] = max(
                            0.0, current_sample_time - float(event["start_monotonic_s"])
                        )
                        event["recovered"] = True
                        post_stop_readback_gap_active = False
                    post_stop_readback_valid_monotonic_s = current_sample_time
                stand_readback_is_current = stand_mode_readback_confirmed(current)
                acceptance_gap = (
                    current_source_readback_gap
                    or (None if stand_readback_is_current else "fresh STAND readback not present")
                    or current_fault
                )
                if acceptance_gap:
                    current["post_stop_gate_eligible"] = False
                    current["post_stop_acceptance_gap"] = acceptance_gap
                    if not post_stop_acceptance_gap_active:
                        if not current_source_readback_gap:
                            post_stop_readback_epoch += 1
                        post_stop_acceptance_gap_events.append({
                            "start_monotonic_s": current_sample_time,
                            "end_monotonic_s": None,
                            "reason": acceptance_gap,
                            "readback_epoch": post_stop_readback_epoch,
                        })
                        post_stop_acceptance_gap_active = True
                else:
                    current["post_stop_gate_eligible"] = current_fault is None
                    current["post_stop_acceptance_gap"] = None
                    if post_stop_acceptance_gap_active and post_stop_acceptance_gap_events:
                        event = post_stop_acceptance_gap_events[-1]
                        if event["end_monotonic_s"] is None:
                            event["end_monotonic_s"] = current_sample_time
                            event["duration_s"] = max(
                                0.0, current_sample_time - float(event["start_monotonic_s"])
                            )
                    post_stop_acceptance_gap_active = False
                current["post_stop_source_readback_gap"] = current_source_readback_gap
                current["post_stop_readback_epoch"] = post_stop_readback_epoch
                current["post_stop_readback_valid_monotonic_s"] = post_stop_readback_valid_monotonic_s
                record_sample("post-stop", current, current_fault)
                post_stop_samples += 1
                pose = current.get("pose")
                if isinstance(pose, Mapping):
                    if post_stop_first_pose is None:
                        post_stop_first_pose = dict(pose)
                    post_stop_last_pose = dict(pose)
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
                    during_zero_callback=during_zero_callback,
                )
                result["post_stop_emergency_zero"] = emergency_stop.run_once()
                break
            remaining = post_stop_end - clock()
            if remaining > 0.0:
                sleep(min(1.0 / PUBLISH_HZ, remaining))

        if post_stop_readback_gap_active:
            post_stop_fault = post_stop_fault or "post_stop_required_readback_not_recovered"
            if post_stop_readback_events and post_stop_readback_events[-1]["end_monotonic_s"] is None:
                post_stop_readback_events[-1]["observation_deadline_monotonic_s"] = post_stop_end
        if not post_stop_zero_extension["publish_exceptions"] and post_stop_fault is None:
            post_stop_zero_extension["status"] = (
                "STAND_READBACK_CONFIRMED_ZERO_CONTINUES_THROUGH_POST_STOP_OBSERVATION"
                if stop_mode_confirmed
                else "ZERO_CONTINUED_THROUGH_POST_STOP_OBSERVATION_STAND_UNCONFIRMED"
            )
        post_stop_zero_extension["end_monotonic_s"] = clock()

    result["pulse_actual_duration_s"] = (
        max(0.0, pulse_end - pulse_start) if pulse_start is not None and pulse_end is not None else 0.0
    )
    # Loop elapsed time includes sensor sampling after the hard command deadline;
    # publish offsets below measure when nonzero sends actually began/returned.
    result["pulse_loop_elapsed_s"] = result.pop("pulse_actual_duration_s")
    if zero_until_stand_timeout_s > 0.0:
        result["stop_transition_confirmation"] = stop_transition_confirmation
        result["post_stop_zero_extension"] = post_stop_zero_extension
    result["post_stop_readback_gap_events"] = post_stop_readback_events
    result["post_stop_acceptance_gap_events"] = post_stop_acceptance_gap_events
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
    result["source_acquired"] = tracker.source_acquired
    result["ld_observed"] = tracker.ld_observed
    result["control_handshake_confirmed"] = tracker.control_handshake_confirmed
    result["control_handshake_status"] = tracker.classify_control_handshake()
    result["control_handshake_evidence"] = [tracker.samples[index] for index in tracker.handshake_sample_indices]
    result["pulse_odometry"] = tracker.odometry_delta()
    if activation_probe_only:
        result["activation_probe"].update({
            "actual_loop_elapsed_s": result["pulse_loop_elapsed_s"],
            "nonzero_publish_calls_returned": result["nonzero_publish_calls_returned"],
            "command_success_offsets_s": list(nonzero_success_offsets_s),
            "actual_forward_projected_displacement_m": (
                result.get("pulse_odometry", {}).get("forward_projected_m")
                if isinstance(result.get("pulse_odometry"), Mapping) else None
            ),
        })
    if source_grounded_route_activation:
        result["source_grounded_route_activation"].update({
            "actual_loop_elapsed_s": result["pulse_loop_elapsed_s"],
            "nonzero_publish_calls_returned": result["nonzero_publish_calls_returned"],
            "command_success_offsets_s": list(nonzero_success_offsets_s),
            "actual_forward_projected_displacement_m": (
                result.get("pulse_odometry", {}).get("forward_projected_m")
                if isinstance(result.get("pulse_odometry"), Mapping) else None
            ),
        })
    pulse_forward = (
        result["pulse_odometry"].get("forward_projected_m") if result["pulse_odometry"] else None
    )
    result["physical_motion_status"] = classify_physical_motion(
        control_handshake_confirmed=tracker.control_handshake_confirmed,
        pulse_forward_displacement_m=pulse_forward,
        baseline_noise_bound_m=baseline_noise_bound_m,
    )
    post_stop_forward = None
    last_in_pulse_sample = next(
        (
            row for row in reversed(result["samples"])
            if row.get("phase") in ("activation", "pulse-deadline-observation")
        ),
        None,
    )
    last_in_pulse_pose = (
        last_in_pulse_sample.get("pose")
        if isinstance(last_in_pulse_sample, Mapping)
        else tracker.last_activation_pose
    )
    if isinstance(last_in_pulse_pose, Mapping) and post_stop_last_pose is not None:
        try:
            post_stop_forward = forward_projected_displacement(last_in_pulse_pose, post_stop_last_pose)
        except ValueError:
            post_stop_forward = None
    result["post_stop_odometry"] = {
        "anchor": "last_in_pulse_pose",
        "last_in_pulse_pose": dict(last_in_pulse_pose) if isinstance(last_in_pulse_pose, Mapping) else None,
        "last_in_pulse_sample_monotonic_s": (
            last_in_pulse_sample.get("sample_monotonic_s")
            if isinstance(last_in_pulse_sample, Mapping)
            else tracker.last_activation_monotonic_s
        ),
        "first_pose": post_stop_first_pose,
        "last_pose": post_stop_last_pose,
        "forward_displacement_from_pulse_endpoint_m": post_stop_forward,
        "baseline_noise_bound_m": baseline_noise_bound_m,
        "kinematic_status": (
            "POST_STOP_ODOMETRY_UNOBSERVED" if post_stop_forward is None
            else "POST_STOP_NOISE_UNRESOLVED" if baseline_noise_bound_m is None or baseline_noise_bound_m <= 0.0
            else "WITHIN_BASELINE_ENVELOPE" if baseline_noise_bound_m is not None and abs(post_stop_forward) <= baseline_noise_bound_m
            else "ABOVE_BASELINE_ENVELOPE"
        ),
        "mc_stop_confirmed": False,
    }
    result["post_stop_forward_tail"] = measure_post_stop_forward_tail(
        result["samples"], baseline_noise_bound_m
    )
    result["legacy_baseline_stop_gate"] = {
        "status": result["post_stop_forward_tail"].get("status", "NOT_RUN"),
        "criterion": "unchanged empirical same-run maximum absolute forward displacement over three one-second stationary windows",
        "tail_window": dict(result["post_stop_forward_tail"]),
        "interpretation": "A separate legacy result; it is not relaxed, replaced, or silently reclassified by PHYSICAL_STOP_GATE.",
    }
    result["physical_stop_gate"] = evaluate_physical_stop_gate(
        result["samples"],
        zero_start_monotonic_s=float(stop_result.get("start_monotonic_s", pulse_end or clock())),
        stand_confirmed_monotonic_s=(
            stop_transition_confirmation.get("confirmed_monotonic_s")
            if zero_until_stand_timeout_s > 0.0 else None
        ),
        baseline_state_summary=baseline_state_summary,
    )
    if activation_lifecycle is not None:
        activation_lifecycle.confirm_stand_after_stop(
            stand_readback_pass=bool(stop_transition_confirmation.get("stand_confirmed")),
            physical_stop_gate_pass=result["physical_stop_gate"].get("status") == "PASS",
            now_s=clock(),
        )
        result["activation_lifecycle"] = activation_lifecycle.snapshot()
    result["mc_receipt_confirmed_for_zero_calls"] = False
    result["watchdog"] = {
        "implementation": "threading.Timer in the first-motion Python process",
        "process_local": True,
        "covers_stalled_ros_spin_while_process_alive": True,
        "covers_process_crash_or_kill": False,
    }
    result["fault"] = pulse_fault or post_stop_fault
    result["post_stop_observation_s"] = POST_STOP_OBSERVE_S
    result["extra_post_stop_observation_s"] = extra_post_stop_observe_s
    result["extra_post_stop_observation_limit_s"] = extra_post_stop_observe_limit_s
    result["zero_stop_min_messages"] = ZERO_STOP_MIN_MESSAGES
    result["post_stop_sample_count"] = post_stop_samples
    result["post_stop_fault"] = post_stop_fault
    if pulse_fault:
        result["status"] = "ABORTED"
    elif stop_result["publish_exceptions"]:
        result["status"] = "ZERO_STOP_PUBLISH_EXCEPTION"
    elif stop_result["loop_duration_s"] < SOURCE_TIMEOUT_MS / 1000.0:
        result["status"] = "ZERO_STOP_CALL_WINDOW_TOO_SHORT"
    elif stop_result["publish_call_return_span_s"] < SOURCE_TIMEOUT_MS / 1000.0:
        result["status"] = "ZERO_STOP_CALL_RETURN_SPAN_TOO_SHORT"
    elif stop_result["zero_publish_calls_returned"] < ZERO_STOP_MIN_MESSAGES:
        result["status"] = "ZERO_STOP_CALL_RETURN_COUNT_TOO_LOW"
    elif (stop_result["max_gap_between_publish_call_returns_s"] or 0.0) >= SOURCE_TIMEOUT_MS / 1000.0:
        result["status"] = "ZERO_STOP_CALL_RETURN_GAP_EXCEEDED_SOURCE_TIMEOUT"
    elif post_stop_fault:
        result["status"] = "POST_STOP_SAFETY_FAILED"
    elif result.get("after_zero_stop_action") is not None and result["after_zero_stop_action"].get("success") is not True:
        result["status"] = "POST_STOP_STAND_REQUEST_FAILED"
    elif post_stop_samples == 0:
        result["status"] = "POST_STOP_OBSERVATION_MISSING"
    elif result["nonzero_publish_calls_returned"] == 0:
        result["status"] = "NO_ACTIVATION_COMMAND"
    else:
        post_stop_status = result["post_stop_forward_tail"]["status"]
        if result["physical_motion_status"] == "PHYSICAL_MOTION_CONFIRMED" and post_stop_status == "WITHIN_BASELINE_ENVELOPE":
            result["status"] = "PHYSICAL_MOTION_AND_POSTSTOP_STATIONARY_OBSERVED"
        elif result["physical_motion_status"] == "PHYSICAL_MOTION_CONFIRMED":
            result["status"] = "PHYSICAL_MOTION_CONFIRMED_POSTSTOP_UNVERIFIED"
        else:
            result["status"] = result["physical_motion_status"]
    return result
