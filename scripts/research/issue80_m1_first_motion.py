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
MODE_SOURCE_READBACK_SKEW_MAX_S = 0.2  # the configured 5 Hz readback polling interval
BASELINE_WINDOW_S = 1.0
BASELINE_WINDOW_COUNT = 3
BASELINE_DURATION_S = BASELINE_WINDOW_S * BASELINE_WINDOW_COUNT
BASELINE_SAMPLE_PERIOD_S = 1.0 / PUBLISH_HZ

MODE_STAND = "STAND_DEFAULT"
MODE_LOCOMOTION = "LOCOMOTION_DEFAULT"
ALLOWED_MODES = frozenset((MODE_STAND, MODE_LOCOMOTION))
MC_ACTION_STATUSES = frozenset((0, 100, 200))  # IDLE, RUNNING, TRANSITION


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
        mode_get = row["get_mode_header_code"] == 0 and row["get_mode_desc"] == MODE_LOCOMOTION
        mode_common = row["common_mode_desc"] == MODE_LOCOMOTION
        source_pair = source_get and source_common
        mode_pair = mode_get and mode_common
        self.get_source_seen |= source_get
        self.common_source_seen |= source_common
        self.get_locomotion_seen |= mode_get
        self.common_locomotion_seen |= mode_common
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
        row["mode_source_readbacks_contemporaneous"] = (
            readback_skew is not None
            and readback_skew <= MODE_SOURCE_READBACK_SKEW_MAX_S
            and common_skew is not None
            and common_skew <= MODE_SOURCE_READBACK_SKEW_MAX_S
        )
        row["source_pair_observed"] = source_pair
        row["ld_pair_observed"] = mode_pair
        row["control_handshake_observed"] = (
            source_pair and mode_pair and row["mode_source_readbacks_contemporaneous"]
        )
        if row["control_handshake_observed"]:
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
            "start_monotonic_s": None,
            "first_publish_attempt_monotonic_s": None,
            "end_monotonic_s": None,
            "loop_duration_s": 0.0,
            "zero_publish_calls_started": 0,
            "zero_publish_calls_returned": 0,
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
                remaining = hard_deadline - self.clock()
                if remaining > 0.0:
                    self.sleep(min(self.period_s, remaining))
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
    forward_command_m_s: float = FORWARD_COMMAND_M_S,
    pulse_duration_s: float = PULSE_LIMIT_S,
    max_pulse_duration_s: float = PULSE_LIMIT_S,
    ramp_start_speed_m_s: float | None = None,
    ramp_duration_s: float = 0.0,
    startup_speed_m_s: float | None = None,
    startup_transition_displacement_m: float | None = None,
    startup_transition_timeout_s: float = 0.0,
    extra_post_stop_observe_s: float = 0.0,
    after_zero_stop: Callable[[], Mapping[str, Any]] | None = None,
    shutdown_event: threading.Event | None = None,
    on_sample: Callable[[str, Mapping[str, Any]], None] | None = None,
    state_timeout_s: float = 0.5,
    min_root_z: float = 0.52,
    max_tilt_deg: float = 15.0,
    max_observed_planar_speed_m_s: float | None = None,
    zero_until_stand_timeout_s: float = 0.0,
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
                else "smoothstep" if ramp_duration_s > 0.0 else "constant"
            ),
            "ramp_start_speed_m_s": ramp_start_speed_m_s,
            "target_speed_m_s": forward_command_m_s,
            "ramp_duration_s": ramp_duration_s,
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
        "registration": dict(registration),
        "set_mc_action": dict(set_action),
        "nonzero_publish_calls_returned": 0,
        "samples": [],
        "fault": None,
        "after_zero_stop_action": None,
    }
    if not authorized:
        result["status"] = "NOT_AUTHORIZED"
        return result

    pulse_fault: str | None = None
    if not math.isfinite(extra_post_stop_observe_s) or not 0.0 <= extra_post_stop_observe_s <= 5.0:
        pulse_fault = "extra_stop_observation_outside_bounded_budget"
    elif (
        not math.isfinite(zero_until_stand_timeout_s)
        or not 0.0 <= zero_until_stand_timeout_s <= 2.5
    ):
        pulse_fault = "zero_until_stand_timeout_outside_bounded_budget"
    elif zero_until_stand_timeout_s > 0.0 and after_zero_stop is None:
        pulse_fault = "zero_until_stand_requires_post_stop_action_client"
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
                            command_offset = clock() - pulse_start
                            nonzero_attempt_offsets_s.append(command_offset)
                            if startup_speed_m_s is not None:
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
        # Start and complete the bounded zero-command window before any
        # telemetry callback can stall. The deadline watchdog remains armed
        # until the stop path has been initiated by run_once().
        stop_result = zero_stop.run_once()
        result["zero_stop"] = stop_result
        if watchdog is not None:
            watchdog.cancel()
        stop_transition_confirmation: dict[str, Any] = {
            "status": "NOT_REQUESTED" if zero_until_stand_timeout_s == 0.0 else "WAITING_FOR_STAND_READBACK",
            "requested_timeout_s": zero_until_stand_timeout_s,
            "request_result": None,
            "started_monotonic_s": None,
            "deadline_monotonic_s": None,
            "confirmed_monotonic_s": None,
            "stand_confirmed": False,
        }
        stop_mode_confirmed = False
        post_stop_zero_extension = {
            "status": "NOT_REQUESTED" if zero_until_stand_timeout_s == 0.0 else "WAITING_FOR_STAND_READBACK",
            "requested_timeout_s": zero_until_stand_timeout_s,
            "zero_publish_calls_returned": 0,
            "first_publish_monotonic_s": None,
            "last_publish_monotonic_s": None,
            "publish_exceptions": [],
            "mc_receipt_confirmed": False,
        }
        if after_zero_stop is not None and tracker.control_handshake_confirmed and result["nonzero_publish_calls_returned"] > 0:
            try:
                result["after_zero_stop_action"] = dict(after_zero_stop())
            except Exception as exc:
                result["after_zero_stop_action"] = {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if zero_until_stand_timeout_s > 0.0:
            stop_transition_confirmation["request_result"] = result.get("after_zero_stop_action")
            stop_transition_confirmation["started_monotonic_s"] = clock()
            stop_transition_confirmation["deadline_monotonic_s"] = (
                float(stop_transition_confirmation["started_monotonic_s"])
                + zero_until_stand_timeout_s
            )
            next_post_stop_zero_publish = float(stop_transition_confirmation["started_monotonic_s"])
        else:
            next_post_stop_zero_publish = None
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
            )
            record_sample("post-stop-first-observation", first_post_stop, immediate_fault)
            first_post_stop_pose = first_post_stop.get("pose")
            post_stop_first_observation = {
                "pose": dict(first_post_stop_pose) if isinstance(first_post_stop_pose, Mapping) else None,
                "sample_attempt_monotonic_s": first_post_stop_sample_attempt_s,
                "sample_monotonic_s": first_post_stop.get("sample_monotonic_s"),
                "safety_fault": immediate_fault,
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
                stop_transition_confirmation["status"] = "STAND_READBACK_CONFIRMED"
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
        post_stop_end = clock() + POST_STOP_OBSERVE_S + (extra_post_stop_observe_s if pulse_fault is None else 0.0)
        post_stop_samples = 0
        while clock() < post_stop_end:
            now = clock()
            stop_extension_publish_fault: str | None = None
            if zero_until_stand_timeout_s > 0.0 and not stop_mode_confirmed:
                deadline = float(stop_transition_confirmation["deadline_monotonic_s"])
                if now >= deadline and stop_transition_confirmation["status"] == "WAITING_FOR_STAND_READBACK":
                    stop_transition_confirmation["status"] = "STAND_READBACK_TIMEOUT"
                    post_stop_zero_extension["status"] = "STAND_READBACK_TIMEOUT_ZERO_CONTINUES"
                    post_stop_fault = post_stop_fault or "post_stop_stand_readback_timeout"
                if next_post_stop_zero_publish is not None and now >= next_post_stop_zero_publish:
                    try:
                        publish(0.0)
                        if post_stop_zero_extension["first_publish_monotonic_s"] is None:
                            post_stop_zero_extension["first_publish_monotonic_s"] = now
                        post_stop_zero_extension["last_publish_monotonic_s"] = now
                        post_stop_zero_extension["zero_publish_calls_returned"] += 1
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        post_stop_zero_extension["publish_exceptions"].append(error)
                        post_stop_zero_extension["status"] = "ZERO_PUBLISH_FAILED"
                        stop_extension_publish_fault = "post_stop_zero_publish_exception:" + error
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
                )
                if stop_extension_publish_fault and not current_fault:
                    current_fault = stop_extension_publish_fault
                if current_fault:
                    post_stop_fault = current_fault
                record_sample("post-stop", current, current_fault)
                post_stop_samples += 1
                pose = current.get("pose")
                if isinstance(pose, Mapping):
                    if post_stop_first_pose is None:
                        post_stop_first_pose = dict(pose)
                    post_stop_last_pose = dict(pose)
                if zero_until_stand_timeout_s > 0.0 and not stop_mode_confirmed:
                    if stand_mode_readback_confirmed(current):
                        stop_mode_confirmed = True
                        prior_status = str(stop_transition_confirmation["status"])
                        stop_transition_confirmation["status"] = (
                            "LATE_STAND_READBACK_CONFIRMED"
                            if prior_status == "STAND_READBACK_TIMEOUT"
                            else "STAND_READBACK_CONFIRMED"
                        )
                        stop_transition_confirmation["stand_confirmed"] = True
                        stop_transition_confirmation["confirmed_monotonic_s"] = float(
                            current.get("sample_monotonic_s", clock())
                        )
                        post_stop_zero_extension["status"] = stop_transition_confirmation["status"]
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
                pending_stand_confirmation = (
                    zero_until_stand_timeout_s > 0.0 and not stop_mode_confirmed
                )
                sleep(min(1.0 / PUBLISH_HZ if pending_stand_confirmation else 0.05, remaining))

    result["pulse_actual_duration_s"] = (
        max(0.0, pulse_end - pulse_start) if pulse_start is not None and pulse_end is not None else 0.0
    )
    # Loop elapsed time includes sensor sampling after the hard command deadline;
    # publish offsets below measure when nonzero sends actually began/returned.
    result["pulse_loop_elapsed_s"] = result.pop("pulse_actual_duration_s")
    if zero_until_stand_timeout_s > 0.0:
        result["stop_transition_confirmation"] = stop_transition_confirmation
        result["post_stop_zero_extension"] = post_stop_zero_extension
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
