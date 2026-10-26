#!/usr/bin/env python3
"""Pure, fail-closed X2 startup lifecycle supervisor.

This research tool plans allowed transitions from externally observed evidence.
It does not import MuJoCo/rclpy, write qpos, call a service, or publish a
command. A separately reviewed executor must perform every returned operation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from typing import Optional


MIN_ROOT_Z_M = 0.52
MAX_ABS_TILT_RAD = math.radians(15.0)
MAX_PLANAR_SPEED_M_S = 0.20
STANDING_HOLD_S = 2.5
STANDING_MIN_UNIQUE_SAMPLES = 25
SET_MC_ACTION_SERVICE = "/aimdk_5Fmsgs/srv/SetMcAction"
GET_MC_ACTION_SERVICE = "/aimdk_5Fmsgs/srv/GetMcAction"
SET_MC_INPUT_SOURCE_SERVICE = "/aimdk_5Fmsgs/srv/SetMcInputSource"
LOCOMOTION_VELOCITY_TOPIC = "/aima/mc/locomotion/velocity"


class StartupPath(str, Enum):
    SIMULATION = "simulation"
    REAL_ROBOT = "real_robot"
    NAVIGATION = "navigation"


class SupervisorState(str, Enum):
    BLOCKED = "blocked"
    INITIAL_CONFIGURATION_REQUIRED = "initial_configuration_required"
    PHYSICAL_RECOVERY_REQUIRED = "physical_recovery_required"
    OBSERVE_NATIVE_STARTUP = "observe_native_startup"
    JD_REQUEST_ALLOWED = "jd_request_allowed"
    JD_TRANSITION = "jd_transition"
    JD_READY = "jd_ready"
    SD_REQUEST_ALLOWED = "sd_request_allowed"
    SD_TRANSITION = "sd_transition"
    STANDING_HOLD = "standing_hold"
    STABLE_IDLE = "stable_idle"
    NAVIGATION_SOURCE_REQUIRED = "navigation_source_required"
    NAVIGATION_LD_PENDING = "navigation_ld_pending"
    NAVIGATION_ACTIVE = "navigation_active"
    NAVIGATION_STOP_VERIFY = "navigation_stop_verify"


class Operation(str, Enum):
    OBSERVE_ONLY = "observe_only"
    INITIALIZE_VERIFIED_SIMULATION_STANDING_CONFIGURATION = (
        "initialize_verified_simulation_standing_configuration"
    )
    REQUEST_OPERATOR_OR_SUPPORTED_PHYSICAL_RECOVERY = (
        "request_operator_or_supported_physical_recovery"
    )
    REQUEST_JD = "request_joint_default"
    REQUEST_SD = "request_stand_default"
    REGISTER_NAVIGATION_SOURCE = "register_navigation_source"
    REQUEST_LD = "request_locomotion_default"
    PUBLISH_BOUNDED_NAVIGATION_COMMAND = "publish_bounded_navigation_command"
    PUBLISH_ZERO_COMMAND = "publish_zero_navigation_command"
    VERIFY_PHYSICAL_STOP = "verify_physical_stop"
    RELEASE_NAVIGATION_SOURCE = "release_navigation_source"


SUPPORTED_AIMDK_INTERFACES = {
    Operation.REQUEST_JD: {
        "kind": "service",
        "name": SET_MC_ACTION_SERVICE,
        "request": {"command.action_desc": "JOINT_DEFAULT"},
    },
    Operation.REQUEST_SD: {
        "kind": "service",
        "name": SET_MC_ACTION_SERVICE,
        "request": {"command.action_desc": "STAND_DEFAULT"},
    },
    Operation.REQUEST_LD: {
        "kind": "service",
        "name": SET_MC_ACTION_SERVICE,
        "request": {"command.action_desc": "LOCOMOTION_DEFAULT"},
    },
    Operation.REGISTER_NAVIGATION_SOURCE: {
        "kind": "service",
        "name": SET_MC_INPUT_SOURCE_SERVICE,
        "request": {"action": "ADD_OR_ENABLE_VERIFIED_SOURCE"},
    },
    Operation.PUBLISH_BOUNDED_NAVIGATION_COMMAND: {
        "kind": "topic",
        "name": LOCOMOTION_VELOCITY_TOPIC,
        "request": {"source": "REGISTERED_SOURCE", "requires": "valid_nonzero_command"},
    },
    Operation.PUBLISH_ZERO_COMMAND: {
        "kind": "topic",
        "name": LOCOMOTION_VELOCITY_TOPIC,
        "request": {"source": "SELECTED_SOURCE", "requires": "zero_command_then_measure_stop"},
    },
    Operation.RELEASE_NAVIGATION_SOURCE: {
        "kind": "service",
        "name": SET_MC_INPUT_SOURCE_SERVICE,
        "request": {"action": "DISABLE_OR_DELETE_AFTER_PHYSICAL_STOP"},
    },
}


@dataclass(frozen=True)
class StartupObservation:
    """Input owned by the backend adapter or a real-robot operator workflow."""

    telemetry_fresh: bool
    source_stamps_strictly_advancing: bool
    root_z_m: Optional[float] = None
    roll_rad: Optional[float] = None
    pitch_rad: Optional[float] = None
    planar_speed_m_s: Optional[float] = None
    joint_state_ready: bool = False
    source_joint_limits_ok: bool = False
    controller_gains_active: bool = False
    feet_supported: Optional[bool] = None
    support_evidence: str = "unknown"
    common_action: Optional[str] = None
    common_action_status: Optional[int] = None
    get_action: Optional[str] = None
    get_action_status: Optional[int] = None
    fsm_state: Optional[int] = None
    body_status: Optional[int] = None
    reset_effect_verified: bool = False
    simulation_initial_configuration_verified: bool = False
    simulation_initial_support_verified: bool = False
    simulation_generation_verified: bool = False
    supported_physical_standup_completed: bool = False
    operator_recovery_confirmed: bool = False
    standing_hold_s: float = 0.0
    standing_unique_samples: int = 0
    navigation_source_registered: bool = False
    navigation_source_selected: bool = False
    valid_nonzero_navigation_command_seen: bool = False
    locomotion_mode_confirmed: bool = False
    physical_movement_confirmed: bool = False
    zero_command_published: bool = False
    physical_stop_confirmed: bool = False
    navigation_source_released: bool = False


@dataclass(frozen=True)
class SupervisorDecision:
    path: StartupPath
    state: SupervisorState
    allowed_operations: tuple[Operation, ...]
    reason: str
    safety_failure: Optional[str] = None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["path"] = self.path.value
        payload["state"] = self.state.value
        payload["allowed_operations"] = [item.value for item in self.allowed_operations]
        payload["interface_contracts"] = [
            SUPPORTED_AIMDK_INTERFACES[item]
            for item in self.allowed_operations
            if item in SUPPORTED_AIMDK_INTERFACES
        ]
        return payload


def _finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(float(value))


def posture_failure(observation: StartupObservation) -> Optional[str]:
    required = (
        observation.root_z_m,
        observation.roll_rad,
        observation.pitch_rad,
        observation.planar_speed_m_s,
    )
    if not all(_finite(value) for value in required):
        return "missing_or_nonfinite_posture_or_twist"
    if float(observation.root_z_m) < MIN_ROOT_Z_M:
        return "root_height_below_minimum"
    if max(abs(float(observation.roll_rad)), abs(float(observation.pitch_rad))) >= MAX_ABS_TILT_RAD:
        return "roll_or_pitch_limit"
    if float(observation.planar_speed_m_s) > MAX_PLANAR_SPEED_M_S:
        return "planar_speed_limit"
    return None


def evidence_failure(observation: StartupObservation) -> Optional[str]:
    if not observation.telemetry_fresh:
        return "stale_or_missing_telemetry"
    if not observation.source_stamps_strictly_advancing:
        return "nonadvancing_source_timestamps"
    if not observation.joint_state_ready:
        return "joint_state_not_ready"
    if not observation.source_joint_limits_ok:
        return "source_joint_limits_not_verified"
    return posture_failure(observation)


def jd_ready(observation: StartupObservation) -> bool:
    return (
        observation.common_action == "JOINT_DEFAULT"
        and observation.common_action_status == 100
        and observation.get_action == "JOINT_DEFAULT"
        and observation.get_action_status == 100
        and observation.fsm_state == 1
        and observation.body_status == 1
        and observation.controller_gains_active
    )


def sd_ready(observation: StartupObservation) -> bool:
    return (
        observation.common_action == "STAND_DEFAULT"
        and observation.common_action_status == 100
        and observation.get_action == "STAND_DEFAULT"
        and observation.get_action_status == 100
        and observation.fsm_state == 4
        and observation.body_status == 1
        and observation.controller_gains_active
    )


def support_ready(observation: StartupObservation) -> bool:
    return observation.feet_supported is True and observation.support_evidence in {
        "contact_sensors", "force_sensors", "verified_simulation_configuration", "operator_confirmed",
    }


class X2StartupSupervisor:
    """Maps safe observations to supported lifecycle operations.

    The caller owns service calls, physical recovery, simulator initialization,
    and navigation publishing. This class intentionally cannot do any of them.
    """

    def decide(self, path: StartupPath, observation: StartupObservation) -> SupervisorDecision:
        if path is StartupPath.SIMULATION:
            return self._simulation(observation)
        if path is StartupPath.REAL_ROBOT:
            return self._real_robot(observation)
        if path is StartupPath.NAVIGATION:
            return self._navigation(observation)
        raise ValueError(f"unsupported startup path: {path}")

    @staticmethod
    def _blocked(path: StartupPath, reason: str) -> SupervisorDecision:
        return SupervisorDecision(path, SupervisorState.BLOCKED, (Operation.OBSERVE_ONLY,), reason, reason)

    def _simulation(self, observation: StartupObservation) -> SupervisorDecision:
        if not observation.simulation_initial_configuration_verified:
            return SupervisorDecision(
                StartupPath.SIMULATION, SupervisorState.INITIAL_CONFIGURATION_REQUIRED,
                (Operation.INITIALIZE_VERIFIED_SIMULATION_STANDING_CONFIGURATION,),
                "simulation must begin from a verified standing configuration before stepping",
            )
        if not observation.simulation_initial_support_verified:
            return self._blocked(StartupPath.SIMULATION, "simulation_support_not_verified_before_stepping")
        if not observation.simulation_generation_verified:
            return self._blocked(StartupPath.SIMULATION, "simulation_generation_not_verified")
        return self._standing_lifecycle(StartupPath.SIMULATION, observation)

    def _real_robot(self, observation: StartupObservation) -> SupervisorDecision:
        if not observation.supported_physical_standup_completed or not observation.operator_recovery_confirmed:
            return SupervisorDecision(
                StartupPath.REAL_ROBOT, SupervisorState.PHYSICAL_RECOVERY_REQUIRED,
                (Operation.REQUEST_OPERATOR_OR_SUPPORTED_PHYSICAL_RECOVERY,),
                "real robot requires supported physical stand-up or assisted recovery before JD/SD",
            )
        if not support_ready(observation):
            return self._blocked(StartupPath.REAL_ROBOT, "feet_support_not_verified")
        return self._standing_lifecycle(StartupPath.REAL_ROBOT, observation)

    def _standing_lifecycle(self, path: StartupPath, observation: StartupObservation) -> SupervisorDecision:
        fault = evidence_failure(observation)
        if fault:
            return self._blocked(path, fault)
        if not support_ready(observation):
            return self._blocked(path, "feet_support_not_verified")
        if sd_ready(observation):
            if observation.standing_hold_s >= STANDING_HOLD_S and observation.standing_unique_samples >= STANDING_MIN_UNIQUE_SAMPLES:
                return SupervisorDecision(path, SupervisorState.STABLE_IDLE, (Operation.OBSERVE_ONLY,), "active standing confirmed")
            return SupervisorDecision(path, SupervisorState.STANDING_HOLD, (Operation.OBSERVE_ONLY,), "SD ready; collect continuous physical standing evidence")
        if observation.common_action == "STAND_DEFAULT" or observation.get_action == "STAND_DEFAULT":
            return SupervisorDecision(path, SupervisorState.SD_TRANSITION, (Operation.OBSERVE_ONLY,), "SD transition pending")
        if jd_ready(observation):
            return SupervisorDecision(path, SupervisorState.SD_REQUEST_ALLOWED, (Operation.REQUEST_SD,), "JD readiness confirmed")
        if observation.common_action == "JOINT_DEFAULT" or observation.get_action == "JOINT_DEFAULT":
            return SupervisorDecision(path, SupervisorState.JD_TRANSITION, (Operation.OBSERVE_ONLY,), "JD transition pending")
        if observation.common_action in (None, "PASSIVE_DEFAULT") and observation.get_action in (None, "PASSIVE_DEFAULT"):
            return SupervisorDecision(path, SupervisorState.JD_REQUEST_ALLOWED, (Operation.REQUEST_JD,), "safe posture and support qualify JD request")
        return self._blocked(path, "unsupported_or_ambiguous_mc_action_state")

    def _navigation(self, observation: StartupObservation) -> SupervisorDecision:
        fault = evidence_failure(observation)
        if fault:
            return self._blocked(StartupPath.NAVIGATION, fault)
        if not support_ready(observation):
            return self._blocked(StartupPath.NAVIGATION, "feet_support_not_verified")
        in_locomotion = (
            observation.locomotion_mode_confirmed
            or observation.common_action == "LOCOMOTION_DEFAULT"
            or observation.get_action == "LOCOMOTION_DEFAULT"
        )
        if in_locomotion:
            if not observation.navigation_source_registered or not observation.navigation_source_selected:
                return self._blocked(StartupPath.NAVIGATION, "locomotion_without_verified_selected_source")
            if not observation.valid_nonzero_navigation_command_seen:
                return SupervisorDecision(
                    StartupPath.NAVIGATION, SupervisorState.NAVIGATION_ACTIVE,
                    (Operation.PUBLISH_BOUNDED_NAVIGATION_COMMAND,),
                    "LD confirmed; await bounded command and physical movement",
                )
            if not observation.physical_movement_confirmed:
                return self._blocked(StartupPath.NAVIGATION, "command_published_without_physical_movement")
            if not observation.zero_command_published:
                return SupervisorDecision(
                    StartupPath.NAVIGATION, SupervisorState.NAVIGATION_STOP_VERIFY,
                    (Operation.PUBLISH_ZERO_COMMAND,),
                    "task movement complete; begin stop procedure",
                )
            if not observation.physical_stop_confirmed:
                return SupervisorDecision(
                    StartupPath.NAVIGATION, SupervisorState.NAVIGATION_STOP_VERIFY,
                    (Operation.VERIFY_PHYSICAL_STOP,),
                    "zero command is not physical stopping evidence",
                )
            if not observation.navigation_source_released:
                return SupervisorDecision(
                    StartupPath.NAVIGATION, SupervisorState.NAVIGATION_STOP_VERIFY,
                    (Operation.RELEASE_NAVIGATION_SOURCE,),
                    "physical stop confirmed; release source before stable idle",
                )
            return SupervisorDecision(
                StartupPath.NAVIGATION, SupervisorState.NAVIGATION_STOP_VERIFY,
                (Operation.REQUEST_SD,),
                "source released; restore SD and independently verify stable idle",
            )
        standing = self._standing_lifecycle(StartupPath.NAVIGATION, observation)
        if standing.state is not SupervisorState.STABLE_IDLE:
            return standing
        if observation.navigation_source_released:
            return SupervisorDecision(
                StartupPath.NAVIGATION, SupervisorState.STABLE_IDLE,
                (Operation.OBSERVE_ONLY,),
                "navigation completed, source released and SD stable idle restored",
            )
        if not observation.navigation_source_registered or not observation.navigation_source_selected:
            return SupervisorDecision(StartupPath.NAVIGATION, SupervisorState.NAVIGATION_SOURCE_REQUIRED,
                                      (Operation.REGISTER_NAVIGATION_SOURCE,), "register and arbitrate a legitimate navigation source")
        return SupervisorDecision(
            StartupPath.NAVIGATION, SupervisorState.NAVIGATION_LD_PENDING,
            (Operation.REQUEST_LD,),
            "standing idle is ready; LD still requires verified mode readback",
        )


def main() -> int:
    """Small JSON demonstration; no backend, service, or simulator side effect."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=[item.value for item in StartupPath], required=True)
    parser.add_argument("--observation-json", required=True)
    args = parser.parse_args()
    observation = StartupObservation(**json.loads(open(args.observation_json, encoding="utf-8").read()))
    decision = X2StartupSupervisor().decide(StartupPath(args.path), observation)
    print(json.dumps(decision.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
