"""Simulation-only source mimic expansion and internal follower servos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np


MODEL_LABEL = "CONTROL-LEVEL / ACTUATED MIMIC EMULATION"


@dataclass(frozen=True)
class SourceJoint:
    name: str
    joint_id: int
    qpos: int
    dof: int
    axis_sign: int
    reference: float
    lower: float
    upper: float
    velocity_limit: float
    effort_limit: float


class VirtualTransmissionController:
    """Expose source active coordinates; derive all follower targets internally."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        urdf_joints: Mapping[str, Any],
        joint_references: Mapping[str, float],
        relations: list[Mapping[str, Any]],
        actuator_by_joint: Mapping[int, int],
        active_joint_names: list[str],
        *,
        kp_nm_per_rad: float,
        kv_nms_per_rad: float,
        max_torque_nm: float,
        soft_limit_margin_rad: float,
        kp_by_joint: Mapping[str, float] | None = None,
        kv_by_joint: Mapping[str, float] | None = None,
        bias_feedforward: bool = False,
    ) -> None:
        values = (kp_nm_per_rad, kv_nms_per_rad, max_torque_nm, soft_limit_margin_rad)
        if not all(np.isfinite(values)) or kp_nm_per_rad <= 0.0 or kv_nms_per_rad < 0.0 or max_torque_nm <= 0.0 or soft_limit_margin_rad <= 0.0:
            raise ValueError("virtual transmission gains and force bound must be finite and nonnegative/positive")
        self.model = model
        self.data = data
        self.kp = float(kp_nm_per_rad)
        self.kv = float(kv_nms_per_rad)
        self.max_torque_nm = float(max_torque_nm)
        self.soft_limit_margin_rad = float(soft_limit_margin_rad)
        self.bias_feedforward = bool(bias_feedforward)
        self._relations = [dict(relation) for relation in relations]
        self._coordinates = {
            joint_name: self._compile_coordinate(joint_name, node, joint_references)
            for joint_name, node in urdf_joints.items()
            if node.get("type") == "revolute"
        }
        self._active = set(active_joint_names)
        self._followers = {str(item["follower_joint"]) for item in relations}
        self._drivers = {str(item["driver_joint"]) for item in relations}
        if self._followers & self._active:
            raise ValueError("mimic followers cannot be public active command coordinates")
        if not self._drivers <= self._active:
            raise ValueError("every URDF mimic driver must be a public active source coordinate")
        if len(self._followers) != len(self._relations):
            raise ValueError("URDF mimic follower names must be unique")
        configured_kp = dict(kp_by_joint or {})
        configured_kv = dict(kv_by_joint or {})
        self._kp_by_joint = {
            name: float(configured_kp.get(name, self.kp))
            for name in self._coordinates
        }
        self._kv_by_joint = {
            name: float(configured_kv.get(name, self.kv))
            for name in self._coordinates
        }
        if any(not np.isfinite(value) or value <= 0.0 for value in self._kp_by_joint.values()):
            raise ValueError("per-joint kp values must be finite and positive")
        if any(not np.isfinite(value) or value < 0.0 for value in self._kv_by_joint.values()):
            raise ValueError("per-joint kv values must be finite and nonnegative")
        self._actuator_by_name: dict[str, int] = {}
        for joint_name, coordinate in self._coordinates.items():
            actuator_id = actuator_by_joint.get(coordinate.joint_id)
            if actuator_id is None:
                raise ValueError(f"joint has no internal/active actuator: {joint_name}")
            self._actuator_by_name[joint_name] = int(actuator_id)
        self._last_source_target: dict[str, float] = {}
        self._active_source_targets = {
            name: self.source_position(name)
            for name in self._active
        }
        self._last_active_target = dict(self._active_source_targets)
        self._last_output: dict[str, dict[str, object]] = {}
        self._last_active_output: dict[str, dict[str, float | bool | str]] = {}
        self.total_signed_work_j = 0.0
        self.total_positive_work_j = 0.0
        self.peak_abs_step_work_j = 0.0
        for name, coordinate in self._coordinates.items():
            low, high = self.operational_limits(name)
            if low >= high:
                raise ValueError(f"operational soft limits eliminate the range for {name}: {low} .. {high}")

    def _compile_coordinate(self, name: str, node: Any, refs: Mapping[str, float]) -> SourceJoint:
        joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if joint_id < 0 or int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"source revolute joint did not compile to a MuJoCo hinge: {name}")
        axis_node = node.find("axis")
        source_axis = np.fromstring("1 0 0" if axis_node is None else axis_node.get("xyz", "1 0 0"), sep=" ")
        compiled_axis = np.asarray(self.model.jnt_axis[joint_id], dtype=float)
        denominator = float(np.linalg.norm(source_axis) * np.linalg.norm(compiled_axis))
        if denominator <= 1e-14:
            raise ValueError(f"invalid source/MuJoCo hinge axis for {name}")
        cosine = float(np.dot(source_axis, compiled_axis) / denominator)
        if abs(abs(cosine) - 1.0) > 1e-6:
            raise ValueError(f"source/MuJoCo joint axes are not collinear for {name}: dot={cosine}")
        limit = node.find("limit")
        if limit is None or any(key not in limit.attrib for key in ("lower", "upper", "velocity", "effort")):
            raise ValueError(f"source joint lacks complete limit metadata: {name}")
        return SourceJoint(
            name=name,
            joint_id=joint_id,
            qpos=int(self.model.jnt_qposadr[joint_id]),
            dof=int(self.model.jnt_dofadr[joint_id]),
            axis_sign=1 if cosine >= 0.0 else -1,
            reference=float(refs.get(name, 0.0)),
            lower=float(limit.get("lower")),
            upper=float(limit.get("upper")),
            velocity_limit=float(limit.get("velocity")),
            effort_limit=float(limit.get("effort")),
        )

    @property
    def active_joint_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._active))

    @property
    def follower_joint_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._followers))

    def source_position(self, name: str, raw_qpos: float | None = None) -> float:
        coordinate = self._coordinates[name]
        raw = float(self.data.qpos[coordinate.qpos] if raw_qpos is None else raw_qpos)
        return coordinate.reference + coordinate.axis_sign * (raw - float(self.model.qpos0[coordinate.qpos]))

    def source_velocity(self, name: str) -> float:
        coordinate = self._coordinates[name]
        return coordinate.axis_sign * float(self.data.qvel[coordinate.dof])

    def operational_limits(self, name: str) -> tuple[float, float]:
        coordinate = self._coordinates[name]
        return (
            coordinate.lower + self.soft_limit_margin_rad,
            coordinate.upper - self.soft_limit_margin_rad,
        )

    def operational_target_from_state(self, name: str, source_position: float) -> float:
        """Project a command target into the operational interval; qpos is untouched."""
        low, high = self.operational_limits(name)
        value = float(source_position)
        if not np.isfinite(value):
            raise ValueError(f"non-finite source target for {name}: {value}")
        return min(high, max(low, value))

    def active_target_sources(self) -> dict[str, float]:
        return dict(self._active_source_targets)

    def _raw_target(self, name: str, source_target: float) -> float:
        coordinate = self._coordinates[name]
        return float(self.model.qpos0[coordinate.qpos]) + coordinate.axis_sign * (
            source_target - coordinate.reference
        )

    def command_active_sources(self, targets: Mapping[str, float]) -> None:
        """Write public source active target coordinates only."""
        unknown = set(targets) - self._active
        if unknown:
            raise KeyError(f"non-active/follower command coordinates rejected: {sorted(unknown)}")
        validated: dict[str, float] = {}
        for name, raw_value in targets.items():
            value = float(raw_value)
            low, high = self.operational_limits(name)
            if not np.isfinite(value) or value < low or value > high:
                raise ValueError(
                    f"source active target outside operational soft limits for {name}: {value} not in "
                    f"[{low}, {high}]"
                )
            validated[name] = value

        current_targets = self.active_target_sources()
        current_targets.update(validated)
        for relation in self._relations:
            follower = str(relation["follower_joint"])
            driver = str(relation["driver_joint"])
            derived = float(relation["multiplier"]) * current_targets[driver] + float(relation["offset"])
            low, high = self.operational_limits(follower)
            if not np.isfinite(derived) or derived < low or derived > high:
                raise ValueError(
                    f"derived follower target outside operational soft limits for {follower}: "
                    f"{derived} not in [{low}, {high}]"
                )

        self._active_source_targets.update(validated)

    def hold_current_active_state(self) -> None:
        current = {
            name: self.operational_target_from_state(name, self.source_position(name))
            for name in self._active
        }
        self.command_active_sources(current)

    def write_internal_controls(self) -> dict[str, dict[str, object]]:
        """Apply bounded bias-aware active and internally derived follower torques."""
        mujoco.mj_forward(self.model, self.data)
        dt = float(self.model.opt.timestep)
        driver_targets = {name: self._active_source_targets[name] for name in self._drivers}
        driver_velocities: dict[str, float] = {}
        for name, target in driver_targets.items():
            previous = self._last_source_target.get(name, self.source_position(name))
            velocity = (target - previous) / dt
            if not np.isfinite(velocity) or abs(velocity) > self._coordinates[name].velocity_limit + 1e-8:
                raise ValueError(f"derived source target velocity exceeds limit for {name}: {velocity}")
            driver_velocities[name] = velocity

        active_output: dict[str, dict[str, float | bool | str]] = {}
        for name in self._active:
            coordinate = self._coordinates[name]
            target = self._active_source_targets[name]
            previous = self._last_active_target.get(name, self.source_position(name))
            target_velocity = (target - previous) / dt
            if not np.isfinite(target_velocity) or abs(target_velocity) > coordinate.velocity_limit + 1e-8:
                raise ValueError(f"active target velocity exceeds source limit for {name}: {target_velocity}")
            raw_position_error = self._raw_target(name, target) - float(self.data.qpos[coordinate.qpos])
            raw_target_velocity = coordinate.axis_sign * target_velocity
            raw_velocity_error = raw_target_velocity - float(self.data.qvel[coordinate.dof])
            kp = self._kp_by_joint[name]
            kv = self._kv_by_joint[name]
            bias_ff = float(self.data.qfrc_bias[coordinate.dof]) if self.bias_feedforward else 0.0
            requested_torque = bias_ff + kp * raw_position_error + kv * raw_velocity_error
            actuator_id = self._actuator_by_name[name]
            gear = float(self.model.actuator_gear[actuator_id, 0])
            if not np.isfinite(gear) or abs(gear) < 1e-12:
                raise ValueError(f"invalid active motor gear for {name}: {gear}")
            effort_bound = min(self.max_torque_nm, coordinate.effort_limit) * abs(gear)
            torque = float(np.clip(requested_torque, -effort_bound, effort_bound))
            command = torque / gear
            if bool(self.model.actuator_ctrllimited[actuator_id]):
                low, high = self.model.actuator_ctrlrange[actuator_id]
                command = float(np.clip(command, float(low), float(high)))
            self.data.ctrl[actuator_id] = command
            active_output[name] = {
                "target_source_rad": float(target),
                "target_velocity_source_rad_s": float(target_velocity),
                "position_source_rad": self.source_position(name),
                "velocity_source_rad_s": self.source_velocity(name),
                "raw_position_error_rad": float(raw_position_error),
                "raw_velocity_error_rad_s": float(raw_velocity_error),
                "bias_feedforward_nm": bias_ff,
                "kp_nm_per_rad": kp,
                "kv_nms_per_rad": kv,
                "requested_torque_nm": requested_torque,
                "commanded_torque_nm": torque,
                "effort_limit_nm": effort_bound,
                "saturated": abs(requested_torque) > effort_bound + 1e-12,
            }

        # Evaluate the current active-driver accelerations without advancing time.
        mujoco.mj_forward(self.model, self.data)
        mass = np.empty((self.model.nv, self.model.nv), dtype=float)
        mujoco.mj_fullM(self.model, mass, self.data.qM)
        output: dict[str, dict[str, object]] = {}
        for relation in self._relations:
            follower = str(relation["follower_joint"])
            driver = str(relation["driver_joint"])
            multiplier = float(relation["multiplier"])
            offset = float(relation["offset"])
            target = multiplier * driver_targets[driver] + offset
            target_velocity = multiplier * driver_velocities[driver]
            coordinate = self._coordinates[follower]
            low, high = self.operational_limits(follower)
            if target < low or target > high:
                raise ValueError(f"derived follower target outside operational soft limits for {follower}: {target} not in [{low}, {high}]")
            if abs(target_velocity) > coordinate.velocity_limit + 1e-8:
                raise ValueError(f"derived follower velocity target exceeds source limit for {follower}: {target_velocity}")
            raw_target = self._raw_target(follower, target)
            raw_target_velocity = coordinate.axis_sign * target_velocity
            position_error = raw_target - float(self.data.qpos[coordinate.qpos])
            velocity_error = raw_target_velocity - float(self.data.qvel[coordinate.dof])
            kp = self._kp_by_joint[follower]
            kv = self._kv_by_joint[follower]
            bias_ff = float(self.data.qfrc_bias[coordinate.dof]) if self.bias_feedforward else 0.0
            coupling_by_driver = {
                driver_name: float(
                    mass[coordinate.dof, self._coordinates[driver_name].dof]
                    * self.data.qacc[self._coordinates[driver_name].dof]
                )
                for driver_name in sorted(self._drivers)
            }
            if not np.isfinite(list(coupling_by_driver.values())).all():
                raise ValueError(f"non-finite dynamic coupling estimate for {follower}")
            coupling_feedforward = float(sum(coupling_by_driver.values()))
            pd_torque = kp * position_error + kv * velocity_error
            requested_torque = bias_ff + pd_torque + coupling_feedforward
            effort_bound = min(self.max_torque_nm, coordinate.effort_limit)
            torque = float(np.clip(requested_torque, -effort_bound, effort_bound))
            actuator_id = self._actuator_by_name[follower]
            gear = float(self.model.actuator_gear[actuator_id, 0])
            if not np.isfinite(gear) or abs(gear) < 1e-12:
                raise ValueError(f"invalid internal follower motor gear for {follower}: {gear}")
            command = torque / gear
            if bool(self.model.actuator_ctrllimited[actuator_id]):
                low, high = self.model.actuator_ctrlrange[actuator_id]
                command = float(np.clip(command, float(low), float(high)))
            self.data.ctrl[actuator_id] = command
            output[follower] = {
                "driver_joint": driver,
                "driver_target_source_rad": driver_targets[driver],
                "driver_target_velocity_source_rad_s": driver_velocities[driver],
                "multiplier": multiplier,
                "offset_rad": offset,
                "follower_target_source_rad": target,
                "follower_target_velocity_source_rad_s": target_velocity,
                "follower_position_source_rad": self.source_position(follower),
                "follower_velocity_source_rad_s": self.source_velocity(follower),
                "follower_position_error_to_target_rad": self.source_position(follower) - target,
                "requested_torque_nm": requested_torque,
                "bias_feedforward_nm": bias_ff,
                "pd_torque_nm": pd_torque,
                "dynamic_coupling_feedforward_nm": coupling_feedforward,
                "dynamic_coupling_by_driver_nm": coupling_by_driver,
                "kp_nm_per_rad": kp,
                "kv_nms_per_rad": kv,
                "commanded_torque_nm": torque,
                "effort_limit_nm": effort_bound,
                "motor_gear": gear,
                "saturated": abs(requested_torque) > effort_bound + 1e-12,
            }
        mujoco.mj_forward(self.model, self.data)
        self._last_source_target.update(driver_targets)
        self._last_active_target.update(self._active_source_targets)
        self._last_output = output
        self._last_active_output = active_output
        return output

    def active_measure(self) -> dict[str, dict[str, float | bool | str]]:
        result: dict[str, dict[str, float | bool | str]] = {}
        for name in self._active:
            coordinate = self._coordinates[name]
            output = self._last_active_output.get(name, {})
            actuator_id = self._actuator_by_name[name]
            gear = float(self.model.actuator_gear[actuator_id, 0])
            result[name] = {
                "target_source_rad": float(self._active_source_targets[name]),
                "target_velocity_source_rad_s": float(output.get("target_velocity_source_rad_s", 0.0)),
                "position_source_rad": self.source_position(name),
                "velocity_source_rad_s": self.source_velocity(name),
                "target_tracking_error_rad": self.source_position(name) - float(self._active_source_targets[name]),
                "bias_feedforward_nm": float(output.get("bias_feedforward_nm", 0.0)),
                "pd_torque_nm": float(output.get("pd_torque_nm", float(output.get("requested_torque_nm", 0.0)) - float(output.get("bias_feedforward_nm", 0.0)))),
                "requested_torque_nm": float(output.get("requested_torque_nm", 0.0)),
                "commanded_torque_nm": float(output.get("commanded_torque_nm", 0.0)),
                "actual_actuator_force_nm": float(self.data.actuator_force[actuator_id]) * gear,
                "effort_limit_nm": float(output.get("effort_limit_nm", min(self.max_torque_nm, coordinate.effort_limit))),
                "saturated": bool(output.get("saturated", False)),
            }
        return result

    def measure(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for relation in self._relations:
            follower = str(relation["follower_joint"])
            driver = str(relation["driver_joint"])
            multiplier = float(relation["multiplier"])
            offset = float(relation["offset"])
            driver_state = self.source_position(driver)
            follower_state = self.source_position(follower)
            driver_target = self._active_source_targets[driver]
            follower_target = multiplier * driver_target + offset
            target_error = follower_state - follower_target
            state_error = follower_state - (multiplier * driver_state + offset)
            output = self._last_output.get(follower, {})
            actuator_id = self._actuator_by_name[follower]
            gear = float(self.model.actuator_gear[actuator_id, 0])
            result[follower] = {
                "driver_joint": driver,
                "multiplier": multiplier,
                "offset_rad": offset,
                "driver_position_source_rad": driver_state,
                "driver_target_source_rad": driver_target,
                "follower_position_source_rad": follower_state,
                "follower_target_source_rad": follower_target,
                "follower_target_velocity_source_rad_s": float(output.get("follower_target_velocity_source_rad_s", 0.0)),
                "follower_velocity_source_rad_s": self.source_velocity(follower),
                "target_tracking_error_rad": target_error,
                "source_relation_error_rad": state_error,
                "command_relation_error_rad": follower_target - (multiplier * driver_target + offset),
                "bias_feedforward_nm": float(output.get("bias_feedforward_nm", 0.0)),
                "dynamic_coupling_feedforward_nm": float(output.get("dynamic_coupling_feedforward_nm", 0.0)),
                "dynamic_coupling_by_driver_nm": dict(output.get("dynamic_coupling_by_driver_nm", {})),
                "pd_torque_nm": float(output.get("pd_torque_nm", 0.0)),
                "requested_torque_nm": float(output.get("requested_torque_nm", 0.0)),
                "commanded_torque_nm": float(output.get("commanded_torque_nm", 0.0)),
                "actual_actuator_force_nm": float(self.data.actuator_force[actuator_id]) * gear,
                "effort_limit_nm": float(output.get("effort_limit_nm", min(self.max_torque_nm, self._coordinates[follower].effort_limit))),
                "saturated": bool(output.get("saturated", False)),
            }
        return result

    def record_step_work(self) -> dict[str, float]:
        dt = float(self.model.opt.timestep)
        signed_work = 0.0
        for follower in self._followers:
            actuator_id = self._actuator_by_name[follower]
            coordinate = self._coordinates[follower]
            generalized_torque = float(self.data.actuator_force[actuator_id]) * float(self.model.actuator_gear[actuator_id, 0])
            signed_work += generalized_torque * float(self.data.qvel[coordinate.dof]) * dt
        positive_work = max(0.0, signed_work)
        self.total_signed_work_j += signed_work
        self.total_positive_work_j += positive_work
        self.peak_abs_step_work_j = max(self.peak_abs_step_work_j, abs(signed_work))
        return {
            "signed_work_step_j": signed_work,
            "positive_work_step_j": positive_work,
            "cumulative_signed_work_j": self.total_signed_work_j,
            "cumulative_positive_work_j": self.total_positive_work_j,
            "peak_abs_step_work_j": self.peak_abs_step_work_j,
        }
