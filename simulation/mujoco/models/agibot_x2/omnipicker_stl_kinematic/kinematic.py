"""Table-driven X2 OmniPicker STL kinematics.

This module deliberately implements a kinematic simulation overlay.  It does
not approximate or simulate the undocumented internal transmission: every
integer command uses the corresponding human-reviewed source-STL fit row.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Any


STL_MODEL_ID = "X2-OmniPicker-STL"
TRAJECTORY_COLUMNS = {
    "right_claw_joint": "root_q_rad",
    "R_hand_wide1_joint": "wide_root_q_rad",
    "R_hand_narrow3_joint": "narrow3_q_rad",
    "R_hand_narrow_loop_joint": "narrow_loop_q_rad",
    "R_hand_wide3_joint": "wide3_q_rad",
    "R_hand_wide_loop_joint": "wide_loop_q_rad",
}
LEFT_JOINT_MAPPING = {
    "right_claw_joint": "left_claw_joint",
    "R_hand_wide1_joint": "L_hand_wide1_joint",
    "R_hand_narrow3_joint": "L_hand_narrow3_joint",
    "R_hand_narrow_loop_joint": "L_hand_narrow_loop_joint",
    "R_hand_wide3_joint": "L_hand_wide3_joint",
    "R_hand_wide_loop_joint": "L_hand_wide_loop_joint",
}
_REQUIRED_COLUMNS = ("command_percent", *TRAJECTORY_COLUMNS.values())


class STLKinematicTrajectoryError(RuntimeError):
    """The registered table cannot safely drive the selected STL model."""


@dataclass(frozen=True)
class KinematicPose:
    """A deterministic source-STL pose at a normalized command percentage."""

    command_percent: float
    positions_rad: dict[str, float]
    velocities_rad_s: dict[str, float]


@dataclass(frozen=True)
class KinematicTrajectory:
    """The exact 101 reviewed samples plus explicit linear interpolation."""

    rows: tuple[dict[str, float], ...]

    @classmethod
    def from_csv(cls, path: Path) -> "KinematicTrajectory":
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or any(column not in reader.fieldnames for column in _REQUIRED_COLUMNS):
                    raise STLKinematicTrajectoryError("trajectory CSV lacks required source-STL columns")
                rows = []
                for row in reader:
                    parsed = {column: float(row[column]) for column in _REQUIRED_COLUMNS}
                    if not all(math.isfinite(value) for value in parsed.values()):
                        raise STLKinematicTrajectoryError("trajectory CSV contains non-finite values")
                    rows.append(parsed)
        except (OSError, ValueError, csv.Error) as exc:
            raise STLKinematicTrajectoryError(f"could not read registered trajectory: {path}") from exc
        if len(rows) != 101:
            raise STLKinematicTrajectoryError("trajectory must contain exactly the 101 reviewed 0-100 samples")
        if [int(row["command_percent"]) for row in rows] != list(range(101)):
            raise STLKinematicTrajectoryError("trajectory command samples must be the integer sequence 0..100")
        if any(abs(row["command_percent"] - round(row["command_percent"])) > 1e-12 for row in rows):
            raise STLKinematicTrajectoryError("trajectory command samples must be integral")
        for index, row in enumerate(rows):
            if abs(row["root_q_rad"] + 0.01 * index) > 1e-12:
                raise STLKinematicTrajectoryError("trajectory root mapping differs from the registered 0..100 command convention")
            if abs(row["wide_root_q_rad"] + row["root_q_rad"]) > 1e-12:
                raise STLKinematicTrajectoryError("trajectory wide-root mimic mapping differs from the source table")
        return cls(rows=tuple(rows))

    def pose(self, command_percent: float, *, command_rate_percent_s: float = 0.0) -> KinematicPose:
        if not math.isfinite(command_percent) or not math.isfinite(command_rate_percent_s):
            raise STLKinematicTrajectoryError("command and command rate must be finite")
        if command_percent < 0.0 or command_percent > 100.0:
            raise STLKinematicTrajectoryError("command percent must remain within the reviewed 0..100 range")
        low = int(math.floor(command_percent))
        high = min(low + 1, 100)
        alpha = command_percent - low
        positions: dict[str, float] = {}
        velocities: dict[str, float] = {}
        for joint, column in TRAJECTORY_COLUMNS.items():
            low_value = self.rows[low][column]
            high_value = self.rows[high][column]
            positions[joint] = low_value + alpha * (high_value - low_value)
            if high > low:
                slope_per_percent = high_value - low_value
            elif low > 0:
                slope_per_percent = low_value - self.rows[low - 1][column]
            else:
                slope_per_percent = 0.0
            velocities[joint] = slope_per_percent * command_rate_percent_s
        return KinematicPose(
            command_percent=command_percent,
            positions_rad=positions,
            velocities_rad_s=velocities,
        )


def load_registered_trajectory(package: Path) -> KinematicTrajectory:
    """Load the trajectory shipped with the selected package, never a reviews path."""
    return KinematicTrajectory.from_csv(package / "trajectory.csv")


def apply_stl_kinematic_pose(
    mujoco: Any,
    model: Any,
    data: Any,
    trajectory: KinematicTrajectory,
    command_percent: float,
    *,
    command_rate_percent_s: float = 0.0,
    side: str = "right",
    forward: bool = True,
) -> KinematicPose:
    """Write the registered table pose to an approved hand-side mapping.

    This is intentionally a kinematic driver.  It must not be used to claim
    dynamic transmission, contact, load-transfer, or grasp validation.
    """
    if side not in {"right", "left"}:
        raise STLKinematicTrajectoryError("side must be 'right' or 'left'")
    pose = trajectory.pose(command_percent, command_rate_percent_s=command_rate_percent_s)
    joint_mapping = {joint: joint for joint in TRAJECTORY_COLUMNS} if side == "right" else LEFT_JOINT_MAPPING
    positions: dict[str, float] = {}
    velocities: dict[str, float] = {}
    for joint, position in pose.positions_rad.items():
        target_joint = joint_mapping[joint]
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target_joint)
        if joint_id < 0:
            raise STLKinematicTrajectoryError(f"selected model lacks {side} table-driven joint: {target_joint}")
        data.qpos[model.jnt_qposadr[joint_id]] = position
        velocity = pose.velocities_rad_s[joint]
        data.qvel[model.jnt_dofadr[joint_id]] = velocity
        positions[target_joint] = position
        velocities[target_joint] = velocity
    if forward:
        mujoco.mj_forward(model, data)
    return KinematicPose(
        command_percent=pose.command_percent,
        positions_rad=positions,
        velocities_rad_s=velocities,
    )
