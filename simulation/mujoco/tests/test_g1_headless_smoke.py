"""Headless load-and-step coverage for the pinned upstream Unitree G1 MJCFs."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LOCK_FILE = REPOSITORY_ROOT / "third_party" / "LOCK.md"
MODEL_CHECKOUT = Path(
    os.environ.get(
        "ROBOTSIM_UNITREE_MUJOCO_DIR",
        REPOSITORY_ROOT / "third_party" / "unitree_mujoco",
    )
).resolve()
MODEL_DIRECTORY = MODEL_CHECKOUT / "unitree_robots" / "g1"

EXPECTED_DIMENSIONS = {
    "g1_23dof.xml": {
        "nq": 36,
        "nv": 35,
        "nu": 29,
        "njnt": 30,
        "nbody": 31,
        "ngeom": 62,
        "nsensordata": 113,
    },
    "g1_29dof.xml": {
        "nq": 36,
        "nv": 35,
        "nu": 29,
        "njnt": 30,
        "nbody": 31,
        "ngeom": 73,
        "nsensordata": 113,
    },
}
IMPORTANT_JOINTS = (
    "floating_base_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_elbow_joint",
)
ACTUATOR_NAME = "left_shoulder_pitch"
JOINT_NAME = "left_shoulder_pitch_joint"
REQUESTED_COMMAND_NM = 1.0
MAX_ABS_COMMAND_NM = 0.5
STEP_COUNT = 50
STATE_CHANGE_TOLERANCE_RAD = 1e-3
REPLAY_TOLERANCE = 1e-12

FINITE_STATE_ARRAYS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "actuator_force",
    "qacc",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "qfrc_actuator",
    "qfrc_constraint",
    "qfrc_bias",
    "qfrc_passive",
    "qfrc_spring",
    "qfrc_damper",
    "qfrc_gravcomp",
    "qfrc_fluid",
    "xpos",
    "xquat",
    "xmat",
    "xipos",
    "ximat",
    "sensordata",
)


def bounded_actuator_command(
    requested: float,
    max_abs_command: float,
    actuator_ctrlrange: Sequence[float],
) -> float:
    """Clamp one requested motor command to a test cap and MJCF ctrlrange."""

    command = float(requested)
    cap = float(max_abs_command)
    limits = np.asarray(actuator_ctrlrange, dtype=float)
    if not math.isfinite(command):
        raise ValueError("requested command must be finite")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("command cap must be finite and positive")
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] > limits[1]:
        raise ValueError("actuator ctrlrange must contain two finite ordered limits")

    lower = max(-cap, float(limits[0]))
    upper = min(cap, float(limits[1]))
    if lower > upper:
        raise ValueError("command cap and actuator ctrlrange do not overlap")
    return min(max(command, lower), upper)


def _locked_value(pattern: str, label: str) -> str:
    text = LOCK_FILE.read_text(encoding="utf-8")
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise AssertionError(f"expected one {label} value in {LOCK_FILE}, found {len(matches)}")
    return matches[0]


def _locked_model_commit() -> str:
    return _locked_value(
        r"^\|\s*unitree_mujoco\s*\|[^|]*\|\s*([0-9a-f]{40})\s*\|",
        "Unitree MuJoCo commit",
    )


def _locked_mujoco_version() -> str:
    return _locked_value(
        r"^- Baseline: MuJoCo ([0-9]+\.[0-9]+\.[0-9]+),", "MuJoCo version"
    )


def _source_commit_and_cleanliness() -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "-C", str(MODEL_CHECKOUT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(MODEL_CHECKOUT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, status


def _model_dimension(model: mujoco.MjModel, field: str) -> int:
    return int(getattr(model, field))


class BoundedCommandUnitTest(unittest.TestCase):
    """Pure command-clamping checks (L1)."""

    def test_caps_both_command_signs(self) -> None:
        ctrlrange = (-25.0, 25.0)
        self.assertEqual(bounded_actuator_command(8.0, 0.5, ctrlrange), 0.5)
        self.assertEqual(bounded_actuator_command(-8.0, 0.5, ctrlrange), -0.5)

    def test_respects_narrower_model_ctrlrange(self) -> None:
        self.assertEqual(bounded_actuator_command(1.0, 0.5, (-0.2, 0.1)), 0.1)

    def test_rejects_nonfinite_or_invalid_bounds(self) -> None:
        with self.assertRaises(ValueError):
            bounded_actuator_command(float("nan"), 0.5, (-25.0, 25.0))
        with self.assertRaises(ValueError):
            bounded_actuator_command(1.0, 0.5, (1.0, 2.0))


class G1MjcfHeadlessSmokeTest(unittest.TestCase):
    """Load both pinned MJCFs and step their physics without a renderer (L2)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.expected_commit = _locked_model_commit()
        cls.expected_mujoco_version = _locked_mujoco_version()
        cls.source_commit, cls.source_status = _source_commit_and_cleanliness()
        if cls.source_commit != cls.expected_commit:
            raise AssertionError(
                f"unitree_mujoco checkout is {cls.source_commit}, expected locked "
                f"commit {cls.expected_commit}"
            )
        if cls.source_status:
            raise AssertionError("unitree_mujoco checkout must be clean before testing")
        if mujoco.__version__ != cls.expected_mujoco_version:
            raise AssertionError(
                f"MuJoCo Python package is {mujoco.__version__}, expected "
                f"{cls.expected_mujoco_version} from {LOCK_FILE}"
            )
        cls.native_version = mujoco.mj_versionString()
        if cls.native_version != cls.expected_mujoco_version:
            raise AssertionError(
                f"MuJoCo native library is {cls.native_version}, expected "
                f"{cls.expected_mujoco_version}"
            )

    def assert_state_is_finite(self, data: mujoco.MjData, variant: str, step: int) -> None:
        self.assertTrue(math.isfinite(float(data.time)), f"{variant} time at step {step}")
        for field in FINITE_STATE_ARRAYS:
            values = np.asarray(getattr(data, field))
            self.assertTrue(
                np.isfinite(values).all(),
                f"{variant} {field} contains a non-finite value at step {step}",
            )

    def _step(self, model: mujoco.MjModel, command: float) -> mujoco.MjData:
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        for step in range(1, STEP_COUNT + 1):
            if command != 0.0:
                data.ctrl[self.actuator_id] = command
            mujoco.mj_step(model, data)
            self.assert_state_is_finite(data, self.variant, step)
        return data

    def test_both_official_variants_load_step_and_respond_to_bounded_input(self) -> None:
        for self.variant, expected in EXPECTED_DIMENSIONS.items():
            with self.subTest(model_variant=self.variant):
                model_path = MODEL_DIRECTORY / self.variant
                self.assertTrue(model_path.is_file(), f"missing pinned MJCF: {model_path}")
                model = mujoco.MjModel.from_xml_path(str(model_path))
                actual_dimensions = {
                    field: _model_dimension(model, field) for field in expected
                }
                self.assertEqual(actual_dimensions, expected)
                self.assertAlmostEqual(float(model.opt.timestep), 0.002, places=12)

                joint_ids = {}
                for joint_name in IMPORTANT_JOINTS:
                    joint_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    self.assertNotEqual(joint_id, -1, f"missing joint {joint_name}")
                    joint_ids[joint_name] = joint_id
                self.assertEqual(
                    int(model.jnt_type[joint_ids["floating_base_joint"]]),
                    int(mujoco.mjtJoint.mjJNT_FREE),
                )

                self.actuator_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATOR_NAME
                )
                self.assertNotEqual(self.actuator_id, -1, f"missing actuator {ACTUATOR_NAME}")
                self.assertTrue(bool(model.actuator_ctrllimited[self.actuator_id]))
                command = bounded_actuator_command(
                    REQUESTED_COMMAND_NM,
                    MAX_ABS_COMMAND_NM,
                    model.actuator_ctrlrange[self.actuator_id],
                )
                self.assertLessEqual(abs(command), MAX_ABS_COMMAND_NM)
                self.assertGreaterEqual(command, model.actuator_ctrlrange[self.actuator_id][0])
                self.assertLessEqual(command, model.actuator_ctrlrange[self.actuator_id][1])

                self.joint_id = joint_ids[JOINT_NAME]
                qpos_address = int(model.jnt_qposadr[self.joint_id])
                zero_command_data = self._step(model, 0.0)
                commanded_data = self._step(model, command)
                replay_data = self._step(model, command)

                zero_qpos = float(zero_command_data.qpos[qpos_address])
                commanded_qpos = float(commanded_data.qpos[qpos_address])
                replay_fields = (
                    "qpos",
                    "qvel",
                    "act",
                    "ctrl",
                    "actuator_force",
                    "qacc",
                    "sensordata",
                )
                replay_error = 0.0
                for field in replay_fields:
                    commanded = np.asarray(getattr(commanded_data, field))
                    replayed = np.asarray(getattr(replay_data, field))
                    if commanded.size:
                        replay_error = max(
                            replay_error,
                            float(np.max(np.abs(commanded - replayed))),
                        )
                replay_error = max(
                    replay_error, abs(float(commanded_data.time - replay_data.time))
                )
                joint_delta = commanded_qpos - zero_qpos
                self.assertGreater(
                    commanded_qpos,
                    zero_qpos + STATE_CHANGE_TOLERANCE_RAD,
                    f"positive input did not move {JOINT_NAME} in the positive direction",
                )
                self.assertLessEqual(replay_error, REPLAY_TOLERANCE)
                self.assertAlmostEqual(
                    float(commanded_data.time), STEP_COUNT * model.opt.timestep, places=12
                )

                print(
                    "G1_MUJOCO_SMOKE "
                    + json.dumps(
                        {
                            "mujoco_python_version": mujoco.__version__,
                            "mujoco_native_version": self.native_version,
                            "python_version": sys.version.split()[0],
                            "numpy_version": np.__version__,
                            "platform": platform.platform(),
                            "architecture": platform.machine(),
                            "model_source_commit": self.source_commit,
                            "model_variant": self.variant,
                            "dimensions": actual_dimensions,
                            "timestep_seconds": float(model.opt.timestep),
                            "steps": STEP_COUNT,
                            "simulated_seconds": float(commanded_data.time),
                            "actuator": ACTUATOR_NAME,
                            "joint": JOINT_NAME,
                            "requested_command_nm": REQUESTED_COMMAND_NM,
                            "command_limit_nm": MAX_ABS_COMMAND_NM,
                            "applied_command_nm": command,
                            "zero_command_joint_qpos_rad": zero_qpos,
                            "commanded_joint_qpos_rad": commanded_qpos,
                            "joint_delta_rad": joint_delta,
                            "replay_max_abs_state_error": replay_error,
                            "random_seed": None,
                            "reproducibility": (
                                "fixed initial state, command, and step count; "
                                "same-process replay"
                            ),
                        },
                        sort_keys=True,
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
