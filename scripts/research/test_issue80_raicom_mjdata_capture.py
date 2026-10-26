"""Focused offline checks for the Issue #80 native MjData capture overlay."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from issue80_raicom_mjdata_capture import (
    CAPTURE_SOURCE,
    CAPTURE_SCHEMA,
    HEADER_SHA256,
    LAUNCH_MARKER,
    MjDataCaptureError,
    instrument_staged_launch,
    summarize_mjdata_capture,
)


def metadata() -> dict:
    return {
        "schema": CAPTURE_SCHEMA,
        "event": "model_metadata",
        "host_monotonic_ns": 10,
        "sim_time_s": 0.0,
        "mujoco_runtime_version": "3.3.7",
    }


def sample(
    *,
    sim_time: float,
    host_time: int,
    generation: int = 0,
    vx: float = 0.1,
    vy: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    contacts: list[dict] | None = None,
) -> dict:
    return {
        "schema": CAPTURE_SCHEMA,
        "event": "post_step",
        "host_monotonic_ns": host_time,
        "sim_time_s": sim_time,
        "generation": generation,
        "pelvis": {
            "state": {
                "world_position_m": [x, y, 0.68],
                "world_spatial_velocity_rot_lin": [0, 0, 0, vx, vy, 0],
            }
        },
        "actuator_force": [1.0, -2.0],
        "qfrc_actuator": [0.0, 4.0],
        "contact_count": len(contacts or []),
        "contacts": contacts or [],
    }


def foot_floor_contact(*, left: bool, normal: float = 20.0) -> dict:
    return {
        "geom1": {"id": 0, "name": "floor", "body_name": "world"},
        "geom2": {"id": 1, "name": "foot", "body_name": "left_ankle_roll_link"},
        "distance_m": -0.0002,
        "involves_selected_left_foot": left,
        "involves_selected_right_foot": not left,
        "normal_force_n": normal,
        "tangential_force_norm_n": 2.0,
        "relative_tangential_speed_m_s": 0.03,
    }


class MjDataCaptureTests(unittest.TestCase):
    def write_trace(self, directory: Path, rows: list[dict]) -> Path:
        path = directory / "native_mjdata_trace.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_pinned_header_manifest_is_complete_and_exact_version_specific(self):
        self.assertEqual(len(HEADER_SHA256), 13)
        self.assertEqual(len(set(HEADER_SHA256.values())), len(HEADER_SHA256))
        self.assertEqual(len(HEADER_SHA256["mujoco.h"]), 64)
        self.assertIn("mjdata.h", HEADER_SHA256)
        self.assertIn("mjmodel.h", HEADER_SHA256)

    def test_hook_source_has_no_native_control_or_state_write_path(self):
        source = CAPTURE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("void mj_step(", source)
        self.assertIn("void mj_forward(", source)
        self.assertIn("real_mj_step(m, d);", source)
        self.assertIn("real_mj_forward(m, d);", source)
        for forbidden in (
            "mj_resetData",
            "mj_setState",
            "mj_setConst",
            "d->ctrl[",
            "d->qpos[",
            "d->qvel[",
        ):
            self.assertNotIn(forbidden, source)

    def test_instruments_only_the_evidence_local_launch_script(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            script = directory / "launch_sim.sh"
            original = (
                "#!/bin/bash\n"
                "export LD_LIBRARY_PATH=/runtime/lib:$LD_LIBRARY_PATH\n"
                "exec /runtime/aima-sim-app\n"
            )
            script.write_text(original, encoding="utf-8")
            hashes = instrument_staged_launch(
                script,
                library_container_path="/workspace/evidence/libcapture.so",
                trace_container_path="/workspace/evidence/trace.jsonl",
                interval_steps=10,
            )
            observed = script.read_text(encoding="utf-8")
            self.assertIn(LAUNCH_MARKER, observed)
            self.assertIn(
                "export LD_PRELOAD=/workspace/evidence/libcapture.so"
                + "$" + "{LD_PRELOAD:+:$" + "{LD_PRELOAD}}",
                observed,
            )
            self.assertIn("ROBOTSIM_MJDATA_CAPTURE_INTERVAL_STEPS=10", observed)
            self.assertEqual(observed.count(LAUNCH_MARKER), 1)
            self.assertNotEqual(hashes["original_sha256"], hashes["instrumented_sha256"])
            with self.assertRaises(MjDataCaptureError):
                instrument_staged_launch(
                    script,
                    library_container_path="/workspace/evidence/libcapture.so",
                    trace_container_path="/workspace/evidence/trace.jsonl",
                    interval_steps=10,
                )

    def test_rejects_invalid_capture_interval_before_script_write(self):
        with tempfile.TemporaryDirectory() as raw:
            script = Path(raw) / "launch_sim.sh"
            original = "export LD_LIBRARY_PATH=/runtime/lib\n"
            script.write_text(original, encoding="utf-8")
            with self.assertRaises(MjDataCaptureError):
                instrument_staged_launch(
                    script,
                    library_container_path="/workspace/evidence/libcapture.so",
                    trace_container_path="/workspace/evidence/trace.jsonl",
                    interval_steps=0,
                )
            self.assertEqual(script.read_text(encoding="utf-8"), original)

    def test_summary_uses_native_contact_and_world_velocity_values(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.write_trace(
                Path(raw),
                [
                    metadata(),
                    sample(
                        sim_time=0.0,
                        host_time=20,
                        x=0.0,
                        contacts=[foot_floor_contact(left=True)],
                    ),
                    sample(
                        sim_time=0.02,
                        host_time=30,
                        x=0.003,
                        contacts=[foot_floor_contact(left=False, normal=22.0)],
                    ),
                ],
            )
            summary = summarize_mjdata_capture(path)
            self.assertEqual(summary["status"], "PASS_NATIVE_MJDATA_CAPTURE")
            self.assertAlmostEqual(summary["max_pelvis_world_planar_speed_m_s"], 0.1)
            self.assertAlmostEqual(
                summary["pelvis_finite_difference_displacement"]["planar_m"], 0.003
            )
            contacts = summary["foot_floor_contact_metrics"]
            self.assertEqual(contacts["max_left_foot_floor_normal_force_n"], 20.0)
            self.assertEqual(contacts["max_right_foot_floor_normal_force_n"], 22.0)
            self.assertEqual(summary["max_abs_actuator_force"], 2.0)
            self.assertEqual(summary["max_abs_generalized_actuator_force"], 4.0)
            self.assertEqual(summary["current_generation_sample_count"], 2)

    def test_capture_error_and_timestamp_regression_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.write_trace(
                Path(raw),
                [
                    metadata(),
                    sample(sim_time=0.04, host_time=20),
                    sample(sim_time=0.02, host_time=30),
                    {
                        "schema": CAPTURE_SCHEMA,
                        "event": "capture_error",
                        "host_monotonic_ns": 40,
                        "code": "VERSION_MISMATCH",
                    },
                ],
            )
            summary = summarize_mjdata_capture(path)
            self.assertEqual(summary["status"], "BLOCKED_NATIVE_MJDATA_CAPTURE")
            self.assertIn("CAPTURE_ERROR_EVENT", summary["failure_reasons"])
            self.assertIn(
                "NONMONOTONIC_SIM_TIMESTAMP_WITHIN_GENERATION",
                summary["failure_reasons"],
            )

    def test_generation_boundary_allows_sim_time_restart(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.write_trace(
                Path(raw),
                [
                    metadata(),
                    sample(sim_time=0.2, host_time=20, generation=0),
                    sample(sim_time=0.0, host_time=30, generation=1),
                    sample(sim_time=0.02, host_time=40, generation=1),
                ],
            )
            summary = summarize_mjdata_capture(path)
            self.assertEqual(summary["status"], "PASS_NATIVE_MJDATA_CAPTURE")
            self.assertEqual(summary["generation_count"], 2)
            self.assertEqual(summary["current_generation"], 1)


if __name__ == "__main__":
    unittest.main()
