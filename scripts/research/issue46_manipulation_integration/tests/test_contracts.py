import math
import unittest

from scripts.research.issue46_manipulation_integration.contracts import (
    ContractError, DockingHandoff, DockingLimits, DockStatus, PoseSample, PoseSource,
    TimeDomain, TransformSample, evaluate_docking_handoff, transform_pose_chain,
)


def pose(*, obj="bottle", frame="camera_optical", xyz=(1.0, 0.0, 0.0), stamp=10,
         domain=TimeDomain.ROS, source=PoseSource.MEASUREMENT, **kw):
    return PoseSample(obj, frame, xyz, (0.0, 0.0, 0.0, 1.0), stamp, source, domain,
                      calibration_id="calib-1" if source in (PoseSource.MEASUREMENT, PoseSource.ESTIMATE) else None,
                      **kw)


class PoseContractTests(unittest.TestCase):
    def test_explicit_static_transform_chain_and_covariance(self):
        sample = pose(covariance_6x6=tuple(tuple(1.0 if i == j else 0.0 for j in range(6)) for i in range(6)))
        rotate_z_90 = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
        first = TransformSample("base_link", "camera_optical", (0.0, 1.0, 0.0), rotate_z_90,
                                0, "static-calibration", True)
        second = TransformSample("map", "base_link", (2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                                 0, "static-map", True)
        out = transform_pose_chain(sample, (first, second), "map")
        self.assertEqual(out.frame_id, "map")
        self.assertAlmostEqual(out.position_m[0], 2.0)
        self.assertAlmostEqual(out.position_m[1], 2.0)
        self.assertEqual(len(out.covariance_6x6), 6)
        self.assertAlmostEqual(out.covariance_6x6[0][0], 2.0)

    def test_dynamic_transform_requires_same_clock_and_exact_acquisition_time(self):
        dynamic = TransformSample("base_link", "camera_optical", (0, 0, 0), (0, 0, 0, 1),
                                  10, "tf-buffer", False, TimeDomain.ROS)
        self.assertEqual(transform_pose_chain(pose(), (dynamic,), "base_link").frame_id, "base_link")
        with self.assertRaisesRegex(ContractError, "TRANSFORM_TIME_MISMATCH"):
            transform_pose_chain(pose(stamp=11), (dynamic,), "base_link")
        other_clock = TransformSample("base_link", "camera_optical", (0, 0, 0), (0, 0, 0, 1),
                                      10, "sim-tf", False, TimeDomain.SIMULATION)
        with self.assertRaisesRegex(ContractError, "TRANSFORM_CLOCK_MISMATCH"):
            transform_pose_chain(pose(), (other_clock,), "base_link")

    def test_generation_and_sequence_are_paired(self):
        with self.assertRaisesRegex(ContractError, "INCOMPLETE_EPOCH"):
            pose(generation="g1")

    def test_sensor_measurement_requires_calibration_and_ground_truth_model(self):
        with self.assertRaisesRegex(ContractError, "CALIBRATION_MISSING"):
            PoseSample("bottle", "camera", (0, 0, 0), (0, 0, 0, 1), 0,
                       PoseSource.MEASUREMENT, TimeDomain.SENSOR)
        with self.assertRaisesRegex(ContractError, "GROUND_TRUTH_MODEL_MISSING"):
            PoseSample("bottle", "world", (0, 0, 0), (0, 0, 0, 1), 0,
                       PoseSource.SIM_GROUND_TRUTH, TimeDomain.SIMULATION)

    def test_freshness_uses_local_monotonic_receipt_time(self):
        p = pose(received_steady_ns=100, stamp=1, domain=TimeDomain.SIMULATION,
                 source=PoseSource.SIM_GROUND_TRUTH, model_identity="x2@sha")
        p.require_fresh(110, 10)
        with self.assertRaisesRegex(ContractError, "STALE_POSE"):
            p.require_fresh(111, 10)


class DockingHandoffTests(unittest.TestCase):
    def setUp(self):
        self.base = pose(obj="base_link", frame="odom", xyz=(1, 2, 0), stamp=30,
                         domain=TimeDomain.ROS, source=PoseSource.MEASUREMENT,
                         generation="nav-g1", sequence=4, received_steady_ns=1000)
        self.goal = pose(obj="dock_target", frame="odom", xyz=(1, 2, 0), stamp=30,
                         domain=TimeDomain.ROS, source=PoseSource.TASK_TARGET)
        self.handoff = DockingHandoff("dock-7", DockStatus.SUCCEEDED, self.base, self.goal,
                                      (0, 0, 0), (0, 0, 0), 0.5, "nav-g1", 4)
        self.limits = DockingLimits(100, 0.02, 0.03, 0.01, 0.02, 0.4)

    def test_valid_measured_dock_handoff(self):
        result = evaluate_docking_handoff(self.handoff, self.limits, now_steady_ns=1050)
        self.assertTrue(result.ready, result.reasons)

    def test_rejects_running_stale_mismatched_epoch_and_motion(self):
        bad = DockingHandoff("dock-7", DockStatus.RUNNING, self.base, self.goal,
                             (0.2, 0, 0), (0, 0, 0), 0.0, "nav-g2", 5)
        result = evaluate_docking_handoff(bad, self.limits, now_steady_ns=1200)
        self.assertFalse(result.ready)
        self.assertIn("NAVIGATION_NOT_SUCCEEDED", result.reasons)
        self.assertIn("BASE_POSE_STALE_OR_UNTIMED", result.reasons)
        self.assertIn("HANDOFF_EPOCH_MISMATCH", result.reasons)
        self.assertIn("BASE_NOT_STATIONARY", result.reasons)


if __name__ == "__main__":
    unittest.main()
