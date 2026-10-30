from __future__ import annotations

import unittest

import numpy as np

from scripts.research.issue46.perception.rgbd_grasp_contract import (
    CameraIntrinsics,
    ParallelJawCandidate,
    PerceptionContractError,
    filter_omnipicker_side_grasps,
    point_cloud_from_depth,
    transform_candidate_to_base,
)


class RgbdGraspContractTests(unittest.TestCase):
    def test_masked_metric_depth_deprojection(self) -> None:
        intrinsics = CameraIntrinsics(width=2, height=2, fx=2.0, fy=2.0, cx=0.0, cy=0.0)
        depth = np.array(((1.0, 2.0), (np.nan, 5.0)), dtype=np.float32)
        points = point_cloud_from_depth(depth, intrinsics, mask=np.array(((1, 0), (1, 1))), max_depth_m=4.0)
        self.assertEqual(points.shape, (1, 3))
        np.testing.assert_allclose(points[0], [0.0, 0.0, 1.0])

    def test_camera_to_base_and_side_grasp_filter(self) -> None:
        candidate = ParallelJawCandidate(
            score=0.9,
            width_m=0.07,
            depth_m=0.02,
            rotation_camera=np.eye(3),
            translation_camera_m=np.array((0.1, 0.2, 0.3)),
        )
        transformed = transform_candidate_to_base(candidate, np.eye(4))
        np.testing.assert_allclose(transformed.tip_camera_m, [0.12, 0.2, 0.3])
        accepted = filter_omnipicker_side_grasps(
            [transformed], preferred_approach_base=np.array((1.0, 0.0, 0.0)),
            max_opening_m=0.09, approach_tolerance_rad=0.1,
        )
        self.assertEqual(accepted, [transformed])

    def test_rejects_width_outside_omnipicker_envelope(self) -> None:
        candidate = ParallelJawCandidate(
            score=0.9, width_m=0.12, depth_m=0.02,
            rotation_camera=np.eye(3), translation_camera_m=np.zeros(3),
        )
        self.assertEqual(
            filter_omnipicker_side_grasps(
                [candidate], preferred_approach_base=np.array((1.0, 0.0, 0.0)),
                max_opening_m=0.09, approach_tolerance_rad=0.1,
            ),
            [],
        )

    def test_rejects_depth_shape_mismatch(self) -> None:
        with self.assertRaises(PerceptionContractError):
            point_cloud_from_depth(np.ones((1, 1)), CameraIntrinsics(2, 2, 1.0, 1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
