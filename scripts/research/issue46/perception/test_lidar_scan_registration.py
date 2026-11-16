"""Synthetic frame/sign checks only, not real LiDAR localization validation."""
import unittest
import numpy as np
from scripts.research.issue46.perception.lidar_scan_registration import (
    Pose2,scan_delta,matrix,compose_body_from_scan_delta)

class ScanFrameTests(unittest.TestCase):
    def test_stationary_landmarks_give_positive_robot_translation(self):
        rng=np.random.default_rng(46)
        world=rng.uniform([-2.,-1.],[3.,2.],size=(180,2))
        previous=world
        current=world-[.02,-.01]
        x,y,angle,residual=scan_delta(previous,current,iterations=12)
        np.testing.assert_allclose([x,y],[.02,-.01],atol=1e-6)
        self.assertLess(abs(angle),1e-6)
        observed=compose_body_from_scan_delta(Pose2(1.,2.,0.),Pose2(.1026,0.,0.),Pose2(.1026,0.,0.),Pose2(x,y,angle))
        np.testing.assert_allclose([observed.x,observed.y],[1.02,1.99],atol=1e-6)

    def test_measured_torso_rotation_does_not_fake_base_motion(self):
        body=Pose2(.3,-.4,.2)
        prev=Pose2(.10,0.,0.);cur=Pose2(.10,.01,.05)
        delta=np.linalg.inv(matrix(prev))@matrix(cur)
        d=Pose2(delta[0,2],delta[1,2],np.arctan2(delta[1,0],delta[0,0]))
        result=compose_body_from_scan_delta(body,prev,cur,d)
        np.testing.assert_allclose(matrix(result),matrix(body),atol=1e-12)

    def test_sensor_lever_arm_is_not_dropped_during_rotation(self):
        old=Pose2(0.,0.,0.);new=Pose2(0.,0.,.05);mount=Pose2(.10,0.,0.)
        delta=np.linalg.inv(matrix(mount))@matrix(new)@matrix(mount)
        d=Pose2(delta[0,2],delta[1,2],np.arctan2(delta[1,0],delta[0,0]))
        out=compose_body_from_scan_delta(old,mount,mount,d)
        np.testing.assert_allclose([out.x,out.y,out.yaw],[0.,0.,.05],atol=1e-12)

if __name__=="__main__":unittest.main()
