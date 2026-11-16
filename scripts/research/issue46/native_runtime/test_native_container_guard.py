"""Actual container classification with read-only Docker responses."""
import copy
import json
import unittest
from unittest import mock
from .native_container_guard import require_exclusive_x2_runtime


class NativeContainerGuardTests(unittest.TestCase):
    def record(self):
        return dict(Name="/mosim-sim-1",Config=dict(Image="mosim/ros1-dev:pin",
                    Labels={"com.docker.compose.project":"mosim"},Env=["ROS_DISTRO=noetic"]),
                    HostConfig=dict(NetworkMode="mosim_default",Privileged=False),
                    Mounts=[dict(Source="/mnt/host/d/MoSim",Destination="/workspace/MoSim")])

    def check(self,records):
        with mock.patch("subprocess.check_output",side_effect=["abc",json.dumps(records)]):
            return require_exclusive_x2_runtime()

    def test_no_containers(self):
        with mock.patch("subprocess.check_output",return_value=""):
            self.assertEqual(require_exclusive_x2_runtime(),[])

    def test_existing_separate_mosim_bridge_is_preserved(self):
        self.assertEqual(self.check([self.record()])[0]["action"],"UNTOUCHED")

    def test_network_disabled_mosim_rebuild_is_preserved(self):
        r=self.record();r['Name']='/mosim-rebuild-check'
        r['HostConfig']['NetworkMode']='none'
        self.assertEqual(self.check([r])[0]['action'],'UNTOUCHED')

    def test_noetic_image_and_read_only_wslg_socket_are_preserved(self):
        r=self.record();r["Config"]["Image"]="mosim:noetic-20261116"
        r["Mounts"].append(dict(Source="/run/desktop/mnt/host/wsl/socket",Destination="/tmp/.X11-unix",RW=False))
        r["Mounts"].append(dict(Source="/usr/lib/wsl",Destination="/usr/lib/wsl",RW=False))
        self.assertEqual(self.check([r])[0]["action"],"UNTOUCHED")
        r["Mounts"][-1]["RW"]=True
        with self.assertRaises(RuntimeError):self.check([r])
        r["Mounts"][-1]["RW"]=False;r["Config"]["Env"]=["ROS_DISTRO=humble"]
        with self.assertRaises(RuntimeError):self.check([r])

    def test_unknown_or_x2_container_is_rejected(self):
        for key in ("Image","Labels"):
            r=self.record();r["Config"][key]="robotsim:runtime" if key=="Image" else {}
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.check([r])

    def test_host_or_privileged_shared_runtime_is_rejected(self):
        for key,value in (("NetworkMode","host"),("Privileged",True),
                          ("PidMode","host"),("IpcMode","host")):
            r=self.record();r["HostConfig"][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.check([r])

    def test_robot_assets_or_unknown_mounts_are_rejected(self):
        for mounts in ([],[dict(Source="/home/RobotSim",Destination="/workspace/MoSim")],
                       [dict(Source="/tmp/other",Destination="/workspace/evidence")]):
            r=self.record();r["Mounts"]=mounts
            with self.subTest(mounts=mounts),self.assertRaises(RuntimeError):self.check([r])

    def test_one_x2_among_unrelated_containers_still_blocks(self):
        r=self.record();r["Name"]="/x2-native";r["Config"]["Image"]="robotsim:humble"
        with self.assertRaisesRegex(RuntimeError,"x2-native"):self.check([self.record(),r])


if __name__=="__main__":unittest.main()
