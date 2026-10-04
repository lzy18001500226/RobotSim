import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest

from simulation.mujoco.m0_pick_place import _mesh_provenance


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "scripts/run_m0_pick_place.sh"
PINNED_HUMANOID = "3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12"
PINNED_UNITREE = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"
PINNED_DEX3 = "5994d4faef0a9cadd3287f8de0199a67eeb2a259"


class RuntimeMeshProvenanceTests(unittest.TestCase):
    def test_only_resolved_paths_inside_verified_mesh_tree_are_accepted(self):
        with tempfile.TemporaryDirectory(prefix="robotsim-mesh-provenance-") as temp:
            root = Path(temp)
            unitree = root / "unitree"
            mesh_root = unitree / "unitree_robots/g1/meshes"
            mesh_root.mkdir(parents=True)
            external = root / "external-meshes"
            external.mkdir()

            accepted = _mesh_provenance(unitree, mesh_root)
            self.assertEqual(accepted["path_class"], "pinned_unitree_g1_mesh_tree")
            self.assertEqual(accepted["path_relative_to_unitree"], "unitree_robots/g1/meshes")

            rejected = _mesh_provenance(unitree, external)
            self.assertEqual(rejected["path_class"], "external_rejected")

            mesh_root.rmdir()
            mesh_root.symlink_to(external, target_is_directory=True)
            rejected_symlink = _mesh_provenance(unitree, mesh_root)
            self.assertEqual(rejected_symlink["path_class"], "external_rejected")


class LauncherPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="robotsim-m0-launcher-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "checkout"
        self.launcher = self.repo / "scripts/run_m0_pick_place.sh"
        self.launcher.parent.mkdir(parents=True)
        shutil.copy2(LAUNCHER, self.launcher)
        self.launcher.chmod(self.launcher.stat().st_mode | stat.S_IXUSR)
        requirements = self.repo / "simulation/mujoco/requirements-m0.txt"
        requirements.parent.mkdir(parents=True)
        requirements.write_text("# launcher test fixture\n", encoding="utf-8")

        self.candidate = self.base / "humanoid_vla"
        self.unitree = self.base / "unitree_mujoco"
        self.dex3 = self.base / "unitree_ros_dex3"
        (self.candidate / ".git").mkdir(parents=True)
        meshes = self.unitree / "unitree_robots/g1/meshes"
        meshes.mkdir(parents=True)
        (self.unitree / ".git").mkdir()
        (self.dex3 / ".git").mkdir(parents=True)
        dex3_model = self.dex3 / "robots/g1_description/g1_29dof_with_hand_rev_1_0.xml"
        dex3_model.parent.mkdir(parents=True)
        dex3_model.write_text("<mujoco/>", encoding="utf-8")
        (dex3_model.parent / "meshes").mkdir()
        self.external_mesh = self.base / "external-meshes"
        self.external_mesh.mkdir()
        self.run_root = self.base / "run-cache"
        self.output_root = self.base / "outputs"

        fake_bin = self.base / "bin"
        fake_bin.mkdir()
        self.git = fake_bin / "git"
        self.git.write_text(
            """#!/usr/bin/env bash
set -eu
[[ \"$1\" == -C ]] || exit 90
repo=\"$2\"
shift 2
operation=\"$*\"
case \"$operation\" in
  \"rev-parse HEAD\")
    if [[ \"$repo\" == \"$TEST_ROOT\" ]]; then printf '%s\\n' \"${TEST_ROBOTSIM_SHA:-test-robotsim-sha}\"
    elif [[ \"$repo\" == \"$TEST_CANDIDATE\" ]]; then printf '%s\\n' \"${TEST_HUMANOID_SHA:-3d4bf2f040d6cb9f867becf1dc1b97b9dc3bef12}\"
    elif [[ \"$repo\" == \"$TEST_UNITREE\" ]]; then printf '%s\\n' \"${TEST_UNITREE_SHA:-1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d}\"
    elif [[ \"$repo\" == \"$TEST_DEX3\" ]]; then printf '%s\\n' \"${TEST_DEX3_SHA:-5994d4faef0a9cadd3287f8de0199a67eeb2a259}\"
    else exit 91; fi
    ;;
  \"status --porcelain --untracked-files=all\")
    if [[ \"$repo\" == \"$TEST_ROOT\" ]]; then printf '%s' \"${TEST_ROBOTSIM_STATUS:-}\"
    elif [[ \"$repo\" == \"$TEST_CANDIDATE\" ]]; then printf '%s' \"${TEST_HUMANOID_STATUS:-}\"
    elif [[ \"$repo\" == \"$TEST_UNITREE\" ]]; then printf '%s' \"${TEST_UNITREE_STATUS:-}\"
    elif [[ \"$repo\" == \"$TEST_DEX3\" ]]; then printf '%s' \"${TEST_DEX3_STATUS:-}\"
    else exit 92; fi
    ;;
  *) exit 93 ;;
esac
""",
            encoding="utf-8",
        )
        self.git.chmod(0o755)
        uv = fake_bin / "uv"
        uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        uv.chmod(0o755)

        self.env = os.environ.copy()
        self.env.update(
            {
                "PATH": str(fake_bin) + os.pathsep + self.env.get("PATH", ""),
                "TEST_ROOT": str(self.repo),
                "TEST_CANDIDATE": str(self.candidate),
                "TEST_UNITREE": str(self.unitree),
                "TEST_DEX3": str(self.dex3),
                "ROBOTSIM_M0_RUN_DIR": str(self.run_root),
                "ROBOTSIM_M0_OUTPUT_DIR": str(self.output_root),
                "ROBOTSIM_M0_CANDIDATE_DIR": str(self.candidate),
                "ROBOTSIM_M0_UNITREE_DIR": str(self.unitree),
                "ROBOTSIM_M0_DEX3_DIR": str(self.dex3),
            }
        )

    def invoke(self, extra_env=None):
        env = self.env.copy()
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(self.launcher)],
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    def current_result(self):
        records = list(self.output_root.glob("m0-*/m0_result.json"))
        self.assertEqual(len(records), 1)
        return json.loads(records[0].read_text(encoding="utf-8"))

    def assert_failure_identity(self, result):
        self.assertEqual(result["state"], "FAILED")
        self.assertFalse(result["passed"])
        self.assertFalse(result["complete"])
        self.assertTrue(result["run_id"].startswith("m0-"))
        for key in (
            "robotsim_sha",
            "robotsim_dirty",
            "python_version",
            "mujoco_version",
            "numpy_version",
            "h5py_version",
            "opencv_version",
            "upstream_shas",
            "mesh_provenance",
            "controller_parameters",
            "seed",
            "acceptance_thresholds",
            "output_directory",
        ):
            self.assertIn(key, result)

    def test_launcher_is_executable(self):
        self.assertTrue(LAUNCHER.stat().st_mode & stat.S_IXUSR)

    def test_wrong_pinned_vendor_revision_is_durable_failure(self):
        completed = self.invoke({"TEST_UNITREE_SHA": "wrong-unitree-revision"})
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = self.current_result()
        self.assert_failure_identity(result)
        self.assertEqual(result["upstream_shas"]["unitree_mujoco"], "wrong-unitree-revision")

    def test_dirty_vendor_checkout_is_durable_failure(self):
        completed = self.invoke({"TEST_UNITREE_STATUS": " M tracked-file"})
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = self.current_result()
        self.assert_failure_identity(result)
        self.assertEqual(result["upstream_shas"]["unitree_mujoco"], PINNED_UNITREE)
        self.assertTrue(result["upstream_dirty"]["unitree_mujoco"])

    def test_wrong_dex3_revision_is_durable_failure(self):
        completed = self.invoke({"TEST_DEX3_SHA": "wrong-dex3-revision"})
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = self.current_result()
        self.assert_failure_identity(result)
        self.assertEqual(result["upstream_shas"]["unitree_ros_dex3"], "wrong-dex3-revision")

    def test_dirty_dex3_checkout_is_durable_failure(self):
        completed = self.invoke({"TEST_DEX3_STATUS": " M tracked-file"})
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = self.current_result()
        self.assert_failure_identity(result)
        self.assertEqual(result["upstream_shas"]["unitree_ros_dex3"], PINNED_DEX3)
        self.assertTrue(result["upstream_dirty"]["unitree_ros_dex3"])

    def test_external_mesh_override_is_rejected_and_recorded(self):
        completed = self.invoke({"ROBOTSIM_M0_MESH_DIR": str(self.external_mesh)})
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = self.current_result()
        self.assert_failure_identity(result)
        self.assertEqual(result["mesh_provenance"]["path_class"], "external_rejected")
        self.assertEqual(
            result["mesh_provenance"]["path_relative_to_unitree"],
            "outside_verified_unitree_g1_mesh_tree",
        )

    def test_nonpositive_transfer_frames_are_rejected_and_recorded(self):
        completed = self.invoke({"ROBOTSIM_M0_TRANSFER_FRAMES": "0"})
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = self.current_result()
        self.assert_failure_identity(result)
        self.assertIsNone(result["controller_parameters"]["transfer_frames"])
        self.assertIn("positive integer", completed.stderr)

    def test_nonpositive_lower_frames_are_rejected_and_recorded(self):
        completed = self.invoke({"ROBOTSIM_M0_LOWER_FRAMES": "0"})
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = self.current_result()
        self.assert_failure_identity(result)
        self.assertIsNone(result["controller_parameters"]["lower_frames"])
        self.assertIn("positive integer", completed.stderr)

    def test_mesh_root_symlink_cannot_escape_pinned_unitree_tree(self):
        mesh_root = self.unitree / "unitree_robots/g1/meshes"
        mesh_root.rmdir()
        mesh_root.symlink_to(self.external_mesh, target_is_directory=True)
        completed = self.invoke()
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = self.current_result()
        self.assert_failure_identity(result)
        self.assertEqual(result["mesh_provenance"]["path_class"], "external_rejected")

    def test_existing_incompatible_python_venv_fails_clearly(self):
        python = self.run_root / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text(
            "#!/usr/bin/env bash\nprintf '3.11.9\\n'\n", encoding="utf-8"
        )
        python.chmod(0o755)
        completed = self.invoke()
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Python 3.10.x", completed.stderr)
        result = self.current_result()
        self.assert_failure_identity(result)
        self.assertEqual(result["python_version"], "3.11.9")

    def test_failed_invocation_cannot_reuse_previous_pass_record(self):
        self.output_root.mkdir(parents=True)
        stale = self.output_root / "m0_result.json"
        stale.write_text('{"passed": true, "state": "COMPLETE"}\n', encoding="utf-8")
        completed = self.invoke({"TEST_UNITREE_SHA": "wrong-unitree-revision"})
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = self.current_result()
        self.assert_failure_identity(result)
        self.assertNotEqual(result["run_id"], "legacy")
        self.assertEqual(result["controller_parameters"]["hand_roll_deg"], 20)
        self.assertEqual(result["controller_parameters"]["grasp_hold_frames"], 2)
        self.assertEqual(result["controller_parameters"]["lift_frames"], 15)
        self.assertEqual(result["controller_parameters"]["commanded_lift_height_m"], 0.16)
        self.assertEqual(result["controller_parameters"]["transfer_frames"], 30)
        self.assertEqual(result["controller_parameters"]["lower_frames"], 15)
        self.assertTrue(json.loads(stale.read_text(encoding="utf-8"))["passed"])

    def test_failure_json_escapes_all_allowed_control_bytes_in_paths(self):
        control_output = self.base / ("outputs-" + "\x01" + "\x0b" + "\x1f")
        completed = self.invoke(
            {
                "ROBOTSIM_M0_OUTPUT_DIR": str(control_output),
                "TEST_UNITREE_SHA": "wrong-unitree-revision",
            }
        )
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        records = list(control_output.glob("m0-*/m0_result.json"))
        self.assertEqual(len(records), 1)
        result = json.loads(records[0].read_text(encoding="utf-8"))
        self.assert_failure_identity(result)
        self.assertEqual(result["output_directory"], str(records[0].parent))


if __name__ == "__main__":
    unittest.main()
