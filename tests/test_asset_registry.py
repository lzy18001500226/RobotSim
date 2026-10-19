from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "third_party" / "asset_registry.json"
LOCK_PATH = ROOT / "third_party" / "LOCK.md"


class UpstreamAssetRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        cls.lock_text = LOCK_PATH.read_text(encoding="utf-8")

    def test_registry_schema_and_repository_ids_are_valid(self) -> None:
        self.assertEqual(1, self.registry["schema_version"])
        self.assertRegex(self.registry["verified_at"], r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}$")
        repositories = self.registry["repositories"]
        self.assertTrue(repositories)
        repo_ids = [repo["id"] for repo in repositories]
        self.assertEqual(len(repo_ids), len(set(repo_ids)))

    def test_verified_repositories_have_immutable_identity_and_license_state(self) -> None:
        for repository in self.registry["repositories"]:
            with self.subTest(repository=repository["id"]):
                self.assertEqual("verified", repository["status"])
                self.assertRegex(repository["repository_url"], r"^https://github\.com/")
                self.assertRegex(repository["commit_sha"], r"^[0-9a-f]{40}$")
                self.assertTrue(repository["commit_url"].endswith(repository["commit_sha"]))
                license_record = repository["license"]
                self.assertEqual("verified", license_record["status"])
                self.assertTrue(license_record["spdx"])
                self.assertTrue(license_record["path"])
                self.assertIn(repository["commit_sha"], license_record["url"])

    def test_canonical_checkout_paths_follow_owner_and_repository(self) -> None:
        for repository in self.registry["repositories"]:
            with self.subTest(repository=repository["id"]):
                parsed = urlparse(repository["repository_url"])
                owner, name = parsed.path.strip("/").removesuffix(".git").split("/", 1)
                self.assertEqual(
                    f"reference/{owner}/{name}",
                    repository["canonical_reference_path"],
                )

        for asset in self.registry["assets"]:
            if asset["status"] == "verified":
                with self.subTest(asset=asset["id"]):
                    self.assertFalse(asset["path"].startswith("reference/"))

    def test_existing_lock_dependencies_match_the_lockfile(self) -> None:
        for repository in self.registry["repositories"]:
            lock_reference = repository.get("lock_reference")
            if lock_reference is None:
                continue
            with self.subTest(repository=repository["id"]):
                self.assertEqual("third_party/LOCK.md", lock_reference)
                self.assertIn(repository["commit_sha"], self.lock_text)
                self.assertIn(repository["repository_url"], self.lock_text)

    def test_verified_asset_paths_and_hashes_are_well_formed(self) -> None:
        repository_ids = {repo["id"] for repo in self.registry["repositories"]}
        assets = self.registry["assets"]
        asset_ids = [asset["id"] for asset in assets]
        self.assertEqual(len(asset_ids), len(set(asset_ids)))

        for asset in assets:
            with self.subTest(asset=asset["id"]):
                if asset["status"] == "unknown":
                    self.assertIsNone(asset["repository_id"])
                    self.assertIsNone(asset["path"])
                    self.assertEqual("unknown", asset["license_status"])
                    self.assertNotIn("sha256", asset)
                    continue

                self.assertEqual("verified", asset["status"])
                self.assertIn(asset["repository_id"], repository_ids)
                path = asset["path"]
                self.assertIsInstance(path, str)
                self.assertFalse(path.startswith("/"))
                self.assertNotIn("..", Path(path).parts)
                if "sha256" in asset:
                    self.assertIn(asset["asset_type"], {"urdf", "mjcf"})
                    self.assertRegex(asset["sha256"], r"^[0-9a-f]{64}$")

    def test_required_robot_and_gripper_sources_are_present(self) -> None:
        assets = {asset["id"] for asset in self.registry["assets"]}
        repositories = {repo["id"] for repo in self.registry["repositories"]}
        self.assertTrue(
            {
                "agibot_x2_ultra_v1_4_0",
                "agibot_x2_omnihand_v1_4_0",
                "agibot_x2_omnipicker_v1_4_0",
                "unitree_g1_with_hand_rev_1_0_mjcf",
                "unitree_dex3_1_left_hand_urdf",
                "unitree_dex3_1_right_hand_urdf",
                "unitree_mujoco_g1_29dof_mjcf",
            }.issubset(assets)
        )
        self.assertTrue(
            {"unitree_sdk2", "unitree_ros2", "unitree_mujoco", "mujoco"}.issubset(
                repositories
            )
        )

    def test_x2_omnipicker_urdf_is_verified_but_step_source_remains_unknown(self) -> None:
        assets = {asset["id"]: asset for asset in self.registry["assets"]}
        urdf = assets["agibot_x2_omnipicker_v1_4_0"]
        self.assertEqual("verified", urdf["status"])
        self.assertEqual("X2_URDF-v1.4.0/X2-Ultra_omnipicker.urdf", urdf["path"])
        step = assets["x2_omnipicker_step_source"]
        self.assertEqual("unknown", step["status"])
        self.assertIsNone(step["repository_id"])
        self.assertIsNone(step["path"])
        self.assertEqual("unknown", step["license_status"])
        self.assertNotIn("sha256", step)


if __name__ == "__main__":
    unittest.main()
