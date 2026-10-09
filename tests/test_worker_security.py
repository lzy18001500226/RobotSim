"""Focused tests for the coordinator-to-Codex process boundary."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.agent import worker_security


class WorkerSecurityTests(unittest.TestCase):
    def test_codex_profile_denies_private_auth_and_result_mounts(self) -> None:
        writer = worker_security.codex_permission_profile_overrides(read_only=False)
        reviewer = worker_security.codex_permission_profile_overrides(read_only=True)

        self.assertEqual(writer[0], "default_permissions=robotsim_worker")
        self.assertIn('"/root" = "deny"', writer[1])
        self.assertIn('"/var/tmp" = "deny"', writer[1])
        self.assertIn('"/home" = "write"', writer[1])
        self.assertIn("permissions.robotsim_worker.network.enabled=false", writer[2])
        self.assertIn('"/home" = "read"', reviewer[1])
        self.assertIn('"/root" = "deny"', reviewer[1])

    def test_worker_environment_is_allowlisted(self) -> None:
        ambient = {
            "PATH": "/sensitive/bin",
            "LANG": "C.UTF-8",
            "AGENTMAIL_API_KEY": "synthetic-agentmail-secret",
            "GH_TOKEN": "synthetic-gh-secret",
            "GITHUB_TOKEN": "synthetic-github-secret",
            "ROBOTSIM_NOTIFY_TO": "maintainer@example.invalid",
            "AWS_SECRET_ACCESS_KEY": "synthetic-cloud-secret",
            "OPENAI_API_KEY": "synthetic-codex-key",
            "SSH_AUTH_SOCK": "/private/ssh-agent.sock",
            "HTTP_PROXY": "http://proxy.example.invalid",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "credential.helper",
            "GIT_CONFIG_VALUE_0": "secret-helper",
        }

        environment = worker_security.build_worker_environment(ambient=ambient)

        self.assertEqual(environment["LANG"], "C.UTF-8")
        self.assertEqual(environment["HOME"], "/tmp/worker-home")
        self.assertEqual(environment["CODEX_HOME"], "/root")
        self.assertEqual(environment["GIT_CONFIG_GLOBAL"], "/dev/null")
        for name in (
            "AGENTMAIL_API_KEY", "GH_TOKEN", "GITHUB_TOKEN", "ROBOTSIM_NOTIFY_TO",
            "AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY", "SSH_AUTH_SOCK", "HTTP_PROXY",
            "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0",
        ):
            self.assertNotIn(name, environment)
        self.assertNotIn("/sensitive/bin", environment["PATH"])

    def test_only_credential_free_model_proxies_can_enter_codex_environment(self) -> None:
        ambient = {
            "HTTPS_PROXY": "http://proxy.example.invalid:3128",
            "https_proxy": "http://proxy.example.invalid:3128",
            "HTTP_PROXY": "http://proxy.example.invalid:3128",
            "http_proxy": "http://proxy.example.invalid:3128",
            "NO_PROXY": "example.invalid",
            "GH_TOKEN": "synthetic-gh-secret",
        }

        environment = worker_security.build_worker_environment(
            ambient=ambient, include_model_proxy=True,
        )

        for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            self.assertEqual(environment[name], "http://proxy.example.invalid:3128")
        self.assertNotIn("NO_PROXY", environment)
        self.assertNotIn("GH_TOKEN", environment)

        with self.assertRaisesRegex(RuntimeError, "credential-free"):
            worker_security.build_worker_environment(
                ambient={"HTTPS_PROXY": "http://user:secret@proxy.example.invalid:3128"},
                include_model_proxy=True,
            )
        with self.assertRaisesRegex(RuntimeError, "conflicting"):
            worker_security.build_worker_environment(
                ambient={
                    "HTTPS_PROXY": "http://proxy-a.example.invalid:3128",
                    "https_proxy": "http://proxy-b.example.invalid:3128",
                },
                include_model_proxy=True,
            )

    def test_codex_workspace_metadata_mountpoints_are_safe_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            project_config = workspace / ".codex"
            project_config.mkdir()
            config = project_config / "config.toml"
            config.write_text("project_setting = true\n", encoding="utf-8")

            worker_security.prepare_codex_workspace(workspace)

            self.assertTrue((workspace / ".agents").is_dir())
            self.assertEqual(config.read_text(encoding="utf-8"), "project_setting = true\n")

    def test_codex_workspace_metadata_mountpoints_reject_non_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            workspace.mkdir()
            (workspace / ".agents").write_text("not a directory", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "not a directory"):
                worker_security.prepare_codex_workspace(workspace)

    def test_codex_profile_copies_only_auth_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_home = root / "source-codex"
            source_home.mkdir()
            (source_home / "auth.json").write_text('{"token":"synthetic-only"}\n', encoding="utf-8")
            (source_home / "config.toml").write_text("unsafe_config = true\n", encoding="utf-8")
            (source_home / "history.jsonl").write_text("conversation\n", encoding="utf-8")
            state = root / "state"

            target = worker_security.prepare_worker_codex_home(
                state, 62, ambient={"CODEX_HOME": str(source_home)},
            )

            self.assertEqual(sorted(path.name for path in target.iterdir()), ["auth.json"])
            self.assertEqual((target / "auth.json").read_text(encoding="utf-8"), '{"token":"synthetic-only"}\n')
            self.assertEqual(target.stat().st_mode & 0o777, 0o700)
            self.assertEqual((target / "auth.json").stat().st_mode & 0o777, 0o600)

    def test_codex_session_must_be_bound_to_private_profile_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory)
            self.assertFalse(worker_security.worker_profile_owns_session(profile, "legacy-session"))

            worker_security.record_worker_session(profile, "private-session-62")

            self.assertTrue(worker_security.worker_profile_owns_session(profile, "private-session-62"))
            self.assertFalse(worker_security.worker_profile_owns_session(profile, "legacy-session"))
            marker = profile / worker_security.WORKER_SESSION_MARKER
            self.assertEqual(marker.stat().st_mode & 0o777, 0o600)

    def test_codex_session_marker_rejects_unsafe_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory)
            with self.assertRaisesRegex(ValueError, "invalid Codex session"):
                worker_security.record_worker_session(profile, "../outside")
            self.assertFalse(worker_security.worker_profile_owns_session(profile, "../outside"))

    @unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("bwrap"), "bubblewrap unavailable")
    def test_bubblewrap_hides_coordinator_path_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            codex_home = root / "worker-codex"
            input_directory = root / "input"
            output_directory = root / "output"
            for path in (workspace, codex_home, input_directory, output_directory):
                path.mkdir()
            (workspace / "visible.txt").write_text("task workspace\n", encoding="utf-8")
            coordinator_secret = root / "coordinator-secret.txt"
            coordinator_secret.write_text("synthetic coordinator secret\n", encoding="utf-8")
            probe = (
                "import os,pathlib,sys; "
                "assert os.environ.get('AGENTMAIL_API_KEY') is None; "
                "assert pathlib.Path('/home/visible.txt').read_text() == 'task workspace\\n'; "
                "assert not pathlib.Path(sys.argv[1]).exists(); "
                "assert os.environ['HOME'] == '/tmp/worker-home'; "
                "print('isolated')"
            )
            command, environment = worker_security.build_isolated_command(
                [sys.executable, "-c", probe, str(coordinator_secret)],
                workspace=workspace,
                codex_home=codex_home,
                input_directory=input_directory,
                output_directory=output_directory,
                runtime=worker_security.resolve_codex_runtime(sys.executable),
            )
            result = subprocess.run(
                command,
                cwd="/",
                env={**environment, "AGENTMAIL_API_KEY": "synthetic-agentmail-secret"},
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "isolated")

    @unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("bwrap"), "bubblewrap unavailable")
    def test_scoped_bubblewrap_write_roots_block_other_checkout_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            allowed = workspace / "allowed"
            forbidden = workspace / "robots"
            codex_home = root / "worker-codex"
            input_directory = root / "input"
            output_directory = root / "output"
            for path in (allowed, forbidden, codex_home, input_directory, output_directory):
                path.mkdir(parents=True)
            probe = "\n".join((
                "import pathlib",
                "pathlib.Path('/home/allowed/new.txt').write_text('allowed')",
                "try:",
                "    pathlib.Path('/home/robots/source.cpp').write_text('blocked')",
                "except OSError:",
                "    print('scoped')",
                "else:",
                "    raise SystemExit('forbidden workspace path was writable')",
            ))
            command, environment = worker_security.build_isolated_command(
                [sys.executable, "-c", probe],
                workspace=workspace,
                codex_home=codex_home,
                input_directory=input_directory,
                output_directory=output_directory,
                workspace_write_roots=(allowed,),
                runtime=worker_security.resolve_codex_runtime(sys.executable),
            )
            result = subprocess.run(
                command,
                cwd="/",
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "scoped")
            self.assertEqual((allowed / "new.txt").read_text(encoding="utf-8"), "allowed")
            self.assertFalse((forbidden / "source.cpp").exists())


if __name__ == "__main__":
    unittest.main()
