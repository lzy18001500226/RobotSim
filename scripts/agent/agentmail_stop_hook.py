#!/usr/bin/env python3
"""Load local AgentMail credentials from Windows DPAPI for the RobotSim Stop hook."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping

try:
    from . import notify_task
except ImportError:  # pragma: no cover - direct script execution
    import notify_task


ROOT = Path(__file__).resolve().parents[2]
CREDENTIAL_SCRIPT = ROOT / "scripts" / "agent" / "agentmail_local_credential.ps1"
NOTIFIER = ROOT / "scripts" / "agent" / "notify_task.py"
MAIL_SETTINGS = ("AGENTMAIL_API_KEY", "AGENTMAIL_INBOX_ID", "ROBOTSIM_NOTIFY_TO")
MAX_BUNDLE_BYTES = 8192
PATH_CONVERSION_TIMEOUT_SECONDS = 2
CREDENTIAL_BRIDGE_TIMEOUT_SECONDS = 3
NOTIFIER_TIMEOUT_SECONDS = 4


class CredentialBridgeError(Exception):
    """A sanitized credential bridge failure."""


def _windows_script_path() -> str:
    converter = shutil.which("wslpath")
    if converter is None:
        raise CredentialBridgeError("WSL path conversion is unavailable")
    try:
        result = subprocess.run(
            [converter, "-w", str(CREDENTIAL_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=PATH_CONVERSION_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CredentialBridgeError("WSL path conversion failed") from exc
    path = result.stdout.strip()
    if result.returncode != 0 or not path:
        raise CredentialBridgeError("WSL path conversion failed")
    return path


def _read_bundle(action: str = "Emit") -> dict[str, str]:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        raise CredentialBridgeError("Windows PowerShell is unavailable")
    script_path = _windows_script_path()
    command = [
        powershell,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        script_path,
        "-Action",
        action,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=CREDENTIAL_BRIDGE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CredentialBridgeError("Windows credential bridge failed") from exc
    if result.returncode != 0:
        raise CredentialBridgeError("Windows credential bridge failed")

    lines = result.stdout.splitlines()
    if len(lines) != 1 or len(lines[0]) > MAX_BUNDLE_BYTES:
        raise CredentialBridgeError("Windows credential bridge returned invalid data")
    if re.fullmatch(rb"[A-Za-z0-9+/]+={0,2}", lines[0]) is None:
        raise CredentialBridgeError("Windows credential bridge returned invalid data")
    try:
        decoded = base64.b64decode(lines[0], validate=True)
        bundle = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CredentialBridgeError("Windows credential bridge returned invalid data") from exc
    if not isinstance(bundle, dict) or set(bundle) != set(MAIL_SETTINGS):
        raise CredentialBridgeError("Windows credential bridge returned invalid data")
    if any(not isinstance(bundle[key], str) or not bundle[key] for key in MAIL_SETTINGS):
        raise CredentialBridgeError("Windows credential bridge returned incomplete settings")
    if len(bundle["AGENTMAIL_API_KEY"]) > 4096:
        raise CredentialBridgeError("Windows credential bridge returned invalid data")
    if any(len(bundle[key]) > 512 for key in MAIL_SETTINGS[1:]):
        raise CredentialBridgeError("Windows credential bridge returned invalid data")
    return bundle


def _read_hook_payload() -> bytes:
    stream = getattr(sys.stdin, "buffer", sys.stdin)
    raw = stream.read(1_048_577)
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="replace")
    return raw if len(raw) <= 1_048_576 else b""


def _has_valid_terminal_closeout(raw: bytes) -> bool:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(payload, dict)
        or payload.get("hook_event_name") != "Stop"
        or payload.get("stop_hook_active") is True
    ):
        return False
    present, event, problem = notify_task.extract_task_closeout_envelope(
        payload.get("last_assistant_message")
    )
    if not present or problem:
        return False
    try:
        if isinstance(event, dict) and event.get("schema_version") == notify_task.RESEARCH_SCHEMA_VERSION:
            notify_task.normalize_research_event(event)
        else:
            notify_task.normalize_task_closeout(event)
    except (ValueError, TypeError):
        return False
    return True


def _run_notifier(
    environ: Mapping[str, str],
    input_data: bytes,
    *,
    dry_run: bool = False,
) -> int:
    command = [sys.executable, str(NOTIFIER), "stop"]
    if dry_run:
        command.append("--dry-run")
    try:
        subprocess.run(
            command,
            env=dict(environ),
            input=input_data,
            timeout=NOTIFIER_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        if not dry_run:
            print("RobotSim Stop notification did not finish; turn completion remains unblocked.", file=sys.stderr)
        return 0
    return 0


def _presence_lines(bundle: Mapping[str, str], environ: Mapping[str, str]) -> str:
    lines = [f"{name}={'present' if bundle.get(name) else 'absent'}" for name in MAIL_SETTINGS]
    lines.append(
        "ROBOTSIM_LOCAL_STOP_HOOK="
        + ("enabled" if environ.get("ROBOTSIM_LOCAL_STOP_HOOK", "").strip() == "1" else "disabled")
    )
    lines.append("CONFIG_READY=" + ("yes" if all(bundle.get(name) for name in MAIL_SETTINGS) else "no"))
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true", help="report credential presence only")
    mode.add_argument("--self-test-bridge", action="store_true", help="test WSL and DPAPI with synthetic data")
    mode.add_argument("--dry-run", action="store_true", help="pass a Stop payload to the notifier in dry-run mode")
    return parser


def main(argv: list[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    args = _parser().parse_args(argv)
    env = dict(os.environ if environ is None else environ)

    if args.self_test_bridge:
        try:
            bundle = _read_bundle("SelfTestEmit")
        except CredentialBridgeError:
            print("DPAPI_WSL_BRIDGE=FAIL")
            return 1
        valid = (
            bundle["AGENTMAIL_API_KEY"].startswith("synthetic-agentmail-")
            and bundle["AGENTMAIL_INBOX_ID"] == "synthetic-inbox@example.invalid"
            and bundle["ROBOTSIM_NOTIFY_TO"] == "synthetic-recipient@example.invalid"
        )
        print("DPAPI_WSL_BRIDGE=" + ("PASS" if valid else "FAIL"))
        return 0 if valid else 1

    if args.check_config:
        try:
            bundle = _read_bundle("Emit")
        except CredentialBridgeError:
            print("AGENTMAIL_API_KEY=absent")
            print("AGENTMAIL_INBOX_ID=absent")
            print("ROBOTSIM_NOTIFY_TO=absent")
            print("ROBOTSIM_LOCAL_STOP_HOOK=" + ("enabled" if env.get("ROBOTSIM_LOCAL_STOP_HOOK", "").strip() == "1" else "disabled"))
            print("CONFIG_READY=no")
            return 0
        print(_presence_lines(bundle, env))
        return 0

    input_data = _read_hook_payload()

    if args.dry_run:
        return _run_notifier(env, input_data, dry_run=True)

    if (
        env.get("ROBOTSIM_LOCAL_STOP_HOOK", "").strip() == "1"
        and _has_valid_terminal_closeout(input_data)
    ):
        try:
            bundle = _read_bundle("Emit")
        except CredentialBridgeError:
            for name in MAIL_SETTINGS:
                env.pop(name, None)
            print("AgentMail credentials unavailable; email delivery skipped.", file=sys.stderr)
        else:
            env.update(bundle)

    return _run_notifier(env, input_data)


if __name__ == "__main__":
    raise SystemExit(main())
