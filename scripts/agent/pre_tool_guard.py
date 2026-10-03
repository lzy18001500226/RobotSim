#!/usr/bin/env python3
"""Block a narrow set of clearly destructive or credential-leaking tool calls."""

from __future__ import annotations

import json
import re
import sys
from typing import Any


DESTRUCTIVE_PATTERNS = (
    re.compile(r"(?:^|[\s;&|])git(?:\s+-C\s+\S+)?\s+push\b[^;&|\n]*(?:--force(?:-with-lease)?\b|(?:^|\s)-f(?:\s|$))", re.I),
    re.compile(r"(?:^|[\s;&|])git(?:\s+-C\s+\S+)?\s+reset\b[^;&|\n]*--hard\b", re.I),
    re.compile(r"(?:^|[\s;&|])git(?:\s+-C\s+\S+)?\s+clean\b[^;&|\n]*(?:--force\b|(?:^|\s)-[a-z]*f[a-z]*(?:\s|$))", re.I),
)
SECRET_VARIABLE = re.compile(
    r"\$(?:\{)?(?:AGENTMAIL_API_KEY|GH_TOKEN|GITHUB_TOKEN|AWS_SECRET_ACCESS_KEY)(?:\})?"
    r"|\$env:(?:AGENTMAIL_API_KEY|GH_TOKEN|GITHUB_TOKEN)",
    re.I,
)
WRITE_SINK = re.compile(r"(?:>|\btee\b|\bset-content\b|\bout-file\b)", re.I)
TOKEN_LITERAL = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"github_pat_[A-Za-z0-9_]{30,}|"
    r"(?:gh[pousr])_[A-Za-z0-9]{30,}|"
    r"AKIA[0-9A-Z]{16}"
)
SENSITIVE_BASENAME = re.compile(
    r"^(?:\.env(?:\.(?!example$|sample$|template$)[A-Za-z0-9._-]+)?|"
    r"\.npmrc|\.pypirc|\.credentials?(?:[._-][A-Za-z0-9._-]+)?|credentials?(?:[._-][A-Za-z0-9._-]+)?|"
    r"id_(?:rsa|ed25519)|[^/\\]+\.pem)$",
    re.I,
)
SAFE_ENV_EXAMPLES = {".env.example", ".env.sample", ".env.template"}


def _tool_input(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("tool_input")
    return value if isinstance(value, dict) else {}


def _is_sensitive_path(value: str) -> bool:
    basename = value.replace("\\", "/").rstrip("/").split("/")[-1]
    if basename.lower() in SAFE_ENV_EXAMPLES:
        return False
    return bool(SENSITIVE_BASENAME.fullmatch(basename))


def _reason(payload: dict[str, Any]) -> str | None:
    tool_name = str(payload.get("tool_name", ""))
    tool_input = _tool_input(payload)
    if tool_name in {"Bash", "exec_command"}:
        command = str(tool_input.get("command", tool_input.get("cmd", ""))).replace("\\\n", " ")
        if any(pattern.search(command) for pattern in DESTRUCTIVE_PATTERNS):
            return "blocked a destructive Git command; inspect the target and use the normal review flow"
        if TOKEN_LITERAL.search(command):
            return "blocked a command containing a credential-shaped literal; keep credentials out of commands and files"
        if SECRET_VARIABLE.search(command) and WRITE_SINK.search(command):
            return "blocked writing a secret environment variable to a file or terminal pipeline"
        staged = re.search(r"(?:^|[;&|])\s*git(?:\s+-C\s+\S+)?\s+add\b([^;&|\n]*)", command, re.I)
        if staged and any(_is_sensitive_path(item.strip("\"'")) for item in re.findall(r"[^\s]+", staged.group(1))):
            return "blocked staging an obvious credential file; review it locally and keep credentials out of Git"

    if tool_name in {"Write", "Edit", "apply_patch"}:
        paths = [tool_input.get(key) for key in ("file_path", "path", "filename")]
        if any(isinstance(path, str) and _is_sensitive_path(path) for path in paths):
            return "blocked writing an obvious credential file into the project"
        patch = str(tool_input.get("patch", ""))
        added_paths = re.findall(r"(?m)^\*\*\* (?:Add|Update|Delete) File: (.+)$", patch)
        if any(_is_sensitive_path(path.strip()) for path in added_paths):
            return "blocked a patch targeting an obvious credential file"
        content = "\n".join(
            str(tool_input.get(key, "")) for key in ("patch", "content", "new_string", "input")
        )
        if TOKEN_LITERAL.search(content):
            return "blocked a patch containing a credential-shaped literal"
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    if not isinstance(payload, dict) or payload.get("hook_event_name") != "PreToolUse":
        return 0
    reason = _reason(payload)
    if reason:
        json.dump({"decision": "block", "reason": reason}, sys.stdout)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
