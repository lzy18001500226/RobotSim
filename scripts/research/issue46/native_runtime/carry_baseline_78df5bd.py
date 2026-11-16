"""Opt-in replay of the successful carry strategy; never restores old I/O guards.

RETAIN_RESEARCH_DIAGNOSTIC. Reads immutable local Git objects, not a checkout.
The current grasp, owner feedback, source/model safety checks remain in force.
"""
from __future__ import annotations
import ast
import hashlib
import math
import subprocess
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
COMMIT = "78df5bdc2bbac85364f74377ba8177416ebbe83e"
BASE = "scripts/research/issue46/native_runtime/"
HASHES = {
    "native_base_micro_probe.py": "fdea5380c72c2f3b3ce3177cfb0b7bffb998ef9a0d5a332df02f2487b565c331",
    "native_task_grasp_region.py": "a9cd310de7091d94313d43f3abbb2679a5468b80e6dcdfd4c312f84a03fa5a88",
}

@lru_cache
def source(filename):
    raw = subprocess.check_output(
        ["git", "show", COMMIT + ":" + BASE + filename], cwd=REPO, timeout=5)
    if hashlib.sha256(raw).hexdigest() != HASHES[filename]:
        raise ValueError("pinned carry baseline source mismatch")
    return raw.decode()

def function_source(filename, name):
    text = source(filename)
    matches = [n for n in ast.parse(text).body
               if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError("pinned carry function absent or ambiguous")
    return ast.get_source_segment(text, matches[0])

def task_route_profile(profile, *, fast_task_timing=False):
    if not fast_task_timing:
        raise ValueError("78df5bd replay requires its full START fast-task profile")
    namespace = {"math": math}
    exec(compile(function_source("native_task_grasp_region.py",
                                "task_route_profile"), COMMIT, "exec"), namespace)
    result = namespace["task_route_profile"](profile, fast_task_timing=True)
    # Keep the maintainer-approved diagnostic speed policy, not a model limit.
    result["planar_speed_warning_only"] = True
    result["carry_baseline"] = provenance()
    return result

def provenance():
    return {
        "commit": COMMIT, "classification": "PINNED_TRANSPORT_STRATEGY_CURRENT_IO_GUARDS",
        "restored_functions": ["task_route_profile", "run_carry_navigation",
                               "table_front_braking_observation"],
        "source_sha256": dict(HASHES),
        "retained_fixes": ["current grasp and prepared stance",
                          "current owner feedback and clock handling",
                          "carry_motion_handshake", "current physical_fault",
                          "current source joint and collision gates",
                          "maintainer-approved planar-speed warning policy"],
        "not_a_whole_commit_replay": True,
    }

def restore_route_script(script):
    replacements = {}
    for name in ("run_carry_navigation", "table_front_braking_observation"):
        replacements[name] = function_source("native_base_micro_probe.py", name)
    old = 'handshake=(row["common_mode_desc"]==cfg.get("motion_mode","LOCOMOTION_DEFAULT") and row["get_mode_desc"]==cfg.get("motion_mode","LOCOMOTION_DEFAULT") and row["get_mode_header_code"]==0 and row["state_age_s"]["get_mode"]<=.20 and row["common_source"]["name"]=="robotsim_official_cmd_vel")'
    new = 'handshake=carry_motion_handshake(row,cfg.get("motion_mode","LOCOMOTION_DEFAULT"),acquired=acquired)'
    if replacements["run_carry_navigation"].count(old) != 1:
        raise ValueError("baseline handshake adaptation no longer exact")
    replacements["run_carry_navigation"] = replacements["run_carry_navigation"].replace(old, new)
    lines = script.splitlines(keepends=True)
    nodes = [n for n in ast.parse(script).body
             if isinstance(n, ast.FunctionDef) and n.name in replacements]
    if len(nodes) != len(replacements):
        raise ValueError("generated carry definitions absent or ambiguous")
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        lines[node.lineno-1:node.end_lineno] = [replacements[node.name] + "\n"]
    result = "".join(lines)
    compile(result, "carry_78df5bd_current_io", "exec")
    return result
