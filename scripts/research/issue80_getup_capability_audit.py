#!/usr/bin/env python3
"""Read-only audit of the installed MC supine/prone recovery action graph."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


RUNTIME = Path("/home/lzy18001500226/projects/RobotSim/simulation/mujoco/raicom2026")
ROBOT = RUNTIME / "mc/bin/mc_param/src/robot/lx2501_3_t2d5"


class GetupAuditError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> dict:
    ruler = ROBOT / "action_ruler.yaml"
    setting = ROBOT / "action_setting.yaml"
    state = ROBOT / "sim/action_state.yaml"
    paths = (ruler, setting, state)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise GetupAuditError("installed MC recovery action source is missing or unsafe")
    rules = yaml.safe_load(ruler.read_text(encoding="utf-8"))
    settings = yaml.safe_load(setting.read_text(encoding="utf-8"))
    if not isinstance(rules, dict) or not isinstance(settings, dict):
        raise GetupAuditError("installed MC action YAML is malformed")
    action_rules = rules.get("action_rules")
    transition = settings.get("transition_actions")
    if not isinstance(action_rules, dict) or not isinstance(transition, dict):
        raise GetupAuditError("installed MC action graph is missing")
    required_edges = {
        "PASSIVE_DEFAULT": "LIE_UP_DEFAULT",
        "LIE_UP_DEFAULT": "GROUNDUP",
        "GROUNDUP": "STAND_DEFAULT",
    }
    for source, target in required_edges.items():
        next_actions = action_rules.get(source, {}).get("next_actions", [])
        if target not in next_actions:
            raise GetupAuditError(f"installed action graph lacks {source}->{target}")
    runners = {}
    for action, config in (("LIE_UP_DEFAULT", "ground_init_config.yaml"), ("GROUNDUP", "ground_up_config.yaml")):
        entries = transition.get(action)
        if not isinstance(entries, list) or len(entries) != 1:
            raise GetupAuditError(f"installed action setting lacks unique {action} runner")
        first = entries[0]
        runner = first.get("runners", [{}])[0] if isinstance(first, dict) else {}
        if not isinstance(runner, dict) or runner.get("cfg_path") != f"./rl/{config}":
            raise GetupAuditError(f"installed {action} runner differs from expected recovery config")
        cfg = ROBOT / "rl" / config
        if cfg.is_symlink() or not cfg.is_file():
            raise GetupAuditError(f"installed recovery config is missing: {config}")
        runners[action] = {
            "runner": runner,
            "period_ms": first.get("period_ms"),
            "config_path": str(cfg),
            "config_sha256": sha256(cfg),
        }
    return {
        "schema": "robotsim.issue80.getup-capability-audit.v1",
        "classification": "INSTALLED_MC_READ_ONLY_RECOVERY_ACTION_AUDIT",
        "status": "PASS_ACTION_GRAPH_PRESENT_NOT_YET_PHYSICALLY_VALIDATED",
        "graph": ["PASSIVE_DEFAULT", "LIE_UP_DEFAULT", "GROUNDUP", "STAND_DEFAULT"],
        "source_files_sha256": {str(path): sha256(path) for path in paths},
        "runners": runners,
        "required_runtime_validation": [
            "A source-valid supine initial configuration with clear surrounding workspace.",
            "One LIE_UP_DEFAULT request from observed PASSIVE_DEFAULT.",
            "Observed LIE_UP_DEFAULT, GROUNDUP, and STAND_DEFAULT transitions.",
            "Fresh upright posture, support/contact evidence, and the existing 2.5-second stable hold.",
        ],
        "limits": [
            "No inference is made from the action graph about a physical get-up success.",
            "The official current X2 guide warns that supine-to-stand is unsupported with a gripper attached.",
            "This installed RAICOM model's recovery behavior remains SIMULATION_ONLY until the runtime test passes.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.result.exists() or args.result.is_symlink():
        parser.error("--result must be new")
    try:
        result = audit()
    except (OSError, GetupAuditError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
