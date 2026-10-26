#!/usr/bin/env python3
"""No-physics checks inside the pinned RAICOM image, before a live episode."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


def run_host(*, standing: bool = False, protocol: str = 'sd') -> int:
    from issue80_raicom_reset_only import (
        IMAGE, IMAGE_DIGEST, RUNTIME, RESET_OBSERVER,
        MC_SIM_CONFIG_DUMP_CONTAINER_PATH, _stage_mc_config_dump_writeback,
        MC_SIM_GENERATED_DIRECTORY,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
    from path_governance import create_unique_output_directory, write_artifact_manifest
    output = create_unique_output_directory(
        "/home/lzy18001500226/projects/RobotSim/reviews/issue-80",
        prefix="raicom-prelaunch",
    ).native_path
    image = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE],
                           check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    if image != IMAGE_DIGEST:
        raise RuntimeError("runtime image identity changed")
    dump, manifest = _stage_mc_config_dump_writeback(output)
    shutil.copyfile(__file__, output / Path(__file__).name)
    shutil.copyfile(RESET_OBSERVER, output / RESET_OBSERVER.name)
    if standing:
        from issue80_raicom_stand_takeover import StandingRun
        StandingRun(argparse.Namespace(attempt_id=str(uuid.uuid4()),protocol=protocol),output,'no-runtime-smoke').prepare_observer()
    name = "robotsim-prelaunch-" + str(uuid.uuid4())
    cmd = ["docker", "run", "--rm", "--pull=never", "--name", name,
           "--network=none", "--read-only", "--cap-drop=ALL",
           "--security-opt=no-new-privileges", "--user=1000:1000",
           "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
           "--tmpfs", MC_SIM_GENERATED_DIRECTORY + ":rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=0755,size=8m",
           "--mount", f"type=bind,source={RUNTIME},target=/workspace/raicom2026,readonly",
           "--mount", f"type=bind,source={output},target=/workspace/evidence",
           "--mount", f"type=bind,source={dump},target={MC_SIM_CONFIG_DUMP_CONTAINER_PATH}",
           "-e", "HOME=/tmp", "-e", "ROS_DOMAIN_ID=20",
           "-e", "RMW_IMPLEMENTATION=rmw_fastrtps_cpp", IMAGE, "bash", "-lc",
           "source /opt/ros/humble/setup.bash && "
           "source /workspace/raicom2026/aimdk_msgs_ws/install/setup.bash && "
           "python3 /workspace/evidence/issue80_raicom_prelaunch.py "
           "--output-dir /workspace/evidence --dump-target " + MC_SIM_CONFIG_DUMP_CONTAINER_PATH]
    if standing:
        cmd[-1] += " --standing --protocol "+protocol
    (output / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        (output / "console.log").write_text(result.stdout + result.stderr)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=4)
    listing = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"],
                             check=True, capture_output=True, text=True, timeout=5)
    absent = name not in listing.stdout.splitlines()
    (output / "cleanup.json").write_text(json.dumps({"container": name, "absent": absent}) + "\n")
    write_artifact_manifest(output)
    print(result.stdout)
    print(json.dumps({"output": str(output), "container_absent": absent}))
    return result.returncode if absent else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dump-target", type=Path, required=True)
    parser.add_argument("--standing", action="store_true")
    parser.add_argument("--protocol", choices=('sd','jd-sd'), default='sd')
    args = parser.parse_args()
    root = args.output_dir.resolve(strict=True)
    target = args.dump_target
    report = {"schema": "robotsim.issue80.raicom-prelaunch.v1", "status": "BLOCKED",
              "physics_started": False, "mc_started": False}
    try:
        import rclpy
        import yaml
        from aimdk_msgs.msg import JointCommandArray, JointStateArray, McCommonState
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu
        report["imports"] = {"python": sys.version, "rclpy": rclpy.__file__,
                             "yaml": yaml.__version__, "types": [t.__name__ for t in
                             (Odometry, Imu, McCommonState, JointStateArray, JointCommandArray)]}
        # Same generated destination and read-only parent as MC's copy_file.
        original = target.read_bytes()
        with target.open("wb") as stream:
            stream.write(b"prelaunch-truncate-write-test\n")
            stream.flush()
            os.fsync(stream.fileno())
        assert target.read_bytes() == b"prelaunch-truncate-write-test\n"
        with target.open("wb") as stream:
            stream.write(original)
        report["dump_open_truncate_overwrite"] = "PASS"
        report["dump_restored_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        sim_directory = target.parent / "sim"
        for seed in (root / "mc_runtime_writeback/sim_seed").glob("*.yaml"):
            shutil.copyfile(seed, sim_directory / seed.name)
        generated = sim_directory / "action_setting.yaml"
        payload = generated.read_bytes()
        with generated.open("wb") as stream:
            stream.write(payload)
        atomic = sim_directory / "prelaunch-tmp.yaml"
        atomic.write_bytes(payload)
        os.replace(atomic, generated)
        assert generated.read_bytes() == payload
        report["sim_tmpfs_generation_and_atomic_replace"] = "PASS"
        try:
            probe = target.parent / "robotsim_prelaunch_forbidden_parent_write"
            with probe.open("xb") as stream:
                stream.write(b"must-not-write")
        except OSError as exc:
            if exc.errno not in (errno.EROFS, errno.EACCES):
                raise
            report["vendor_parent_readonly"] = {"status": "PASS", "errno": exc.errno}
        else:
            raise RuntimeError("vendor config parent unexpectedly writable")
        # A file bind mount supports truncation, but rename/unlink is not
        # promised. Record the actual kernel result instead of assuming it.
        replacement = root / "replacement-test.tmp"
        replacement.write_bytes(original)
        try:
            os.replace(replacement, target)
        except OSError as exc:
            report["atomic_replace"] = {"supported": False, "errno": exc.errno,
                                        "reason": str(exc)}
        else:
            report["atomic_replace"] = {"supported": True}
        observer = root / ("issue80_raicom_stand_observer.py" if args.standing
                           else "issue80_raicom_reset_only_observer.py")
        capture = root / "observer_smoke"
        capture.mkdir()
        if args.standing:
            shutil.copyfile(root/'standing_joint_limits.json',capture/'standing_joint_limits.json')
        smoke_command=[sys.executable, str(observer), "--construction-smoke",
                                 "--output-dir", str(capture), "--reset-event-file",
                                 str(capture / "reset_dispatch_event.json")]
        if args.standing:
            smoke_command += ['--protocol',args.protocol]
        result = subprocess.run(smoke_command,
                                capture_output=True, text=True, timeout=10)
        report["observer_returncode"] = result.returncode
        report["observer_stdout"] = result.stdout
        report["observer_stderr"] = result.stderr
        smoke = capture / ("standing_observer_smoke.json" if args.standing else "reset_only_observer_smoke.json")
        if result.returncode != 0 or not smoke.is_file():
            raise RuntimeError("actual ROS observer construction failed")
        report["observer_smoke"] = json.loads(smoke.read_text())
        if report["observer_smoke"]["status"] != "PASS_CONSTRUCTION_SMOKE":
            raise RuntimeError("observer interface counts failed")
        report["status"] = "PASS_NO_PHYSICS_PRELAUNCH"
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
    (root / "prelaunch_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS_NO_PHYSICS_PRELAUNCH" else 2


if __name__ == "__main__":
    raise SystemExit(run_host(standing="--standing" in sys.argv,protocol='jd-sd' if '--jd-sd' in sys.argv else 'sd') if "--host" in sys.argv else main())
