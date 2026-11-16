"""Small file/stdout lifecycle protocol; never controls physics."""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

PREFIX = "RUN_EVENT "

def publish(root, event, **fields):
    root = Path(root)
    payload = dict(schema="robotsim.run-lifecycle.v1", event=event,
                   observed_host_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(),
                   **fields)
    path = root / (event.lower() + ".json")
    # One producer per event; a consumer may briefly observe the new file.
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    print(PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)
    return payload

def forward_process(argv, *, cwd, log_path):
    """Stream lifecycle lines immediately, retain the complete child output."""
    with Path(log_path).open("x") as stream:
        child = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                if line.startswith((PREFIX, "ATTEMPT ", "LIVE_WINDOW_", "FIRST_PERSON_RGB_VIDEO ", "EVIDENCE ")):
                    print(line, end="", flush=True)
            return child.wait()
        finally:
            child.stdout.close()
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=8)

def observe_episode(episode, seen, *, destination):
    """Separate route termination from episode termination and safe teardown."""
    episode = Path(episode)
    for name, event in (("native_carry_route_result.json", "CARRY_ROUTE_FINISHED"),
                        ("task_execution_finished.json", "TASK_EXECUTION_FINISHED")):
        if event in seen:
            continue
        path = episode / name
        try:
            result = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        seen.add(event)
        publish(destination, event, episode=str(episode), source_path=str(path),
                source_mtime_ns=path.stat().st_mtime_ns,
                detected_delay_ms=max(0., (time.time_ns()-path.stat().st_mtime_ns)/1e6),
                status=result.get("status"), failure=result.get("failure"),
                gates=result.get("gates"), route_pass=result.get("pass_result"),
                physics_stopped=False, cleanup_complete=False)
