"""Build a read-only native-MjData capture overlay for Issue #80.

The vendor simulator owns live MuJoCo state.  This helper compiles an
evidence-local LD_PRELOAD library against the exact public MuJoCo version used
by the pinned simulator and modifies only the staged per-run launch script.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

CAPTURE_SCHEMA = "robotsim.issue80.native-mjdata-capture.v1"
MUJOCO_TAG = "3.3.7"
MUJOCO_HEADERS_BASE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco/"
    f"{MUJOCO_TAG}/include/mujoco/"
)
HEADER_SHA256 = {
    "mjdata.h": "f05eb63f68c10c67558c99ed703135d886060ee3bf1fecbd4a69d7e6d0e7e015",
    "mjexport.h": "2d0ec44615ce9e0dd3d80ce9ca2a8bcad1906840d2ab6b5a9af785f7be989f8b",
    "mjmacro.h": "067e74ccb1a17c3ca555d46cbae5cd86778220e5238706e40ea9bc7139cb7012",
    "mjmodel.h": "9462092aa6d958d54d7d20edfb85f2c16585e712fea15ed4c1593c5cb430522a",
    "mjplugin.h": "65d5e3cb193736a3ec0f834e6864a083e2c39e61b193cb82cdc245e361e9a58e",
    "mjrender.h": "497b9bd11971be5e79d68110b87d775787cf37a33aec51c11df47fea936999d8",
    "mjsan.h": "42389c4d645fe6f4c5d1195d72556f1238c126f979d05e56d86772949dc6b818",
    "mjspec.h": "726c733e44b37642fad15274f4ac359a26729b5f4db2185265432ef1d3dbc1ed",
    "mjthread.h": "f6bf0b51ed63049f0e6a04b19e97b76d36b261d27344a23c11b532640e94f0d7",
    "mjtnum.h": "85a6f45f592172df40d153267cb8d79357b1fa084e3f37fbffe14aee8283f154",
    "mjui.h": "74d8d97ce4777646f41589d870c754ed6e05cd6ac2350f2be1435c8405ed8ada",
    "mjvisualize.h": "94282a2905ca92db2df5412be6f1c8105436b367bf1eaab4ba2c390ebb8123f6",
    "mujoco.h": "8fd4333c3a19ee2e5cf235cd578cd8d2954ca55e585a62eb4393b34addf45a8f",
}
CAPTURE_SOURCE = Path(__file__).with_name("issue80_mjdata_capture.c")
CAPTURE_LIBRARY_NAME = "libissue80_mjdata_capture.so"
CAPTURE_TRACE_NAME = "native_mjdata_trace.jsonl"
CAPTURE_SUMMARY_NAME = "native_mjdata_capture_summary.json"
LAUNCH_MARKER = "# ROBOTSIM_ISSUE80_NATIVE_MJDATA_CAPTURE"
DEFAULT_INTERVAL_STEPS = 10


class MjDataCaptureError(RuntimeError):
    """The capture overlay cannot safely represent the pinned native state."""


@dataclass(frozen=True)
class NativeMjDataLivePose:
    """One read-only, native world-frame pelvis sample from the trace tail."""

    generation: int
    sim_time_s: float
    host_monotonic_s: float
    x_m: float
    y_m: float
    z_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float
    wx_rad_s: float
    wy_rad_s: float
    wz_rad_s: float


class NativeMjDataLivePoseReader:
    """Tail existing read-only capture evidence without changing simulator state."""

    _SAMPLE_EVENTS = frozenset((
        "post_forward_initial",
        "post_forward_reset_generation",
        "post_step_after_reset_generation",
        "post_step",
    ))

    def __init__(
        self,
        trace_path: Path,
        *,
        maximum_age_s: float,
        maximum_read_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if not isinstance(maximum_read_bytes, int) or maximum_read_bytes < 4096:
            raise ValueError("maximum_read_bytes must be an integer of at least 4096")
        if not math.isfinite(float(maximum_age_s)) or float(maximum_age_s) <= 0.0:
            raise ValueError("maximum_age_s must be finite and positive")
        self.trace_path = Path(trace_path)
        self.maximum_age_s = float(maximum_age_s)
        self.maximum_read_bytes = maximum_read_bytes
        self._offset = 0
        self._partial = b""
        self._latest: NativeMjDataLivePose | None = None
        self._latest_generation: int | None = None
        self._latest_sim_time_s: float | None = None
        self._generation_changed = False
        self._accepted_samples = 0
        self._skipped_initial_bytes = 0

    @staticmethod
    def _finite_vector(value: Any, length: int, label: str) -> tuple[float, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
            raise MjDataCaptureError(f"native live pose {label} is missing or malformed")
        parsed = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in parsed):
            raise MjDataCaptureError(f"native live pose {label} is nonfinite")
        return parsed

    @staticmethod
    def _rpy_from_wxyz(quaternion: Sequence[float]) -> tuple[float, float, float]:
        w, x, y, z = quaternion
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(pitch_term)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return roll, pitch, yaw

    def _parse_sample(self, row: Mapping[str, Any]) -> NativeMjDataLivePose | None:
        if row.get("schema") != CAPTURE_SCHEMA or row.get("event") not in self._SAMPLE_EVENTS:
            return None
        generation = row.get("generation")
        host_ns = row.get("host_monotonic_ns")
        sim_time = _as_finite_number(row.get("sim_time_s"))
        if not isinstance(generation, int) or generation < 0 or not isinstance(host_ns, int) or host_ns <= 0 or sim_time is None:
            raise MjDataCaptureError("native live pose has malformed generation or timestamp")
        state = row.get("pelvis", {}).get("state")
        if not isinstance(state, Mapping):
            raise MjDataCaptureError("native live pose has no pelvis state")
        position = self._finite_vector(state.get("world_position_m"), 3, "world position")
        quaternion = self._finite_vector(state.get("world_quaternion_wxyz"), 4, "world quaternion")
        velocity = self._finite_vector(state.get("world_spatial_velocity_rot_lin"), 6, "world spatial velocity")
        if self._latest_generation is not None:
            if generation < self._latest_generation:
                raise MjDataCaptureError("native live pose generation regressed")
            if generation > self._latest_generation:
                self._generation_changed = True
                self._latest_sim_time_s = None
            elif self._latest_sim_time_s is not None and sim_time < self._latest_sim_time_s - 1e-12:
                raise MjDataCaptureError("native live pose simulation time regressed within one generation")
            elif self._latest_sim_time_s is not None and abs(sim_time - self._latest_sim_time_s) <= 1e-12:
                return None
        self._latest_generation = generation
        self._latest_sim_time_s = sim_time
        self._accepted_samples += 1
        roll, pitch, yaw = self._rpy_from_wxyz(quaternion)
        return NativeMjDataLivePose(
            generation=generation, sim_time_s=sim_time, host_monotonic_s=host_ns / 1_000_000_000.0,
            x_m=position[0], y_m=position[1], z_m=position[2],
            roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
            wx_rad_s=velocity[0], wy_rad_s=velocity[1], wz_rad_s=velocity[2],
            vx_m_s=velocity[3], vy_m_s=velocity[4], vz_m_s=velocity[5],
        )

    def poll(self, *, now_monotonic_s: float) -> NativeMjDataLivePose | None:
        now = float(now_monotonic_s)
        if not math.isfinite(now):
            raise ValueError("now_monotonic_s must be finite")
        if not self.trace_path.exists():
            return None
        if self.trace_path.is_symlink() or not self.trace_path.is_file():
            raise MjDataCaptureError("native live pose trace is missing or unsafe")
        size = self.trace_path.stat().st_size
        if self._offset > size:
            raise MjDataCaptureError("native live pose trace was truncated during route execution")
        with self.trace_path.open("rb") as stream:
            start = self._offset
            if start == 0 and size > self.maximum_read_bytes:
                start = size - self.maximum_read_bytes
                stream.seek(start)
                stream.readline()
                start = stream.tell()
                self._skipped_initial_bytes = start
            elif size - start > self.maximum_read_bytes:
                start = size - self.maximum_read_bytes
                stream.seek(start)
                stream.readline()
                start = stream.tell()
                self._partial = b""
            else:
                stream.seek(start)
            payload = stream.read()
            self._offset = stream.tell()
        combined = self._partial + payload
        lines = combined.split(b"\n")
        self._partial = lines.pop() if combined and not combined.endswith(b"\n") else b""
        newest: NativeMjDataLivePose | None = None
        for raw_line in lines:
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise MjDataCaptureError("native live pose trace contains an invalid complete JSONL record") from exc
            if not isinstance(row, Mapping):
                raise MjDataCaptureError("native live pose trace record is not an object")
            candidate = self._parse_sample(row)
            if candidate is not None:
                newest = candidate
        if newest is not None:
            self._latest = newest
        if self._latest is not None:
            age = now - self._latest.host_monotonic_s
            if age < -0.02 or age > self.maximum_age_s:
                raise MjDataCaptureError(f"native live pose is stale or clock-skewed: age={age:.6f}s")
        return self._latest

    def snapshot(self) -> dict[str, Any]:
        return {
            "trace": str(self.trace_path),
            "maximum_age_s": self.maximum_age_s,
            "accepted_sample_count": self._accepted_samples,
            "generation_changed": self._generation_changed,
            "skipped_initial_bytes": self._skipped_initial_bytes,
            "latest": (
                {
                    "generation": self._latest.generation,
                    "sim_time_s": self._latest.sim_time_s,
                    "host_monotonic_s": self._latest.host_monotonic_s,
                    "x_m": self._latest.x_m,
                    "y_m": self._latest.y_m,
                    "z_m": self._latest.z_m,
                    "roll_rad": self._latest.roll_rad,
                    "pitch_rad": self._latest.pitch_rad,
                    "yaw_rad": self._latest.yaw_rad,
                } if self._latest is not None else None
            ),
        }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise MjDataCaptureError(f"refusing to overwrite evidence: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise MjDataCaptureError(f"{label} is missing or unsafe: {path}")


def _download_exact_headers(evidence: Path) -> dict[str, Any]:
    """Fetch only pinned header inputs and verify every tagged-source digest."""
    header_root = evidence / f"mujoco-{MUJOCO_TAG}-headers" / "include" / "mujoco"
    header_root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "source": "OFFICIAL_UPSTREAM_BUILD_HEADER_ONLY",
        "repository": "https://github.com/google-deepmind/mujoco",
        "tag": MUJOCO_TAG,
        "base_url": MUJOCO_HEADERS_BASE_URL,
        "files": {},
    }
    for filename, expected in sorted(HEADER_SHA256.items()):
        request = Request(
            MUJOCO_HEADERS_BASE_URL + filename,
            headers={"User-Agent": "RobotSim-Issue80-MjDataCapture/1"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                payload = response.read()
        except Exception as exc:  # pragma: no cover - network-specific
            raise MjDataCaptureError(
                f"could not obtain exact MuJoCo {MUJOCO_TAG} header {filename}: {exc!r}"
            ) from exc
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected:
            raise MjDataCaptureError(
                f"MuJoCo header digest mismatch for {filename}: {observed} != {expected}"
            )
        destination = header_root / filename
        destination.write_bytes(payload)
        manifest["files"][filename] = {
            "url": MUJOCO_HEADERS_BASE_URL + filename,
            "sha256": observed,
            "bytes": len(payload),
        }
    manifest["include_root"] = str(header_root.parent)
    return manifest


def instrument_staged_launch(
    launch_script: Path,
    *,
    library_container_path: str,
    trace_container_path: str,
    interval_steps: int,
) -> dict[str, str]:
    """Inject a one-run LD_PRELOAD declaration into an evidence-local script."""
    _require_regular_file(launch_script, "staged simulator launch script")
    if not isinstance(interval_steps, int) or not 1 <= interval_steps <= 1000:
        raise MjDataCaptureError("capture interval must be an integer in [1, 1000]")
    original = launch_script.read_text(encoding="utf-8")
    if LAUNCH_MARKER in original:
        raise MjDataCaptureError("staged launch script is already capture-instrumented")
    lines = original.splitlines(keepends=True)
    indices = [
        index for index, line in enumerate(lines)
        if line.startswith("export LD_LIBRARY_PATH=")
    ]
    if len(indices) != 1:
        raise MjDataCaptureError("staged launch script has no unique LD_LIBRARY_PATH export")
    insertion = (
        f"{LAUNCH_MARKER}\n"
        "# Read-only post-mj_step/post-mj_forward state capture; no control path.\n"
        f"export LD_PRELOAD={library_container_path}" + "__LD_PRELOAD_REF__" + "\n"
        f"export ROBOTSIM_MJDATA_CAPTURE_PATH={trace_container_path}\n"
        f"export ROBOTSIM_MJDATA_CAPTURE_INTERVAL_STEPS={interval_steps}\n"
        f"export ROBOTSIM_MJDATA_CAPTURE_EXPECTED_VERSION={MUJOCO_TAG}\n"
        "export ROBOTSIM_MJDATA_CAPTURE_PELVIS_BODY=pelvis\n"
        "export ROBOTSIM_MJDATA_CAPTURE_LEFT_FOOT_BODY=left_ankle_roll_link\n"
        "export ROBOTSIM_MJDATA_CAPTURE_RIGHT_FOOT_BODY=right_ankle_roll_link\n"
    ).replace("__LD_PRELOAD_REF__", "$" + "{LD_PRELOAD:+:$" + "{LD_PRELOAD}}")
    lines.insert(indices[0] + 1, insertion)
    instrumented = "".join(lines)
    launch_script.write_text(instrumented, encoding="utf-8")
    return {
        "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "instrumented_sha256": hashlib.sha256(instrumented.encode()).hexdigest(),
    }

def prepare_mjdata_capture_overlay(
    evidence: Path,
    *,
    staged_launch_script: Path,
    interval_steps: int = DEFAULT_INTERVAL_STEPS,
    compiler: str | None = None,
) -> dict[str, Any]:
    """Build a native 3.3.7 capture shared library in a fresh evidence run."""
    _require_regular_file(CAPTURE_SOURCE, "capture source")
    if evidence.is_symlink() or not evidence.is_dir():
        raise MjDataCaptureError(f"evidence directory is missing or unsafe: {evidence}")
    compiler = compiler or shutil.which("gcc")
    if not compiler:
        raise MjDataCaptureError("gcc is required to build the evidence-local capture overlay")

    header_manifest = _download_exact_headers(evidence)
    include_root = Path(header_manifest["include_root"])
    library = evidence / CAPTURE_LIBRARY_NAME
    command = [
        compiler, "-shared", "-fPIC", "-O2", "-std=c11", "-Wall", "-Wextra",
        "-Werror=return-type", "-I", str(include_root), str(CAPTURE_SOURCE),
        "-o", str(library), "-ldl", "-pthread",
    ]
    built = subprocess.run(command, text=True, capture_output=True, timeout=30)
    if built.returncode != 0 or library.is_symlink() or not library.is_file():
        raise MjDataCaptureError(
            "native capture overlay did not compile: "
            + json.dumps(
                {
                    "returncode": built.returncode,
                    "stdout": built.stdout[-4000:],
                    "stderr": built.stderr[-4000:],
                },
                sort_keys=True,
            )
        )
    dynamic = subprocess.run(
        ["readelf", "-d", str(library)], text=True, capture_output=True, timeout=10
    )
    if dynamic.returncode != 0 or "libmujoco" in dynamic.stdout:
        raise MjDataCaptureError(
            "capture overlay must resolve MuJoCo only from the live simulator process"
        )
    launch_hashes = instrument_staged_launch(
        staged_launch_script,
        library_container_path=f"/workspace/evidence/{CAPTURE_LIBRARY_NAME}",
        trace_container_path=f"/workspace/evidence/{CAPTURE_TRACE_NAME}",
        interval_steps=interval_steps,
    )
    return {
        "schema": "robotsim.issue80.native-mjdata-capture-overlay.v1",
        "status": "PASS_OVERLAY_BUILT",
        "classification": "SIMULATION_ONLY_READ_ONLY_NATIVE_MJDATA_OBSERVABILITY",
        "source_file": str(CAPTURE_SOURCE),
        "source_sha256": sha256_file(CAPTURE_SOURCE),
        "compiler": compiler,
        "compile_command": command,
        "compiler_stdout": built.stdout,
        "compiler_stderr": built.stderr,
        "library": str(library),
        "library_sha256": sha256_file(library),
        "library_dynamic_section": dynamic.stdout,
        "expected_mujoco_runtime_version": MUJOCO_TAG,
        "header_manifest": header_manifest,
        "trace": str(evidence / CAPTURE_TRACE_NAME),
        "interval_steps": interval_steps,
        "staged_launch_script": str(staged_launch_script),
        "launch_hashes": launch_hashes,
        "hook_behavior": {
            "intercepts": ["mj_step", "mj_forward"],
            "read_only_after_original_call": True,
            "never_writes": ["mjData.ctrl", "mjData.qpos", "mjData.qvel", "model", "vendor_runtime"],
            "never_calls": [
                "additional mj_step",
                "additional mj_forward",
                "SetMcAction",
                "velocity publish",
            ],
        },
    }


def run_mjdata_capture_smoke(
    evidence: Path,
    *,
    overlay: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    """Exercise the hook with the pinned 3.3.7 library on a tiny stepped model.

    This is an L2 instrumentation check only.  Its contact values are never
    interpreted as X2 data and it never starts the vendor simulator or MC.
    """
    library = Path(str(overlay.get("library", "")))
    _require_regular_file(library, "compiled capture library")
    mujoco_library = runtime_root / "sim_mujoco/lib/libmujoco.so.3.3.7"
    _require_regular_file(mujoco_library, "pinned runtime MuJoCo library")
    compiler = str(overlay.get("compiler", ""))
    if not compiler:
        raise MjDataCaptureError("overlay does not record a compiler")
    smoke_xml = evidence / "native_mjdata_capture_smoke.xml"
    smoke_xml.write_text(
        """<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="pelvis" pos="0 0 .3">
      <freejoint/>
      <body name="left_ankle_roll_link"><geom type="box" size=".1 .1 .1" mass="1"/></body>
      <body name="right_ankle_roll_link" pos="0 .3 0"><geom type="box" size=".1 .1 .1" mass="1"/></body>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    smoke_source = r"""
#include <stdio.h>
#include <mujoco/mujoco.h>
int main(int argc, char** argv) {
  char error[1024] = {0};
  mjModel* model = mj_loadXML(argv[1], 0, error, sizeof(error));
  if (!model) { fprintf(stderr, "%s\n", error); return 2; }
  mjData* data = mj_makeData(model);
  for (int i = 0; i < 300; ++i) mj_step(model, data);
  printf("time=%.6f contacts=%d\n", data->time, data->ncon);
  mj_deleteData(data); mj_deleteModel(model); return 0;
}
"""
    executable = evidence / "native_mjdata_capture_smoke"
    header_root = str(overlay["header_manifest"]["include_root"])
    compile_command = [
        compiler,
        "-x",
        "c",
        "-",
        "-I",
        header_root,
        "-L",
        str(mujoco_library.parent),
        f"-Wl,-rpath,{mujoco_library.parent}",
        "-Wl,-l:libmujoco.so.3.3.7",
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(
        compile_command,
        input=smoke_source,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if compiled.returncode != 0:
        raise MjDataCaptureError(
            "native MjData smoke executable did not compile: " + compiled.stderr[-4000:]
        )
    trace = evidence / "native_mjdata_capture_smoke.jsonl"
    environment = {
        **os.environ,
        "LD_PRELOAD": str(library),
        "LD_LIBRARY_PATH": str(mujoco_library.parent),
        "ROBOTSIM_MJDATA_CAPTURE_PATH": str(trace),
        "ROBOTSIM_MJDATA_CAPTURE_INTERVAL_STEPS": "10",
        "ROBOTSIM_MJDATA_CAPTURE_EXPECTED_VERSION": MUJOCO_TAG,
    }
    executed = subprocess.run(
        [str(executable), str(smoke_xml)],
        text=True,
        capture_output=True,
        timeout=15,
        env=environment,
    )
    if executed.returncode != 0:
        raise MjDataCaptureError(
            "native MjData smoke executable did not run: " + executed.stderr[-4000:]
        )
    summary = summarize_mjdata_capture(trace)
    if (
        summary["status"] != "PASS_NATIVE_MJDATA_CAPTURE"
        or summary["foot_floor_contact_metrics"]["max_total_contact_count"] <= 0
    ):
        raise MjDataCaptureError("native MjData smoke did not capture physical contacts")
    return {
        "status": "PASS_NATIVE_MJDATA_CAPTURE_SMOKE",
        "classification": "L2_SYNTHETIC_HOOK_INSTRUMENTATION_ONLY",
        "runtime_mujoco_library": str(mujoco_library),
        "runtime_mujoco_library_sha256": sha256_file(mujoco_library),
        "compile_command": compile_command,
        "compile_stdout": compiled.stdout,
        "compile_stderr": compiled.stderr,
        "execution_stdout": executed.stdout,
        "execution_stderr": executed.stderr,
        "smoke_trace": str(trace),
        "smoke_trace_sha256": sha256_file(trace),
        "smoke_summary": summary,
    }


def _as_finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _trace_rows(trace_path: Path) -> tuple[list[dict[str, Any]], bool]:
    _require_regular_file(trace_path, "native MjData trace")
    rows: list[dict[str, Any]] = []
    raw_lines = trace_path.read_bytes().splitlines(keepends=True)
    nonempty_indices = [
        index for index, raw_line in enumerate(raw_lines) if raw_line.strip()
    ]
    last_nonempty = nonempty_indices[-1] if nonempty_indices else -1
    truncated_final_record = False
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        line_number = index + 1
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            if index == last_nonempty and not raw_line.endswith(b"\n"):
                truncated_final_record = True
                break
            raise MjDataCaptureError(
                f"native MjData trace line {line_number} is not JSON"
            ) from exc
        if not isinstance(item, dict) or item.get("schema") != CAPTURE_SCHEMA:
            raise MjDataCaptureError(
                f"native MjData trace line {line_number} has an unexpected schema"
            )
        rows.append(item)
    if not rows:
        raise MjDataCaptureError("native MjData trace contains no rows")
    return rows, truncated_final_record


def _extract_standing_events(standing_trace: Path | None) -> list[dict[str, Any]]:
    if standing_trace is None or standing_trace.is_symlink() or not standing_trace.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in standing_trace.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, Mapping):
            continue
        if row.get("event") in {
            "JD_REQUEST",
            "JD_RESPONSE",
            "SD_REQUEST",
            "SD_RESPONSE",
            "NATIVE_TRANSITION",
        }:
            records.append(
                {
                    "event": row.get("event"),
                    "receipt_monotonic_ns": row.get("receipt_monotonic_ns"),
                    "phase": row.get("phase"),
                    "state": row.get("state"),
                }
            )
    return records


def _is_floor_contact(contact: Mapping[str, Any]) -> bool:
    def names(endpoint: Any) -> tuple[str, str]:
        if not isinstance(endpoint, Mapping):
            return "", ""
        return (
            str(endpoint.get("name") or "").lower(),
            str(endpoint.get("body_name") or "").lower(),
        )

    names1 = names(contact.get("geom1"))
    names2 = names(contact.get("geom2"))
    return any("floor" in name or name == "world" for name in names1 + names2)


def _world_planar_speed(sample: Mapping[str, Any]) -> float | None:
    vector = (
        sample.get("pelvis", {})
        .get("state", {})
        .get("world_spatial_velocity_rot_lin")
    )
    if not isinstance(vector, Sequence) or len(vector) != 6:
        return None
    vx, vy = _as_finite_number(vector[3]), _as_finite_number(vector[4])
    return math.hypot(vx, vy) if vx is not None and vy is not None else None


def summarize_mjdata_capture(
    trace_path: Path,
    *,
    standing_trace: Path | None = None,
) -> dict[str, Any]:
    """Summarize actual native samples without fabricating unavailable forces."""
    rows, truncated_final_record = _trace_rows(trace_path)
    metadata = next((row for row in rows if row.get("event") == "model_metadata"), None)
    errors = [row for row in rows if row.get("event") == "capture_error"]
    samples = [
        row
        for row in rows
        if row.get("event") in {
            "post_forward_initial",
            "post_forward_reset_generation",
            "post_step",
        }
    ]
    status = "PASS_NATIVE_MJDATA_CAPTURE"
    reasons: list[str] = []
    if metadata is None:
        status = "BLOCKED_NATIVE_MJDATA_CAPTURE"
        reasons.append("MODEL_METADATA_MISSING")
    if errors:
        status = "BLOCKED_NATIVE_MJDATA_CAPTURE"
        reasons.append("CAPTURE_ERROR_EVENT")
    if not samples:
        status = "BLOCKED_NATIVE_MJDATA_CAPTURE"
        reasons.append("NO_PHYSICAL_SAMPLES")
    if truncated_final_record and status == "PASS_NATIVE_MJDATA_CAPTURE":
        status = "PARTIAL_NATIVE_MJDATA_CAPTURE"
        reasons.append("TRUNCATED_FINAL_RECORD")

    samples_by_generation: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        generation = sample.get("generation")
        sim_time = _as_finite_number(sample.get("sim_time_s"))
        host_time = sample.get("host_monotonic_ns")
        if not isinstance(generation, int) or sim_time is None or not isinstance(host_time, int):
            status = "BLOCKED_NATIVE_MJDATA_CAPTURE"
            reasons.append("MALFORMED_SAMPLE_TIMESTAMP")
            continue
        samples_by_generation.setdefault(generation, []).append(sample)
    timestamp_violations = 0
    for group in samples_by_generation.values():
        previous = None
        for sample in group:
            current = _as_finite_number(sample.get("sim_time_s"))
            if previous is not None and current is not None and current + 1e-12 < previous:
                timestamp_violations += 1
            previous = current
    if timestamp_violations:
        status = "BLOCKED_NATIVE_MJDATA_CAPTURE"
        reasons.append("NONMONOTONIC_SIM_TIMESTAMP_WITHIN_GENERATION")

    current_generation = max(samples_by_generation) if samples_by_generation else None
    current_samples = samples_by_generation.get(current_generation, []) if current_generation is not None else []
    planar_speeds = [
        value for value in (_world_planar_speed(sample) for sample in current_samples)
        if value is not None
    ]
    positions = [
        sample.get("pelvis", {}).get("state", {}).get("world_position_m")
        for sample in current_samples
    ]
    valid_positions = [
        position
        for position in positions
        if isinstance(position, Sequence)
        and len(position) == 3
        and all(_as_finite_number(value) is not None for value in position)
    ]
    displacement = None
    if len(valid_positions) >= 2:
        first, last = valid_positions[0], valid_positions[-1]
        dx, dy, dz = (
            float(last[0]) - float(first[0]),
            float(last[1]) - float(first[1]),
            float(last[2]) - float(first[2]),
        )
        displacement = {
            "dx_m": dx,
            "dy_m": dy,
            "planar_m": math.hypot(dx, dy),
            "dz_m": dz,
        }

    contact_metrics: dict[str, Any] = {
        "max_total_contact_count": 0,
        "min_foot_floor_distance_m": None,
        "max_left_foot_floor_normal_force_n": None,
        "max_right_foot_floor_normal_force_n": None,
        "max_left_foot_floor_tangential_force_n": None,
        "max_right_foot_floor_tangential_force_n": None,
        "max_left_foot_floor_relative_tangential_speed_m_s": None,
        "max_right_foot_floor_relative_tangential_speed_m_s": None,
    }
    actuator_peak = 0.0
    qfrc_peak = 0.0
    for sample in current_samples:
        contact_metrics["max_total_contact_count"] = max(
            contact_metrics["max_total_contact_count"],
            int(sample.get("contact_count", 0) or 0),
        )
        for value in sample.get("actuator_force", []):
            numeric = _as_finite_number(value)
            if numeric is not None:
                actuator_peak = max(actuator_peak, abs(numeric))
        for value in sample.get("qfrc_actuator", []):
            numeric = _as_finite_number(value)
            if numeric is not None:
                qfrc_peak = max(qfrc_peak, abs(numeric))
        for contact in sample.get("contacts", []):
            if not isinstance(contact, Mapping) or not _is_floor_contact(contact):
                continue
            distance = _as_finite_number(contact.get("distance_m"))
            if distance is not None:
                old = contact_metrics["min_foot_floor_distance_m"]
                contact_metrics["min_foot_floor_distance_m"] = (
                    distance if old is None else min(float(old), distance)
                )
            for side in ("left", "right"):
                if not contact.get(f"involves_selected_{side}_foot"):
                    continue
                for suffix, source_key in (
                    ("normal_force_n", "normal_force_n"),
                    ("tangential_force_n", "tangential_force_norm_n"),
                    ("relative_tangential_speed_m_s", "relative_tangential_speed_m_s"),
                ):
                    numeric = _as_finite_number(contact.get(source_key))
                    if numeric is None:
                        continue
                    metric_key = f"max_{side}_foot_floor_{suffix}"
                    old = contact_metrics[metric_key]
                    contact_metrics[metric_key] = (
                        abs(numeric) if old is None else max(float(old), abs(numeric))
                    )

    sample_timeline = []
    for sample in current_samples:
        state = sample.get("pelvis", {}).get("state", {})
        sample_timeline.append(
            {
                "sim_time_s": sample.get("sim_time_s"),
                "host_monotonic_ns": sample.get("host_monotonic_ns"),
                "event": sample.get("event"),
                "planar_speed_world_m_s": _world_planar_speed(sample),
                "pelvis_world_position_m": state.get("world_position_m")
                if isinstance(state, Mapping)
                else None,
                "contact_count": sample.get("contact_count"),
            }
        )

    return {
        "schema": "robotsim.issue80.native-mjdata-capture-summary.v1",
        "status": status,
        "failure_reasons": sorted(set(reasons)),
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "row_count": len(rows),
        "truncated_final_record": truncated_final_record,
        "capture_error_events": errors,
        "model_metadata": metadata,
        "generation_count": len(samples_by_generation),
        "current_generation": current_generation,
        "current_generation_sample_count": len(current_samples),
        "timestamp_violations_within_generation": timestamp_violations,
        "current_generation_initial_pelvis_world_pose": (
            current_samples[0].get("pelvis", {}).get("state") if current_samples else None
        ),
        "current_generation_final_pelvis_world_pose": (
            current_samples[-1].get("pelvis", {}).get("state") if current_samples else None
        ),
        "max_pelvis_world_planar_speed_m_s": max(planar_speeds) if planar_speeds else None,
        "pelvis_finite_difference_displacement": displacement,
        "foot_floor_contact_metrics": contact_metrics,
        "max_abs_actuator_force": actuator_peak,
        "max_abs_generalized_actuator_force": qfrc_peak,
        "standing_transition_events_from_observer": _extract_standing_events(standing_trace),
        "timestamp_semantics": {
            "sim_time_s": "exact MuJoCo mjData.time at the post-call sample",
            "host_monotonic_ns": "CLOCK_MONOTONIC from the simulator container",
            "contact_force": (
                "mj_contactForce raw normal/tangent contact-frame components; "
                "force-times-timestep is an explicit per-step diagnostic approximation"
            ),
            "relative_contact_velocity": (
                "geom2 minus geom1, estimated at the contact point from "
                "mj_objectVelocity world-frame spatial velocities"
            ),
        },
        "sample_timeline": sample_timeline,
    }
