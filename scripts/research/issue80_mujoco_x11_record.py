#!/usr/bin/env python3
"""Passive, window-verified X11 recording for an isolated Issue 80 MuJoCo run.

This helper reads only the MuJoCo window named by the run's verified Reset event.
It sends no input and begins recording at the first nonzero linear or angular command.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

XWD_HEADER_WORDS = 25
XWD_HEADER_BYTES = XWD_HEADER_WORDS * 4
XWD_FILE_VERSION = 7
XWD_Z_PIXMAP = 2
XWD_TRUE_COLOR = 4


def parse_xwd_header(contents: bytes) -> dict[str, int | str]:
    """Parse and validate the fixed XWD header without image-library dependencies."""
    if len(contents) < XWD_HEADER_BYTES:
        raise ValueError("truncated XWD header")
    candidates: list[tuple[str, tuple[int, ...]]] = []
    for endian in ("<", ">"):
        fields = struct.unpack(f"{endian}{XWD_HEADER_WORDS}I", contents[:XWD_HEADER_BYTES])
        if fields[1] == XWD_FILE_VERSION:
            candidates.append((endian, fields))
    if len(candidates) != 1:
        raise ValueError("XWD byte order or file version is ambiguous")
    endian, fields = candidates[0]
    (
        header_size, version, pixmap_format, depth, width, height, _xoffset,
        byte_order, _bitmap_unit, _bitmap_bit_order, _bitmap_pad, bits_per_pixel,
        bytes_per_line, visual_class, red_mask, green_mask, blue_mask, _bits_per_rgb,
        _colormap_entries, ncolors, _window_width, _window_height, _x, _y, _border_width,
    ) = fields
    if not XWD_HEADER_BYTES <= header_size <= len(contents):
        raise ValueError("invalid XWD header size")
    if version != XWD_FILE_VERSION or pixmap_format != XWD_Z_PIXMAP:
        raise ValueError("XWD must use the supported version and ZPixmap format")
    if not 0 < width <= 16384 or not 0 < height <= 16384:
        raise ValueError("XWD dimensions are outside the supported range")
    if byte_order not in (0, 1) or bits_per_pixel not in (24, 32):
        raise ValueError("unsupported XWD byte order or pixel width")
    bytes_per_pixel = (bits_per_pixel + 7) // 8
    if bytes_per_line < width * bytes_per_pixel:
        raise ValueError("XWD row stride is shorter than its image width")
    if visual_class != XWD_TRUE_COLOR or not (red_mask and green_mask and blue_mask):
        raise ValueError("only TrueColor XWD images with RGB masks are supported")
    data_offset = header_size + ncolors * 12
    if len(contents) < data_offset + bytes_per_line * height:
        raise ValueError("truncated XWD color table or pixel data")
    return {
        "endian": endian,
        "header_size": header_size,
        "width": width,
        "height": height,
        "byte_order": byte_order,
        "bits_per_pixel": bits_per_pixel,
        "bytes_per_line": bytes_per_line,
        "visual_class": visual_class,
        "red_mask": red_mask,
        "green_mask": green_mask,
        "blue_mask": blue_mask,
        "color_count": ncolors,
        "pixel_data_offset": data_offset,
    }


def _verify_x11_window(window_id: int, expected_pid: int, expected_title: str) -> dict[str, Any]:
    xid = f"0x{window_id:x}"
    info = subprocess.run(
        ["xwininfo", "-id", xid], capture_output=True, text=True, timeout=3, check=False
    )
    if info.returncode != 0:
        raise RuntimeError("verified MuJoCo window is no longer available")
    title_match = re.search(r'Window id:.*?"(.*?)"', info.stdout)
    if not title_match or title_match.group(1) != expected_title:
        raise RuntimeError("MuJoCo window title no longer matches the Reset event")
    properties = subprocess.run(
        ["xprop", "-id", xid, "_NET_WM_PID"],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    pid_match = re.search(r"=\s*(\d+)", properties.stdout)
    if properties.returncode != 0 or not pid_match or int(pid_match.group(1)) != expected_pid:
        raise RuntimeError("MuJoCo window PID no longer matches the Reset event")
    return {"xid": xid, "pid": expected_pid, "title": expected_title}


def _capture_xwd(window_id: int) -> bytes:
    xid = f"0x{window_id:x}"
    with tempfile.NamedTemporaryFile(prefix="issue80-xwd-", suffix=".xwd") as temporary:
        result = subprocess.run(
            ["xwd", "-silent", "-nobdrs", "-id", xid, "-out", temporary.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("xwd failed to capture the verified MuJoCo window")
        temporary.seek(0)
        contents = temporary.read()
    parse_xwd_header(contents)
    return contents


def _xwd_to_rgb_image(contents: bytes):
    """Decode a validated 24/32-bit TrueColor XWD image to a Pillow RGB image."""
    import numpy as np
    from PIL import Image

    header = parse_xwd_header(contents)
    width, height = int(header["width"]), int(header["height"])
    stride = int(header["bytes_per_line"])
    offset = int(header["pixel_data_offset"])
    bpp = int(header["bits_per_pixel"])
    byte_order = int(header["byte_order"])
    payload = memoryview(contents)[offset:offset + stride * height]
    if bpp == 32:
        dtype = np.dtype("<u4" if byte_order == 0 else ">u4")
        packed = np.frombuffer(payload, dtype=dtype).reshape(height, stride // 4)[:, :width]
    else:
        rows = np.frombuffer(payload, dtype=np.uint8).reshape(height, stride)
        pixels = rows[:, :width * 3].reshape(height, width, 3).astype(np.uint32)
        if byte_order == 0:
            packed = pixels[:, :, 0] | (pixels[:, :, 1] << 8) | (pixels[:, :, 2] << 16)
        else:
            packed = (pixels[:, :, 0] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 2]

    channels = []
    for name in ("red_mask", "green_mask", "blue_mask"):
        mask = int(header[name])
        shift = (mask & -mask).bit_length() - 1
        maximum = mask >> shift
        values = (packed & mask) >> shift
        channels.append(((values * 255 + maximum // 2) // maximum).astype(np.uint8))
    rgb = np.stack(channels, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _next_complete_trace_event(trace_path: Path, offset: int) -> tuple[int, list[dict[str, Any]]]:
    if not trace_path.exists():
        return offset, []
    with trace_path.open("rb") as stream:
        stream.seek(offset)
        chunk = stream.read()
        new_offset = stream.tell()
    complete_lines = chunk.splitlines(keepends=True)
    parsed: list[dict[str, Any]] = []
    for index, line in enumerate(complete_lines):
        if not line.endswith(b"\n"):
            new_offset -= len(line)
            break
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            parsed.append(value)
    return new_offset, parsed


def record_verified_walk_window(
    evidence_dir: Path,
    *,
    fps: float = 5.0,
    duration_s: float = 11.5,
    wait_timeout_s: float = 180.0,
    output_name: str = "mujoco_walk.gif",
) -> dict[str, Any]:
    """Wait for the authorized pulse, then record only its verified MuJoCo X11 window."""
    if not 1.0 <= fps <= 10.0:
        raise ValueError("recording fps must be between 1 and 10")
    if not 1.0 <= duration_s <= 12.0:
        raise ValueError("recording duration must be between 1 and 12 seconds")
    root = evidence_dir.resolve(strict=True)
    reset_path = root / "reset_dispatch_event.json"
    trace_path = root / "m1_first_motion_trace.jsonl"
    output_path = root / output_name
    manifest_path = root / "x11_walk_capture.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite existing X11 recording evidence")
    deadline = time.monotonic() + wait_timeout_s
    while not reset_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not reset_path.is_file():
        raise TimeoutError("verified Reset dispatch event did not appear")
    reset_event = json.loads(reset_path.read_text(encoding="utf-8"))
    window = reset_event.get("window")
    if not isinstance(window, dict):
        raise ValueError("Reset dispatch event lacks the verified MuJoCo window")
    window_id = int(window["window_id"])
    expected_pid = int(window["pid"])
    expected_title = str(window["title"])
    verified_window = _verify_x11_window(window_id, expected_pid, expected_title)

    offset = 0
    pulse_start: float | None = None
    trace_deadline = time.monotonic() + wait_timeout_s
    while pulse_start is None and time.monotonic() < trace_deadline:
        offset, events = _next_complete_trace_event(trace_path, offset)
        for event in events:
            command_values = (
                event.get("forward_velocity_m_s", 0.0),
                event.get("lateral_velocity_m_s", 0.0),
                event.get("angular_velocity_rad_s", 0.0),
            )
            if event.get("event") == "velocity_command" and any(
                abs(float(value or 0.0)) > 0.0 for value in command_values
            ):
                pulse_start = float(event["wall_monotonic_sec"])
                break
        if pulse_start is None:
            time.sleep(0.02)
    if pulse_start is None:
        raise TimeoutError("nonzero M1 linear or angular command was not observed in the trace")

    from PIL import Image

    frames = []
    frame_times = []
    errors = []
    start = time.monotonic()
    stop_at = start + duration_s
    next_frame = start
    next_identity_check = start
    consecutive_capture_errors = 0
    while time.monotonic() < stop_at:
        now = time.monotonic()
        if now >= next_identity_check:
            try:
                verified_window = _verify_x11_window(window_id, expected_pid, expected_title)
            except Exception as exc:
                errors.append(f"window identity lost: {type(exc).__name__}: {exc}")
                break
            next_identity_check = now + 1.0
        try:
            image = _xwd_to_rgb_image(_capture_xwd(window_id))
            if image.width > 1280:
                height = max(1, round(image.height * (1280 / image.width)))
                image = image.resize((1280, height), Image.Resampling.LANCZOS)
            frames.append(image.copy())
            frame_times.append(time.monotonic() - pulse_start)
            consecutive_capture_errors = 0
        except Exception as exc:
            consecutive_capture_errors += 1
            errors.append(f"frame capture: {type(exc).__name__}: {exc}")
            if consecutive_capture_errors >= 3:
                break
        next_frame += 1.0 / fps
        time.sleep(max(0.0, next_frame - time.monotonic()))

    if frames:
        frames[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=round(1000.0 / fps),
            loop=0,
            optimize=True,
            disposal=2,
        )
    result = {
        "status": "PASS" if len(frames) >= max(2, int(fps * min(duration_s, 5.0))) else "INCOMPLETE",
        "window": verified_window,
        "reset_event": str(reset_path),
        "trace_path": str(trace_path),
        "pulse_start_monotonic_s": pulse_start,
        "requested_fps": fps,
        "requested_duration_s": duration_s,
        "frame_count": len(frames),
        "frame_elapsed_s": frame_times,
        "video_path": str(output_path) if frames else None,
        "errors": errors,
        "limitations": [
            "Passive X11 screen recording; no key or pointer events are sent.",
            "The output is an animated GIF, not an MP4; frame rate and scale are bounded to limit runtime overhead.",
        ],
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--duration-s", type=float, default=11.5)
    parser.add_argument("--wait-timeout-s", type=float, default=180.0)
    parser.add_argument("--output-name", default="mujoco_walk.gif")
    args = parser.parse_args()
    result = record_verified_walk_window(
        Path(args.evidence_dir),
        fps=args.fps,
        duration_s=args.duration_s,
        wait_timeout_s=args.wait_timeout_s,
        output_name=args.output_name,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
