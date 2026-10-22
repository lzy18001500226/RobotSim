#!/usr/bin/env python3
"""One-shot, process-verified Backspace dispatch for the running MuJoCo X11 window."""

from __future__ import annotations

import argparse
import binascii
import ctypes
import ctypes.util
import json
import os
import struct
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IS_VIEWABLE = 2
REVERT_TO_PARENT = 2
CURRENT_TIME = 0
Z_PIXMAP = 2
ALL_PLANES = ctypes.c_ulong(-1).value
CLIENT_MESSAGE = 33
SUBSTRUCTURE_NOTIFY_MASK = 1 << 19
SUBSTRUCTURE_REDIRECT_MASK = 1 << 20


class XClientMessageData(ctypes.Union):
    _fields_ = [
        ("b", ctypes.c_char * 20),
        ("s", ctypes.c_short * 10),
        ("l", ctypes.c_long * 5),
    ]


class XClientMessageEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("message_type", ctypes.c_ulong),
        ("format", ctypes.c_int),
        ("data", XClientMessageData),
    ]


class XEvent(ctypes.Union):
    _fields_ = [
        ("type", ctypes.c_int),
        ("xclient", XClientMessageEvent),
        ("pad", ctypes.c_long * 24),
    ]


def build_net_active_window_event(display: int, window: int, atom: int) -> XEvent:
    """Build the EWMH request used to focus one already-verified viewer window."""
    event = XEvent()
    event.xclient.type = CLIENT_MESSAGE
    event.xclient.serial = 0
    event.xclient.send_event = 1
    event.xclient.display = display
    event.xclient.window = window
    event.xclient.message_type = atom
    event.xclient.format = 32
    event.xclient.data.l[0] = 2  # source indication: pager/window automation
    event.xclient.data.l[1] = CURRENT_TIME
    event.xclient.data.l[2] = 0
    event.xclient.data.l[3] = 0
    event.xclient.data.l[4] = 0
    return event


class XWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("border_width", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("visual", ctypes.c_void_p),
        ("root", ctypes.c_ulong),
        ("window_class", ctypes.c_int),
        ("bit_gravity", ctypes.c_int),
        ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int),
        ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong),
        ("save_under", ctypes.c_int),
        ("colormap", ctypes.c_ulong),
        ("map_installed", ctypes.c_int),
        ("map_state", ctypes.c_int),
        ("all_event_masks", ctypes.c_long),
        ("your_event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int),
        ("screen", ctypes.c_void_p),
    ]


class XImagePrefix(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("xoffset", ctypes.c_int),
        ("format", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("byte_order", ctypes.c_int),
        ("bitmap_unit", ctypes.c_int),
        ("bitmap_bit_order", ctypes.c_int),
        ("bitmap_pad", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("bytes_per_line", ctypes.c_int),
        ("bits_per_pixel", ctypes.c_int),
        ("red_mask", ctypes.c_ulong),
        ("green_mask", ctypes.c_ulong),
        ("blue_mask", ctypes.c_ulong),
    ]


class XClassHint(ctypes.Structure):
    _fields_ = [("res_name", ctypes.c_void_p), ("res_class", ctypes.c_void_p)]


class X11:
    def __init__(self) -> None:
        x11_path = ctypes.util.find_library("X11")
        xtst_path = ctypes.util.find_library("Xtst")
        if not x11_path or not xtst_path:
            raise RuntimeError("libX11 and libXtst are required; no packages were installed")
        self.lib = ctypes.CDLL(x11_path)
        self.xtst = ctypes.CDLL(xtst_path)
        self._declare_functions()
        self.display = self.lib.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError(f"cannot open DISPLAY={os.environ.get('DISPLAY')!r}")
        self.root = int(self.lib.XDefaultRootWindow(self.display))
        opcode, event, error = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        if not self.lib.XQueryExtension(
            self.display,
            b"XTEST",
            ctypes.byref(opcode),
            ctypes.byref(event),
            ctypes.byref(error),
        ):
            self.close()
            raise RuntimeError("XTEST extension is not available on the current display")

    def _declare_functions(self) -> None:
        self.lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.lib.XOpenDisplay.restype = ctypes.c_void_p
        self.lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self.lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        self.lib.XDefaultRootWindow.restype = ctypes.c_ulong
        self.lib.XQueryExtension.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.XQueryExtension.restype = ctypes.c_int
        self.lib.XQueryTree.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.XQueryTree.restype = ctypes.c_int
        self.lib.XQueryPointer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.XQueryPointer.restype = ctypes.c_int
        self.lib.XGetWindowAttributes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(XWindowAttributes),
        ]
        self.lib.XGetWindowAttributes.restype = ctypes.c_int
        self.lib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.lib.XInternAtom.restype = ctypes.c_ulong
        self.lib.XGetWindowProperty.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
        ]
        self.lib.XGetWindowProperty.restype = ctypes.c_int
        self.lib.XFetchName.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.XFetchName.restype = ctypes.c_int
        self.lib.XGetClassHint.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(XClassHint),
        ]
        self.lib.XGetClassHint.restype = ctypes.c_int
        self.lib.XFree.argtypes = [ctypes.c_void_p]
        self.lib.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.lib.XTranslateCoordinates.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong),
        ]
        self.lib.XTranslateCoordinates.restype = ctypes.c_int
        self.lib.XSendEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_long,
            ctypes.POINTER(XEvent),
        ]
        self.lib.XSendEvent.restype = ctypes.c_int
        self.lib.XSetInputFocus.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self.lib.XGetInputFocus.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.XFlush.argtypes = [ctypes.c_void_p]
        self.lib.XStringToKeysym.argtypes = [ctypes.c_char_p]
        self.lib.XStringToKeysym.restype = ctypes.c_ulong
        self.lib.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.lib.XKeysymToKeycode.restype = ctypes.c_uint
        self.xtst.XTestFakeKeyEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self.xtst.XTestFakeKeyEvent.restype = ctypes.c_int
        self.xtst.XTestFakeMotionEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self.xtst.XTestFakeMotionEvent.restype = ctypes.c_int
        self.xtst.XTestFakeButtonEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self.xtst.XTestFakeButtonEvent.restype = ctypes.c_int
        self.lib.XGetImage.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_ulong,
            ctypes.c_int,
        ]
        self.lib.XGetImage.restype = ctypes.POINTER(XImagePrefix)
        self.lib.XGetPixel.argtypes = [
            ctypes.POINTER(XImagePrefix),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.XGetPixel.restype = ctypes.c_ulong
        self.lib.XDestroyImage.argtypes = [ctypes.POINTER(XImagePrefix)]

    def close(self) -> None:
        display = getattr(self, "display", None)
        if display:
            self.lib.XCloseDisplay(display)
            self.display = None

    def atom(self, name: str) -> int:
        return int(self.lib.XInternAtom(self.display, name.encode(), 1))

    def property_bytes(self, window: int, name: str, length: int = 2048) -> tuple[int, int, bytes]:
        atom = self.atom(name)
        if atom == 0:
            return 0, 0, b""
        actual_type = ctypes.c_ulong()
        actual_format = ctypes.c_int()
        nitems = ctypes.c_ulong()
        bytes_after = ctypes.c_ulong()
        data = ctypes.POINTER(ctypes.c_ubyte)()
        status = self.lib.XGetWindowProperty(
            self.display,
            window,
            atom,
            0,
            length,
            0,
            0,
            ctypes.byref(actual_type),
            ctypes.byref(actual_format),
            ctypes.byref(nitems),
            ctypes.byref(bytes_after),
            ctypes.byref(data),
        )
        if status != 0 or not data:
            return 0, 0, b""
        if actual_format.value == 8:
            value = ctypes.string_at(data, nitems.value)
        elif actual_format.value == 32:
            value = b"".join(
                int(ctypes.cast(data, ctypes.POINTER(ctypes.c_ulong))[i]).to_bytes(
                    ctypes.sizeof(ctypes.c_ulong), byteorder=sys.byteorder
                )
                for i in range(nitems.value)
            )
        else:
            value = b""
        self.lib.XFree(data)
        return int(actual_type.value), int(actual_format.value), value

    def window_pid(self, window: int) -> int | None:
        _, fmt, value = self.property_bytes(window, "_NET_WM_PID", length=1)
        if fmt != 32 or not value:
            return None
        return int.from_bytes(value[: ctypes.sizeof(ctypes.c_ulong)], byteorder=sys.byteorder)

    def title(self, window: int) -> str:
        _, fmt, value = self.property_bytes(window, "_NET_WM_NAME")
        if fmt == 8 and value:
            return value.decode("utf-8", errors="replace").rstrip("\0")
        name = ctypes.c_void_p()
        if self.lib.XFetchName(self.display, window, ctypes.byref(name)) and name.value:
            try:
                return ctypes.string_at(name.value).decode("utf-8", errors="replace")
            finally:
                self.lib.XFree(name)
        return ""

    def wm_class(self, window: int) -> dict[str, str]:
        hint = XClassHint()
        if not self.lib.XGetClassHint(self.display, window, ctypes.byref(hint)):
            return {"res_name": "", "res_class": ""}
        try:
            return {
                "res_name": ctypes.string_at(hint.res_name).decode("utf-8", errors="replace")
                if hint.res_name else "",
                "res_class": ctypes.string_at(hint.res_class).decode("utf-8", errors="replace")
                if hint.res_class else "",
            }
        finally:
            if hint.res_name:
                self.lib.XFree(hint.res_name)
            if hint.res_class:
                self.lib.XFree(hint.res_class)

    def attributes(self, window: int) -> XWindowAttributes | None:
        attrs = XWindowAttributes()
        if not self.lib.XGetWindowAttributes(self.display, window, ctypes.byref(attrs)):
            return None
        return attrs

    def children(self, parent: int) -> list[int]:
        root_ret, parent_ret = ctypes.c_ulong(), ctypes.c_ulong()
        child_ptr = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        ok = self.lib.XQueryTree(
            self.display,
            parent,
            ctypes.byref(root_ret),
            ctypes.byref(parent_ret),
            ctypes.byref(child_ptr),
            ctypes.byref(count),
        )
        if not ok or not child_ptr:
            return []
        try:
            return [int(child_ptr[i]) for i in range(count.value)]
        finally:
            self.lib.XFree(child_ptr)

    def list_windows(self, max_depth: int = 4) -> list[dict[str, Any]]:
        seen: set[int] = set()
        result: list[dict[str, Any]] = []

        def walk(parent: int, depth: int) -> None:
            if depth > max_depth:
                return
            for window in self.children(parent):
                if window in seen:
                    continue
                seen.add(window)
                attrs = self.attributes(window)
                if attrs:
                    title = self.title(window)
                    pid = self.window_pid(window)
                    if title or pid is not None:
                        result.append(
                            {
                                "window_id": window,
                                "pid": pid,
                                "title": title,
                                "wm_class": self.wm_class(window),
                                "x": int(attrs.x),
                                "y": int(attrs.y),
                                "width": int(attrs.width),
                                "height": int(attrs.height),
                                "depth": int(attrs.depth),
                                "map_state": int(attrs.map_state),
                                "viewable": int(attrs.map_state) == IS_VIEWABLE,
                            }
                        )
                walk(window, depth + 1)

        walk(self.root, 0)
        return result

    @staticmethod
    def _channel(pixel: int, mask: int) -> int:
        if mask == 0:
            return 0
        shift = (mask & -mask).bit_length() - 1
        max_value = mask >> shift
        value = (pixel & mask) >> shift
        return int(value * 255 / max_value) if max_value else 0

    def capture_png(self, window: int, output: Path) -> dict[str, Any]:
        attrs = self.attributes(window)
        if not attrs or attrs.width <= 0 or attrs.height <= 0:
            return {"status": "UNAVAILABLE", "reason": "window has no drawable geometry"}
        image = self.lib.XGetImage(
            self.display,
            window,
            0,
            0,
            attrs.width,
            attrs.height,
            ALL_PLANES,
            Z_PIXMAP,
        )
        if not image:
            return {"status": "UNAVAILABLE", "reason": "XGetImage returned null"}
        try:
            im = image.contents
            rows = bytearray()
            for y in range(im.height):
                rows.append(0)
                for x in range(im.width):
                    pixel = int(self.lib.XGetPixel(image, x, y))
                    rows.extend(
                        (
                            self._channel(pixel, int(im.red_mask)),
                            self._channel(pixel, int(im.green_mask)),
                            self._channel(pixel, int(im.blue_mask)),
                        )
                    )
            raw = bytes(rows)
            def chunk(tag: bytes, payload: bytes) -> bytes:
                return (
                    struct.pack(">I", len(payload))
                    + tag
                    + payload
                    + struct.pack(">I", binascii.crc32(tag + payload) & 0xFFFFFFFF)
                )
            png = b"\x89PNG\r\n\x1a\n"
            png += chunk(b"IHDR", struct.pack(">IIBBBBB", im.width, im.height, 8, 2, 0, 0, 0))
            png += chunk(b"IDAT", zlib.compress(raw, level=5))
            png += chunk(b"IEND", b"")
            with output.open("xb") as stream:
                stream.write(png)
            return {"status": "CAPTURED", "path": str(output), "width": im.width, "height": im.height}
        finally:
            self.lib.XDestroyImage(image)

    def _input_focus(self) -> int:
        focus = ctypes.c_ulong()
        revert = ctypes.c_int()
        self.lib.XGetInputFocus(self.display, ctypes.byref(focus), ctypes.byref(revert))
        return int(focus.value)

    def _wait_for_input_focus(self, window: int, timeout_s: float) -> int:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            focused = self._input_focus()
            if focused == window or time.monotonic() >= deadline:
                return focused
            time.sleep(0.02)

    def _target_identity(self, pid: int, title: str, window: int) -> dict[str, Any] | None:
        """Return the unique visible process/title/XID match, or fail closed."""
        matches = [
            item for item in self.list_windows()
            if item["pid"] == pid
            and item["title"] == title
            and item["window_id"] == window
            and item["viewable"]
        ]
        return matches[0] if len(matches) == 1 else None

    def activate_window(self, window: int, timeout_s: float = 1.0) -> dict[str, Any]:
        """Best-effort EWMH request; focus is determined only by XGetInputFocus."""
        atom = int(self.lib.XInternAtom(self.display, b"_NET_ACTIVE_WINDOW", 0))
        if not atom:
            focus = self._input_focus()
            return {
                "status": "EWMH_UNAVAILABLE",
                "target_window_id": window,
                "focus_before": focus,
                "focus_after": focus,
                "send_event_return": None,
                "reason": "_NET_ACTIVE_WINDOW atom is unavailable",
            }
        event = build_net_active_window_event(self.display, window, atom)
        mask = SUBSTRUCTURE_REDIRECT_MASK | SUBSTRUCTURE_NOTIFY_MASK
        focus_before = self._input_focus()
        sent = int(self.lib.XSendEvent(self.display, self.root, 0, mask, ctypes.byref(event)))
        self.lib.XFlush(self.display)
        self.lib.XSync(self.display, 0)
        focus_after = self._wait_for_input_focus(window, timeout_s)
        result = {
            "status": "FOCUSED" if focus_after == window else "FOCUS_NOT_CONFIRMED",
            "target_window_id": window,
            "focus_before": focus_before,
            "focus_after": focus_after,
            "send_event_return": sent,
            "send_event_return_is_focus_proof": False,
        }
        return result

    def focus_window_direct(self, window: int, timeout_s: float = 0.5) -> dict[str, Any]:
        """Try the narrowly scoped XRaiseWindow/XSetInputFocus path and verify it."""
        self.lib.XRaiseWindow(self.display, window)
        self.lib.XSync(self.display, 0)
        self.lib.XSetInputFocus(self.display, window, REVERT_TO_PARENT, CURRENT_TIME)
        self.lib.XSync(self.display, 0)
        focus_after = self._wait_for_input_focus(window, timeout_s)
        return {
            "status": "FOCUSED" if focus_after == window else "FOCUS_NOT_CONFIRMED",
            "target_window_id": window,
            "focus_after": focus_after,
            "xraise_attempted": True,
            "xsetinputfocus_attempted": True,
            "xsync_after_each_request": True,
        }

    def focus_window_with_neutral_click(
        self,
        window: int,
        expected_pid: int,
        expected_title: str,
        timeout_s: float = 0.5,
    ) -> dict[str, Any]:
        """Try one neutral click in the verified viewer, then verify actual X focus."""
        identity_before = self._target_identity(expected_pid, expected_title, window)
        if identity_before is None:
            return {
                "status": "FOCUS_NOT_CONFIRMED",
                "reason": "exact process/title/window/viewability match lost before pointer focus",
                "xtest_pointer_events_sent": 0,
                "xtest_button_events_sent": 0,
            }
        width, height = int(identity_before["width"]), int(identity_before["height"])
        if width < 32 or height < 32:
            return {
                "status": "FOCUS_NOT_CONFIRMED",
                "reason": "verified viewer geometry is too small for a neutral focus click",
                "xtest_pointer_events_sent": 0,
                "xtest_button_events_sent": 0,
            }
        # Use a point inset from the bottom-right edge of the MuJoCo viewport;
        # a press/release focuses/selects only and never drags or changes qpos.
        local_x, local_y = width - 16, height - 16
        root_x, root_y, child = ctypes.c_int(), ctypes.c_int(), ctypes.c_ulong()
        translated = int(self.lib.XTranslateCoordinates(
            self.display, window, self.root, local_x, local_y,
            ctypes.byref(root_x), ctypes.byref(root_y), ctypes.byref(child),
        ))
        if not translated:
            return {
                "status": "FOCUS_NOT_CONFIRMED",
                "reason": "could not translate the neutral viewer point to root coordinates",
                "xtest_pointer_events_sent": 0,
                "xtest_button_events_sent": 0,
            }
        move_return = int(self.xtst.XTestFakeMotionEvent(
            self.display, 0, root_x.value, root_y.value, CURRENT_TIME,
        ))
        self.lib.XFlush(self.display)
        self.lib.XSync(self.display, 0)
        identity_before_click = self._target_identity(expected_pid, expected_title, window)
        if identity_before_click is None:
            return {
                "status": "FOCUS_NOT_CONFIRMED",
                "reason": "exact process/title/window/viewability match lost before pointer click",
                "local_point": [local_x, local_y],
                "root_point": [root_x.value, root_y.value],
                "xtest_pointer_events_sent": int(bool(move_return)),
                "xtest_button_events_sent": 0,
            }
        root_return, child_return = ctypes.c_ulong(), ctypes.c_ulong()
        root_x_return, root_y_return = ctypes.c_int(), ctypes.c_int()
        win_x_return, win_y_return = ctypes.c_int(), ctypes.c_int()
        mask_return = ctypes.c_uint()
        pointer_inside = bool(self.lib.XQueryPointer(
            self.display,
            window,
            ctypes.byref(root_return),
            ctypes.byref(child_return),
            ctypes.byref(root_x_return),
            ctypes.byref(root_y_return),
            ctypes.byref(win_x_return),
            ctypes.byref(win_y_return),
            ctypes.byref(mask_return),
        ))
        pointer_verified = bool(
            move_return
            and pointer_inside
            and 0 <= win_x_return.value < width
            and 0 <= win_y_return.value < height
        )
        pointer_verification = {
            "status": "POINTER_VERIFIED_INSIDE_TARGET" if pointer_verified else "POINTER_NOT_VERIFIED",
            "query_return": int(pointer_inside),
            "root_point": [root_x_return.value, root_y_return.value],
            "window_point": [win_x_return.value, win_y_return.value],
            "child_window_id": int(child_return.value),
            "mask": int(mask_return.value),
        }
        if not pointer_verified:
            return {
                "status": "FOCUS_NOT_CONFIRMED",
                "reason": "XQueryPointer did not verify the pointer inside the exact target window; no click sent",
                "local_point": [local_x, local_y],
                "root_point": [root_x.value, root_y.value],
                "pointer_verification": pointer_verification,
                "xtest_pointer_events_sent": int(bool(move_return)),
                "xtest_button_events_sent": 0,
            }
        press_return = int(self.xtst.XTestFakeButtonEvent(self.display, 1, 1, CURRENT_TIME))
        release_return = int(self.xtst.XTestFakeButtonEvent(self.display, 1, 0, CURRENT_TIME))
        self.lib.XFlush(self.display)
        self.lib.XSync(self.display, 0)
        focus_after = self._wait_for_input_focus(window, timeout_s)
        identity_after = self._target_identity(expected_pid, expected_title, window)
        return {
            "status": (
                "FOCUSED"
                if identity_after is not None and focus_after == window and press_return and release_return
                else "FOCUS_NOT_CONFIRMED"
            ),
            "target_window_id": window,
            "identity_before": identity_before,
            "identity_before_click": identity_before_click,
            "identity_after": identity_after,
            "local_point": [local_x, local_y],
            "root_point": [root_x.value, root_y.value],
            "focus_after": focus_after,
            "pointer_verification": pointer_verification,
            "xtest_pointer_return": move_return,
            "xtest_button_press_return": press_return,
            "xtest_button_release_return": release_return,
            "xtest_pointer_events_sent": 1 if move_return else 0,
            "xtest_button_events_sent": int(bool(press_return)) + int(bool(release_return)),
            "return_values_are_focus_proof": False,
        }

    def send_backspace_once(
        self,
        window: int,
        expected_pid: int,
        expected_title: str,
        *,
        ewmh_timeout_s: float = 1.0,
        direct_timeout_s: float = 0.5,
    ) -> dict[str, Any]:
        """Send one key only after rechecking exact identity and verified X focus."""
        identity_before = self._target_identity(expected_pid, expected_title, window)
        if identity_before is None:
            return {
                "status": "RESET_NOT_DELIVERED",
                "reason": "exact process/title/window/viewability match lost before focus",
                "target_window_id": window,
                "expected_pid": expected_pid,
                "expected_title": expected_title,
                "xtest_key_events_sent": 0,
            }
        key_sym = int(self.lib.XStringToKeysym(b"BackSpace"))
        keycode = int(self.lib.XKeysymToKeycode(self.display, key_sym))
        if not key_sym or not keycode:
            raise RuntimeError("BackSpace keysym is not mapped on this X server")
        focus_before = self._input_focus()
        activation = self.activate_window(window, timeout_s=ewmh_timeout_s)
        direct_focus = {"status": "NOT_NEEDED", "target_window_id": window}
        if activation["status"] != "FOCUSED":
            direct_focus = self.focus_window_direct(window, timeout_s=direct_timeout_s)

        pointer_focus = {"status": "NOT_NEEDED", "target_window_id": window}
        identity_ready = self._target_identity(expected_pid, expected_title, window)
        self.lib.XSync(self.display, 0)
        focus_after = self._input_focus()
        if identity_ready is not None and focus_after != window:
            pointer_focus = self.focus_window_with_neutral_click(
                window, expected_pid, expected_title, timeout_s=direct_timeout_s,
            )

        # Recheck identity and focus after all activation requests, immediately
        # before synthesizing a key. EWMH/XTest return values are not proof of
        # focus; only the exact XID returned by XGetInputFocus is accepted.
        identity_ready = self._target_identity(expected_pid, expected_title, window)
        self.lib.XSync(self.display, 0)
        focus_after = self._input_focus()
        final_focus_reassertion = {"status": "NOT_NEEDED"}
        if identity_ready is not None and focus_after != window:
            # WSLg can revoke focus asynchronously after the first direct
            # request. One bounded reassertion is allowed, never another key.
            final_focus_reassertion = self.focus_window_direct(
                window, timeout_s=direct_timeout_s
            )
            identity_ready = self._target_identity(expected_pid, expected_title, window)
            self.lib.XSync(self.display, 0)
            focus_after = self._input_focus()
        if identity_ready is None or focus_after != window:
            return {
                "status": "RESET_NOT_DELIVERED",
                "reason": "target identity or XGetInputFocus verification failed; no key sent",
                "target_window_id": window,
                "expected_pid": expected_pid,
                "expected_title": expected_title,
                "identity_before": identity_before,
                "identity_before_dispatch": identity_ready,
                "ewmh_activation": activation,
                "direct_focus": direct_focus,
                "pointer_focus": pointer_focus,
                "final_focus_reassertion": final_focus_reassertion,
                "focus_before": focus_before,
                "focus_after": focus_after,
                "xtest_key_events_sent": 0,
            }
        dispatch_utc = datetime.now(timezone.utc).isoformat()
        dispatch_monotonic_ns = time.monotonic_ns()
        press_ok = int(self.xtst.XTestFakeKeyEvent(self.display, keycode, 1, CURRENT_TIME))
        release_ok = int(self.xtst.XTestFakeKeyEvent(self.display, keycode, 0, CURRENT_TIME))
        self.lib.XFlush(self.display)
        self.lib.XSync(self.display, 0)
        time.sleep(0.05)
        restored = None
        if focus_before not in (0, 1, window):
            self.lib.XSetInputFocus(
                self.display, focus_before, REVERT_TO_PARENT, CURRENT_TIME
            )
            self.lib.XSync(self.display, 0)
            focus_restored = ctypes.c_ulong()
            revert_restored = ctypes.c_int()
            self.lib.XGetInputFocus(
                self.display, ctypes.byref(focus_restored), ctypes.byref(revert_restored)
            )
            restored = int(focus_restored.value)
        if not press_ok or not release_ok:
            raise RuntimeError(
                f"XTest rejected key event: press={press_ok}, release={release_ok}"
            )
        return {
            "status": "KEY_DISPATCHED",
            "key": "BackSpace",
            "keycode": keycode,
            "target_window_id": window,
            "expected_pid": expected_pid,
            "expected_title": expected_title,
            "identity_before": identity_before,
            "identity_before_dispatch": identity_ready,
            "ewmh_activation": activation,
            "direct_focus": direct_focus,
            "pointer_focus": pointer_focus,
            "final_focus_reassertion": final_focus_reassertion,
            "focus_before": focus_before,
            "focus_verified_target": focus_after,
            "focus_restored": restored,
            "press_return": press_ok,
            "release_return": release_ok,
            "xtest_key_events_sent": 2,
            "dispatch_wall_utc": dispatch_utc,
            "dispatch_monotonic_ns": dispatch_monotonic_ns,
            "wall_utc": datetime.now(timezone.utc).isoformat(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect", help="list only visible X windows for an expected PID")
    inspect_parser.add_argument("--expected-pid", type=int, required=True)
    capture_parser = sub.add_parser("capture", help="capture one exact visible MuJoCo window")
    capture_parser.add_argument("--expected-pid", type=int, required=True)
    capture_parser.add_argument("--expected-title", required=True)
    capture_parser.add_argument("--window-id", type=int, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    send_parser = sub.add_parser("send-reset", help="send exactly one BackSpace to a verified window")
    send_parser.add_argument("--expected-pid", type=int, required=True)
    send_parser.add_argument("--expected-title", required=True)
    send_parser.add_argument("--window-id", type=int, required=True)
    send_parser.add_argument("--before-png", type=Path)
    send_parser.add_argument("--after-png", type=Path)
    send_parser.add_argument(
        "--dispatch-event-json",
        type=Path,
        help="persist the verified key event immediately, before slow post-event screenshot capture",
    )
    args = parser.parse_args()
    x11 = None
    try:
        x11 = X11()
        if args.command == "inspect":
            matches = [
                w for w in x11.list_windows()
                if w["pid"] == args.expected_pid and w["viewable"]
            ]
            print(json.dumps({"status": "INSPECTED", "display": os.environ.get("DISPLAY"), "expected_pid": args.expected_pid, "windows": matches}, indent=2))
            return 0
        if args.command == "capture":
            matches = [
                w for w in x11.list_windows()
                if w["pid"] == args.expected_pid
                and w["title"] == args.expected_title
                and w["window_id"] == args.window_id
                and w["viewable"]
            ]
            if len(matches) != 1:
                print(json.dumps({"status": "NOT_CAPTURED", "reason": "exact process/title/window/viewability match required", "expected_pid": args.expected_pid, "expected_title": args.expected_title, "expected_window_id": args.window_id, "matches": matches}, indent=2))
                return 2
            image = x11.capture_png(args.window_id, args.output)
            print(json.dumps({"status": image["status"], "window": matches[0], "image": image}, indent=2))
            return 0
        matches = [
            w for w in x11.list_windows()
            if w["pid"] == args.expected_pid
            and w["title"] == args.expected_title
            and w["window_id"] == args.window_id
            and w["viewable"]
        ]
        if len(matches) != 1:
            print(json.dumps({"status": "RESET_NOT_DELIVERED", "reason": "exact process/title/window/viewability match required", "expected_pid": args.expected_pid, "expected_title": args.expected_title, "expected_window_id": args.window_id, "matches": matches}, indent=2))
            return 2
        window = matches[0]
        before = x11.capture_png(args.window_id, args.before_png) if args.before_png else {"status": "NOT_REQUESTED"}
        event = x11.send_backspace_once(
            args.window_id,
            expected_pid=args.expected_pid,
            expected_title=args.expected_title,
        )
        event_record = {
            "schema": "robotsim.issue80.reset-dispatch-event.v1",
            "window": window,
            "before_image": before,
            "event": event,
            "classification": (
                "KEY_DISPATCH_ONLY_STATE_CHANGE_MUST_BE_VERIFIED_BY_ROS_TRACE"
                if event.get("status") == "KEY_DISPATCHED"
                else "FOCUS_OR_IDENTITY_BLOCKED_NO_KEY_SENT"
            ),
        }
        if args.dispatch_event_json:
            args.dispatch_event_json.parent.mkdir(parents=True, exist_ok=True)
            with args.dispatch_event_json.open("x", encoding="utf-8") as stream:
                json.dump(event_record, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        if event.get("status") != "KEY_DISPATCHED":
            print(json.dumps({**event_record, "after_image": {"status": "NOT_CAPTURED_KEY_NOT_SENT"}}, indent=2))
            return 2
        after = x11.capture_png(args.window_id, args.after_png) if args.after_png else {"status": "NOT_REQUESTED"}
        print(json.dumps({**event_record, "after_image": after}, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "RESET_NOT_DELIVERED", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    finally:
        if x11:
            x11.close()


if __name__ == "__main__":
    raise SystemExit(main())
