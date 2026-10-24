"""Offline safety tests for process-verified native MuJoCo Reset input paths."""

from __future__ import annotations

import ctypes
import unittest

from issue80_mujoco_x11_reset import CLIENT_MESSAGE, X11, build_net_active_window_event


class FakeXlib:
    def __init__(self, *, accept_direct_focus: bool):
        self.focus = 1  # X PointerRoot; deliberately not the target XID.
        self.accept_direct_focus = accept_direct_focus
        self.calls: list[tuple[object, ...]] = []
        self.pointer_root = (0, 0)

    def XInternAtom(self, _display: int, _name: bytes, _only_if_exists: int) -> int:
        return 0xABC

    def XGetInputFocus(self, _display: int, focus_ptr: object, revert_ptr: object) -> int:
        ctypes.cast(focus_ptr, ctypes.POINTER(ctypes.c_ulong))[0] = self.focus
        ctypes.cast(revert_ptr, ctypes.POINTER(ctypes.c_int))[0] = 2
        self.calls.append(("get-focus", self.focus))
        return 1

    def XTranslateCoordinates(
        self, _display: int, _source: int, _target: int, src_x: int, src_y: int,
        dest_x_ptr: object, dest_y_ptr: object, child_ptr: object,
    ) -> int:
        ctypes.cast(dest_x_ptr, ctypes.POINTER(ctypes.c_int))[0] = src_x + 50
        ctypes.cast(dest_y_ptr, ctypes.POINTER(ctypes.c_int))[0] = src_y + 70
        ctypes.cast(child_ptr, ctypes.POINTER(ctypes.c_ulong))[0] = 0
        self.calls.append(("translate", src_x, src_y))
        return 1

    def XQueryPointer(
        self, _display: int, _window: int, root_ptr: object, child_ptr: object,
        root_x_ptr: object, root_y_ptr: object, win_x_ptr: object, win_y_ptr: object,
        mask_ptr: object,
    ) -> int:
        root_x, root_y = self.pointer_root
        for ptr, value, ctype in (
            (root_ptr, 0x1000, ctypes.c_ulong),
            (child_ptr, 0, ctypes.c_ulong),
            (root_x_ptr, root_x, ctypes.c_int),
            (root_y_ptr, root_y, ctypes.c_int),
            (win_x_ptr, root_x - 50, ctypes.c_int),
            (win_y_ptr, root_y - 70, ctypes.c_int),
            (mask_ptr, 0, ctypes.c_uint),
        ):
            ctypes.cast(ptr, ctypes.POINTER(ctype))[0] = value
        self.calls.append(("query-pointer", root_x, root_y))
        return int(50 <= root_x < 1756 and 70 <= root_y < 1136)

    def XSendEvent(self, _display: int, _root: int, _propagate: int, _mask: int, _event: object) -> int:
        self.calls.append(("ewmh-send",))
        # Like XSendEvent, this confirms request delivery to the X server only.
        # The fake window manager does not activate the target as a side effect.
        return 1

    def XFlush(self, _display: int) -> int:
        return 1

    def XSync(self, _display: int, _discard: int) -> int:
        self.calls.append(("sync",))
        return 1

    def XRaiseWindow(self, _display: int, window: int) -> int:
        self.calls.append(("raise", window))
        return 1

    def XSetInputFocus(self, _display: int, window: int, _revert: int, _time: int) -> int:
        self.calls.append(("set-focus", window))
        if self.accept_direct_focus:
            self.focus = window
        return 1

    def XStringToKeysym(self, _name: bytes) -> int:
        return 0xFF08

    def XKeysymToKeycode(self, _display: int, _key_sym: int) -> int:
        return 22


class FakeXTest:
    def __init__(self, on_button=None, on_motion=None):
        self.events: list[tuple[int, int]] = []
        self.pointer_events: list[tuple[object, ...]] = []
        self.on_button = on_button
        self.on_motion = on_motion

    def XTestFakeKeyEvent(self, _display: int, keycode: int, pressed: int, _time: int) -> int:
        self.events.append((keycode, pressed))
        return 1

    def XTestFakeMotionEvent(self, _display: int, screen: int, x: int, y: int, _time: int) -> int:
        self.pointer_events.append(("motion", screen, x, y))
        if self.on_motion:
            self.on_motion(x, y)
        return 1

    def XTestFakeButtonEvent(self, _display: int, button: int, pressed: int, _time: int) -> int:
        self.pointer_events.append(("button", button, pressed))
        if self.on_button:
            self.on_button(button, pressed)
        return 1


def make_x11(
    *, accept_direct_focus: bool, accept_pointer_focus: bool = False, window: int = 0x5678,
) -> X11:
    x11 = X11.__new__(X11)
    x11.display = 0x1234
    x11.root = 0x1000
    x11.lib = FakeXlib(accept_direct_focus=accept_direct_focus)
    x11.xtst = FakeXTest(
        on_button=(
            lambda button, pressed: setattr(x11.lib, "focus", window)
            if accept_pointer_focus and button == 1 and pressed else None
        ),
        on_motion=lambda x, y: setattr(x11.lib, "pointer_root", (x, y)),
    )
    x11.list_windows = lambda: [
        {
            "window_id": window,
            "pid": 4321,
            "title": "MuJoCo : X2 simulator",
            "viewable": True,
            "width": 1706,
            "height": 1066,
        }
    ]
    return x11


class EwmhActivationEventTests(unittest.TestCase):
    def test_active_window_event_targets_only_requested_window(self) -> None:
        event = build_net_active_window_event(display=0x1234, window=0x5678, atom=0x9ABC)
        self.assertEqual(event.xclient.type, CLIENT_MESSAGE)
        self.assertEqual(event.xclient.send_event, 1)
        self.assertEqual(event.xclient.display, 0x1234)
        self.assertEqual(event.xclient.window, 0x5678)
        self.assertEqual(event.xclient.message_type, 0x9ABC)
        self.assertEqual(event.xclient.format, 32)
        self.assertEqual(event.xclient.data.l[0], 2)
        self.assertEqual(event.xclient.data.l[1], 0)


class VerifiedFocusDispatchTests(unittest.TestCase):
    def test_ewmh_send_return_one_does_not_claim_focus(self) -> None:
        x11 = make_x11(accept_direct_focus=False)

        result = x11.activate_window(0x5678, timeout_s=0.0)

        self.assertEqual(result["status"], "FOCUS_NOT_CONFIRMED")
        self.assertEqual(result["send_event_return"], 1)
        self.assertFalse(result["send_event_return_is_focus_proof"])

    def test_direct_focus_fallback_runs_after_ewmh_and_allows_key_only_when_verified(self) -> None:
        x11 = make_x11(accept_direct_focus=True)

        result = x11.send_backspace_once(
            0x5678,
            expected_pid=4321,
            expected_title="MuJoCo : X2 simulator",
            ewmh_timeout_s=0.0,
            direct_timeout_s=0.0,
        )

        self.assertEqual(result["status"], "KEY_DISPATCHED")
        self.assertEqual(result["focus_verified_target"], 0x5678)
        self.assertEqual(x11.xtst.events, [(22, 1), (22, 0)])
        names = [call[0] for call in x11.lib.calls]
        self.assertLess(names.index("ewmh-send"), names.index("raise"))
        self.assertLess(names.index("raise"), names.index("set-focus"))
        self.assertEqual(result["identity_before_dispatch"]["pid"], 4321)

    def test_no_key_is_sent_when_ewmh_and_direct_focus_both_fail(self) -> None:
        x11 = make_x11(accept_direct_focus=False)

        result = x11.send_backspace_once(
            0x5678,
            expected_pid=4321,
            expected_title="MuJoCo : X2 simulator",
            ewmh_timeout_s=0.0,
            direct_timeout_s=0.0,
        )

        self.assertEqual(result["status"], "RESET_NOT_DELIVERED")
        self.assertIn("no key sent", result["reason"])
        self.assertEqual(result["xtest_key_events_sent"], 0)
        self.assertEqual(x11.xtst.events, [])
        self.assertEqual(result["pointer_focus"]["status"], "FOCUS_NOT_CONFIRMED")

    def test_native_reset_button_click_needs_verified_pointer_and_not_keyboard_focus(self) -> None:
        x11 = make_x11(accept_direct_focus=False)

        result = x11.click_native_reset_button_once(
            0x5678,
            expected_pid=4321,
            expected_title="MuJoCo : X2 simulator",
        )

        self.assertEqual(result["status"], "RESET_BUTTON_CLICKED")
        self.assertEqual(result["input_path"], "MUJOCO_NATIVE_SIMULATION_RESET_BUTTON")
        self.assertEqual(result["pointer_verification"]["local_point"], [68, 456])
        self.assertEqual(x11.xtst.events, [])
        self.assertEqual(
            [event for event in x11.xtst.pointer_events if event[0] == "button"],
            [("button", 1, 1), ("button", 1, 0)],
        )
        self.assertEqual(result["key_events_sent"], 0)

    def test_native_reset_button_refuses_uncalibrated_window_geometry(self) -> None:
        x11 = make_x11(accept_direct_focus=False)
        x11.list_windows = lambda: [{
            "window_id": 0x5678,
            "pid": 4321,
            "title": "MuJoCo : X2 simulator",
            "viewable": True,
            "width": 1280,
            "height": 720,
        }]

        result = x11.click_native_reset_button_once(
            0x5678,
            expected_pid=4321,
            expected_title="MuJoCo : X2 simulator",
        )

        self.assertEqual(result["status"], "RESET_NOT_DELIVERED")
        self.assertIn("calibrated native Reset button layout", result["reason"])
        self.assertEqual(x11.xtst.pointer_events, [])
        self.assertEqual(x11.xtst.events, [])

    def test_transient_focus_loss_is_reasserted_but_still_verified(self) -> None:
        x11 = make_x11(accept_direct_focus=True)
        matching = x11.list_windows()[0]
        count = [0]
        def identities():
            count[0] += 1
            if count[0] == 2:
                x11.lib.focus = 0  # WSLg revokes initial direct focus
            return [matching]
        x11.list_windows = identities
        result = x11.send_backspace_once(
            0x5678, expected_pid=4321, expected_title="MuJoCo : X2 simulator",
            ewmh_timeout_s=0.0, direct_timeout_s=0.0,
        )
        self.assertEqual(result["status"], "KEY_DISPATCHED")
        self.assertEqual(result["final_focus_reassertion"]["status"], "FOCUSED")
        self.assertEqual(result["focus_verified_target"], 0x5678)
        self.assertEqual(x11.xtst.events, [(22, 1), (22, 0)])
        self.assertGreater(count[0], 3)

    def test_verified_neutral_viewer_click_can_establish_focus_before_key(self) -> None:
        x11 = make_x11(accept_direct_focus=False, accept_pointer_focus=True)
        result = x11.send_backspace_once(
            0x5678,
            expected_pid=4321,
            expected_title="MuJoCo : X2 simulator",
            ewmh_timeout_s=0.0,
            direct_timeout_s=0.0,
        )
        self.assertEqual(result["status"], "KEY_DISPATCHED")
        self.assertEqual(result["pointer_focus"]["status"], "FOCUSED")
        self.assertEqual(result["focus_verified_target"], 0x5678)
        self.assertEqual(x11.xtst.events, [(22, 1), (22, 0)])
        self.assertEqual(x11.xtst.pointer_events[-2:], [("button", 1, 1), ("button", 1, 0)])

    def test_pointer_click_is_not_sent_after_window_identity_changes(self) -> None:
        x11 = make_x11(accept_direct_focus=False)
        matching = x11.list_windows()[0]
        count = [0]
        def identities():
            count[0] += 1
            return [matching] if count[0] <= 2 else []
        x11.list_windows = identities
        result = x11.send_backspace_once(
            0x5678, expected_pid=4321, expected_title="MuJoCo : X2 simulator",
            ewmh_timeout_s=0.0, direct_timeout_s=0.0,
        )
        self.assertEqual(result["status"], "RESET_NOT_DELIVERED")
        self.assertEqual(result["xtest_key_events_sent"], 0)
        self.assertFalse(any(event[0] == "button" for event in x11.xtst.pointer_events))
        self.assertEqual(x11.xtst.events, [])

    def test_identity_mismatch_blocks_focus_and_key_dispatch(self) -> None:
        x11 = make_x11(accept_direct_focus=True)
        x11.list_windows = lambda: []

        result = x11.send_backspace_once(
            0x5678,
            expected_pid=4321,
            expected_title="wrong title",
            ewmh_timeout_s=0.0,
            direct_timeout_s=0.0,
        )

        self.assertEqual(result["status"], "RESET_NOT_DELIVERED")
        self.assertEqual(result["xtest_key_events_sent"], 0)
        self.assertEqual(x11.xtst.events, [])
        self.assertEqual(x11.lib.calls, [])

    def test_identity_is_rechecked_immediately_before_key_dispatch(self) -> None:
        x11 = make_x11(accept_direct_focus=True)
        matching_window = x11.list_windows()[0]
        identity_checks = [0]
        def disappear_before_dispatch():
            identity_checks[0] += 1
            return [matching_window] if identity_checks[0] == 1 else []
        x11.list_windows = disappear_before_dispatch

        result = x11.send_backspace_once(
            0x5678,
            expected_pid=4321,
            expected_title="MuJoCo : X2 simulator",
            ewmh_timeout_s=0.0,
            direct_timeout_s=0.0,
        )

        self.assertEqual(result["status"], "RESET_NOT_DELIVERED")
        self.assertIsNone(result["identity_before_dispatch"])
        self.assertEqual(x11.xtst.events, [])


if __name__ == "__main__":
    unittest.main()
