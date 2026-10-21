"""Offline safety tests for process-verified X11 Reset focus and key dispatch."""

from __future__ import annotations

import ctypes
import unittest

from issue80_mujoco_x11_reset import CLIENT_MESSAGE, X11, build_net_active_window_event


class FakeXlib:
    def __init__(self, *, accept_direct_focus: bool):
        self.focus = 1  # X PointerRoot; deliberately not the target XID.
        self.accept_direct_focus = accept_direct_focus
        self.calls: list[tuple[object, ...]] = []

    def XInternAtom(self, _display: int, _name: bytes, _only_if_exists: int) -> int:
        return 0xABC

    def XGetInputFocus(self, _display: int, focus_ptr: object, revert_ptr: object) -> int:
        ctypes.cast(focus_ptr, ctypes.POINTER(ctypes.c_ulong))[0] = self.focus
        ctypes.cast(revert_ptr, ctypes.POINTER(ctypes.c_int))[0] = 2
        self.calls.append(("get-focus", self.focus))
        return 1

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
    def __init__(self):
        self.events: list[tuple[int, int]] = []

    def XTestFakeKeyEvent(self, _display: int, keycode: int, pressed: int, _time: int) -> int:
        self.events.append((keycode, pressed))
        return 1


def make_x11(*, accept_direct_focus: bool, window: int = 0x5678) -> X11:
    x11 = X11.__new__(X11)
    x11.display = 0x1234
    x11.root = 0x1000
    x11.lib = FakeXlib(accept_direct_focus=accept_direct_focus)
    x11.xtst = FakeXTest()
    x11.list_windows = lambda: [
        {
            "window_id": window,
            "pid": 4321,
            "title": "MuJoCo : X2 simulator",
            "viewable": True,
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
        self.assertEqual(count[0], 3)

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
        observed = iter(([matching_window], []))
        x11.list_windows = lambda: next(observed)

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
