"""Offline tests for the safe EWMH focus request used by Reset automation."""

from __future__ import annotations

import unittest

from issue80_mujoco_x11_reset import CLIENT_MESSAGE, build_net_active_window_event


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


if __name__ == "__main__":
    unittest.main()
