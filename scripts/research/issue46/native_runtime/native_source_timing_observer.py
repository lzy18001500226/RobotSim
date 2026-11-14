"""Read-only timing diagnostic; never used as a control/safety bypass."""
import json
import copy
import threading
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from aimdk_msgs.msg import JointCommandArray, McCommonState


class TimingObserver(Node):
    def __init__(self):
        super().__init__('robotsim_mc_source_timing_observer')
        self.channels = {}
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.writer = threading.Thread(target=self.write_loop, daemon=True)
        self.started = time.monotonic()
        self.out = Path('/workspace/evidence')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        for key, topic, message in (
            ('waist', '/robotsim/issue46/mc/waist/command', JointCommandArray),
            ('common', '/aima/mc/common/state', McCommonState),
        ):
            self.create_subscription(message, topic, lambda m, k=key: self.received(k, m), qos)
        self.writer.start()

    def received(self, key, message):
        now = time.monotonic_ns()
        stamp = int(message.header.stamp.sec) * 1000000000 + int(message.header.stamp.nanosec)
        with self.lock:
            self.store_message(key, now, stamp)

    def store_message(self, key, now, stamp):
        ch = self.channels.setdefault(key, dict(count=0, repeated=0, regressed=0, max_gap_s=0., gaps=[]))
        if ch['count']:
            gap = (now - ch['receipt_ns']) * 1e-9
            ch['max_gap_s'] = max(gap, ch['max_gap_s'])
            if gap > .02:
                ch['gaps'].append(dict(receipt_ns=now, gap_s=gap, stamp_ns=stamp))
                ch['gaps'] = ch['gaps'][-16:]
            ch['repeated'] += int(stamp == ch['stamp_ns'])
            ch['regressed'] += int(stamp < ch['stamp_ns'])
        ch.update(count=ch['count'] + 1, receipt_ns=now, stamp_ns=stamp,
                  age_at_receipt_s=(time.time_ns() - stamp) * 1e-9)

    def record(self):
        now = time.monotonic_ns()
        with self.lock:
            channels=copy.deepcopy(self.channels)
        value = dict(classification='READ_ONLY_INDEPENDENT_ROS_SOURCE_TIMING',
                     pid=os.getpid(), host_ns=now, wall_ns=time.time_ns(),
                     channels=channels, robot_command_publishers=0)
        temporary = self.out / 'mc_source_timing.tmp'
        temporary.write_text(json.dumps(value) + '\n')
        os.replace(temporary, self.out / 'mc_source_timing.json')
        with (self.out / 'mc_source_timing.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')

    def write_loop(self):
        while not self.finished.wait(.2):
            self.record()


def main():
    rclpy.init()
    node = TimingObserver()
    try:
        while rclpy.ok() and time.monotonic() - node.started < 650.:
            rclpy.spin_once(node, timeout_sec=.05)
    finally:
        node.finished.set()
        node.writer.join(timeout=2.)
        node.record()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
