"""Real Humble generated-message regression; isolated ROS graph, no MC/physics.

Run with ROS_DOMAIN_ID isolated from live MC, inside --network=none Docker.
The ordinary non-ROS test environment reports an explicit skip.
"""
import threading
import unittest

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from aimdk_msgs.srv import GetMcAction, SetMcAction
    from aimdk_msgs.msg import McCommonState
    from geometry_msgs.msg import Twist
except ImportError:
    rclpy = None


@unittest.skipIf(rclpy is None, 'requires installed ROS 2 Humble and aimdk_msgs')
class GeneratedRosModeTransportTests(unittest.TestCase):
    def test_mode_read_after_spin_once_detaches_node(self):
        self.run_generated(defer_step=False)

    def test_step_arming_is_not_claimed_as_applied_before_velocity(self):
        self.run_generated(defer_step=True)

    def run_generated(self, defer_step):
        from scripts.research.issue46.native_runtime import xvfb_native_reset_stand as launch
        code = launch.locomotion_dock_script(
            world_start_xy=[0., 0.], world_start_yaw_rad=0.,
            dock_world_xy=[1., 0.], dock_world_yaw_rad=0.,
            dock_position_only=False, refine_world_xy=None,
            refine_world_yaw_rad=None, refine_position_only=False,
            dock_position_tolerance_m=.1, refine_position_tolerance_m=.1,
            timeout_s=10.)
        namespace = {}
        exec(compile(code.split('rclpy.init();n=Dock();registered=False')[0],
                     'generated_dock', 'exec'), namespace)
        namespace['PLAN']['pickup_navigation'] = True
        namespace['PLAN']['native_pose_feedback'] = False
        namespace['PLAN']['use_cmd_vel_bridge'] = True

        class Services(Node):
            def __init__(self):
                super().__init__('isolated_mode_service')
                self.mode = 'STAND_DEFAULT'
                self.set_count = 0
                self.step_armed = False
                self.pub = self.create_publisher(McCommonState, '/aima/mc/common/state', 10)
                self.create_timer(.01, self.state)
                self.create_subscription(Twist, '/cmd_vel', self.command, 10)
                self.create_service(GetMcAction, '/aimdk_5Fmsgs/srv/GetMcAction', self.get)
                self.create_service(SetMcAction, '/aimdk_5Fmsgs/srv/SetMcAction', self.set)

            def get(self, request, response):
                response.header.code = 0
                response.info.action_desc = self.mode
                response.info.status.value = 100
                return response

            def set(self, request, response):
                self.set_count += 1
                self.step_armed = request.command.action_desc == 'LOCOMOTION_STEP'
                self.mode = ('STAND_DEFAULT' if defer_step and self.step_armed
                             else request.command.action_desc)
                response.response.header.code = 0
                response.response.status.value = 1
                response.response.message = 'synthetic acknowledgement'
                return response

            def state(self):
                value = McCommonState()
                value.action_info.action_desc = self.mode
                value.action_info.status.value = 100
                value.speed_status.speed_mode.value = 4 if self.step_armed else 1
                value.body_status.value = 1
                self.pub.publish(value)

            def command(self, value):
                if self.step_armed and abs(value.linear.x) > 0:
                    self.mode = 'LOCOMOTION_STEP'

        rclpy.init()
        server = Services()
        owner = SingleThreadedExecutor()
        owner.add_node(server)
        thread = threading.Thread(target=owner.spin, daemon=True)
        thread.start()
        client = namespace['Dock']()
        try:
            for target in ('STAND_DEFAULT', 'LOCOMOTION_STEP', 'STAND_DEFAULT', 'LOCOMOTION_STEP'):
                # Exact caller pattern: normal spinning followed by a bounded
                # synchronous query. Humble retains a stale executor property.
                client.spin(.3)
                self.assertNotIn(client, rclpy.get_global_executor().get_nodes())
                client.stand(target)
                verified = client.mode_requests[-1]['verification']
                if defer_step and target == 'LOCOMOTION_STEP':
                    self.assertEqual(verified['outcome'], 'ARMED_PENDING_VELOCITY')
                    self.assertIsNone(verified['applied'])
                    self.assertEqual(server.mode, 'STAND_DEFAULT')
                    # Synthetic ROS message only; no MC/physics in this graph.
                    client.publish(.2)
                    client.spin(.1)
                    client.stand(target)
                    verified = client.mode_requests[-1]['verification']
                self.assertEqual(verified['applied']['name'], target)
                self.assertNotIn(client, rclpy.get_global_executor().get_nodes())
            self.assertEqual(server.set_count, 3)
            # An already registered node must retain its original membership.
            rclpy.get_global_executor().add_node(client)
            client.stand('LOCOMOTION_STEP')
            self.assertIn(client, rclpy.get_global_executor().get_nodes())
            self.assertEqual(server.set_count, 3)
        finally:
            client.destroy_node()
            owner.shutdown()
            thread.join(timeout=2.)
            server.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
