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
except ImportError:
    rclpy = None


@unittest.skipIf(rclpy is None, 'requires installed ROS 2 Humble and aimdk_msgs')
class GeneratedRosModeTransportTests(unittest.TestCase):
    def test_mode_read_after_spin_once_detaches_node(self):
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

        class Services(Node):
            def __init__(self):
                super().__init__('isolated_mode_service')
                self.mode = 'STAND_DEFAULT'
                self.set_count = 0
                self.create_service(GetMcAction, '/aimdk_5Fmsgs/srv/GetMcAction', self.get)
                self.create_service(SetMcAction, '/aimdk_5Fmsgs/srv/SetMcAction', self.set)

            def get(self, request, response):
                response.header.code = 0
                response.info.action_desc = self.mode
                response.info.status.value = 100
                return response

            def set(self, request, response):
                self.set_count += 1
                self.mode = request.command.action_desc
                response.response.header.code = 0
                response.response.status.value = 1
                response.response.message = 'synthetic acknowledgement'
                return response

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
                self.assertEqual(client.mode_requests[-1]['verification']['applied']['name'], target)
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
