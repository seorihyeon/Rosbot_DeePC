#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from controller_manager_msgs.srv import ListControllers


class SimReadyNotifier(Node):
    def __init__(self):
        super().__init__('sim_ready_notifier')

        self.declare_parameter('spawner_node_name', '/spawner_differential_drive_controller')
        self.declare_parameter('stable_ready_count', 3)

        self.spawner_node_name = self.get_parameter('spawner_node_name').get_parameter_value().string_value
        self.stable_ready_count = self.get_parameter('stable_ready_count').get_parameter_value().integer_value
        self.clock_ok = False
        self.odom_ok = False
        self.controller_ok = False
        self.spawner_gone = False

        self.ready_streak = 0
        self.last_waiting_reason = None
        self.done = False

        self.last_clock_time = None
        self.controller_client = self.create_client(
            ListControllers,
            '/controller_manager/list_controllers'
        )

        self.create_subscription(Clock, '/clock', self.clock_cb, 10)
        self.create_subscription(
            Odometry,
            '/model/rosbot/odometry_gt',
            self.odom_cb,
            10
        )

        self.timer = self.create_timer(0.5, self.check_ready)

        self.get_logger().info('Waiting for simulation to become ready...')

    def clock_cb(self, msg: Clock):
        self.clock_ok = True
        self.last_clock_time = msg.clock

    def odom_cb(self, msg: Odometry):
        self.odom_ok = True

    def get_graph_node_full_names(self) -> set[str]:
        names = set()
        for name, namespace in self.get_node_names_and_namespaces():
            if namespace == '/' or namespace == '':
                names.add(f'/{name}')
            else:
                names.add(f'{namespace.rstrip("/")}/{name}')
        return names

    def is_spawner_gone(self) -> bool:
        graph_nodes = self.get_graph_node_full_names()
        if self.spawner_node_name in graph_nodes:
            return False
        return True


    def check_ready(self):
        if self.done:
            return

        # controller_manager 서비스가 아직 없으면 다음 주기에 다시 확인
        if self.controller_client.wait_for_service(timeout_sec=0.0):
            req = ListControllers.Request()
            future = self.controller_client.call_async(req)
            future.add_done_callback(self._controller_response_cb)

    def _controller_response_cb(self, future):
        if self.done:
            return

        try:
            result = future.result()
        except Exception as e:
            self.get_logger().warn(f'Failed to query controllers: {e}')
            return

        active_names = {
            c.name for c in result.controller
            if c.state == 'active'
        }

        self.controller_ok = 'differential_drive_controller' in active_names

        self.spawner_gone = self.is_spawner_gone()

        if self.clock_ok and self.odom_ok and self.controller_ok and self.spawner_gone:
            self.ready_streak += 1
        else:
            self.ready_streak = 0

        if self.ready_streak > self.stable_ready_count:
            self.get_logger().info('=' * 60)
            self.get_logger().info('Simulation READY')
            self.get_logger().info('    /clock: OK')
            self.get_logger().info('    /model/rosbot/odometry_gt: OK')
            self.get_logger().info('    differential_drive_controller: ACTIVE')
            self.get_logger().info('    spawner_gone: OK')
            self.get_logger().info('=' * 60)

            self.done = True
            self.timer.cancel()


def main():
    rclpy.init()
    node = SimReadyNotifier()

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()