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

        self.clock_ok = False
        self.odom_ok = False
        self.controller_ok = False
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

        if self.clock_ok and self.odom_ok and self.controller_ok:
            self.get_logger().info('=' * 60)
            self.get_logger().info('Simulation READY')
            self.get_logger().info(f'    /clock: {"OK" if self.clock_ok else "WAIT"}')
            self.get_logger().info(f'    /model/rosbot/odometry_gt: {"OK" if self.odom_ok else "WAIT"}')
            self.get_logger().info(
                f'    differential_drive_controller: '
                f'{"ACTIVE" if self.controller_ok else "WAIT"}'
            )
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