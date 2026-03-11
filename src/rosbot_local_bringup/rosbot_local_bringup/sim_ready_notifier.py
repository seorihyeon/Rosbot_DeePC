#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from controller_manager_msgs.srv import ListControllers

class sim_ready_notifier(Node):
    def __init__(self):
        super().__init__('sim_ready_notifier')

        self.clock_ok = False
        self.gt_ok = False
        self.ctrl_ok = False
        self.done = False

        self.create_subscription(Clock, '/clock', self.clock_cb, 10)
        self.create_subscription(Odometry, '/model/rosbot/odometry_gt', self.gt_cb, 10)

        self.cli = self.create_client(ListControllers, '/controller_manager/list_controllers')
        self.timer = self.create_timer(0.5, self.check_ready)

        self.get_logger().info('Waiting for simulation to become ready...')

    def clock_cb(self, msg):
        self.clock_ok = True

    def gt_cb(self, msg):
        self.gt_ok = True

    def check_ready(self):
        if self.done:
            return
        
        if self.cli.service_is_ready():
            req = ListControllers.Request()
            future = self.cli.call_async(req)
            future.add_done_callback(self.controllers_cb)

        self.maybe_print_ready()

    def controllers_cb(self, future):
        try:
            result = future.result()
            active_names = {c.name for c in result.controller if c.state == 'active'}
            self.ctrl_ok = 'differential_drive_controller' in active_names
        except Exception as e:
            self.get_logger().warn(f'ListControllers faile: {e}')

        self.maybe_print_ready()

    def maybe_print_ready(self):
        if self.done:
            return
        
        if self.clock_ok and self.gt_ok and self.ctrl_ok:
            self.done = True
            self.get_logger().info('='*60)
            self.get_logger().info('Simulation READY')
            self.get_logger().info('    /clock: OK')
            self.get_logger().info('    /model/rosbot/odometry_gt: OK')
            self.get_logger().info('    differential_drive_controller: ACTIVE')
            self.get_logger().info('='*60)

def main():
    rclpy.init()
    node = sim_ready_notifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()