#!/usr/bin/env python3
import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from std_msgs.msg import Empty
from std_msgs.msg import Header
from geometry_msgs.msg import TwistStamped, Pose
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose


class ResetRosbotServer(Node):
    def __init__(self) -> None:
        super().__init__('reset_rosbot_server')

        self.declare_parameter('trigger_topic', '/reset_rosbot')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('set_pose_service', '/world/flat_empty/set_pose')
        self.declare_parameter('entity_name', 'rosbot')
        self.declare_parameter('entity_type', int(Entity.MODEL))
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_z', 0.1)
        self.declare_parameter('target_yaw', 0.0)
        self.declare_parameter('pre_zero_publish_count', 3)
        self.declare_parameter('post_zero_publish_count', 5)
        self.declare_parameter('zero_publish_period', 0.03)
        self.declare_parameter('service_timeout_sec', 1.0)

        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.set_pose_service = self.get_parameter('set_pose_service').value
        self.entity_name = self.get_parameter('entity_name').value
        self.entity_type = self.get_parameter('entity_type').value
        self.frame_id = self.get_parameter('frame_id').value
        self.target_x = self.get_parameter('target_x').value
        self.target_y = self.get_parameter('target_y').value
        self.target_z = self.get_parameter('target_z').value
        self.target_yaw = self.get_parameter('target_yaw').value
        self.pre_zero_publish_count = self.get_parameter('pre_zero_publish_count').value
        self.post_zero_publish_count = self.get_parameter('post_zero_publish_count').value
        self.zero_publish_period = self.get_parameter('zero_publish_period').value
        self.service_timeout_sec = self.get_parameter('service_timeout_sec').value

        cmd_vel_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, cmd_vel_qos)
        self.trigger_sub = self.create_subscription(Empty, self.trigger_topic, self.on_reset_trigger, 10)
        self.pose_client = self.create_client(SetEntityPose, self.set_pose_service)
        self.is_busy = False
        self.pose_future = None

        self.get_logger().info(
            'ready: trigger=%s, cmd_vel=%s, service=%s, entity=%s'
            % (self.trigger_topic, self.cmd_vel_topic, self.set_pose_service, self.entity_name)
        )

    def make_zero_twist(self) -> TwistStamped:
        msg = TwistStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        return msg

    def publish_zero_burst(self, count: int) -> None:
        for _ in range(max(count, 0)):
            self.cmd_pub.publish(self.make_zero_twist())
            if self.zero_publish_period > 0.0:
                time.sleep(self.zero_publish_period)

    def make_pose_request(self) -> SetEntityPose.Request:
        req = SetEntityPose.Request()
        req.entity.name = self.entity_name
        req.entity.type = self.entity_type

        req.pose = Pose()
        req.pose.position.x = self.target_x
        req.pose.position.y = self.target_y
        req.pose.position.z = self.target_z

        half_yaw = self.target_yaw * 0.5
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = math.sin(half_yaw)
        req.pose.orientation.w = math.cos(half_yaw)
        return req

    def call_set_pose(self) -> None:
        if not self.pose_client.wait_for_service(timeout_sec=self.service_timeout_sec):
            self.get_logger().error(
                f'service unavailable: {self.set_pose_service} '
                '(start ros_gz_bridge for /world/<world>/set_pose first)'
            )
            self.publish_zero_burst(self.post_zero_publish_count)
            self.is_busy = False
            return

        self.pose_future = self.pose_client.call_async(self.make_pose_request())
        self.pose_future.add_done_callback(self.on_set_pose_done)
    
    def on_set_pose_done(self, future) -> None:
        try:
            resp = future.result()
            if resp is None or not resp.success:
                self.get_logger().error('set_pose returned success=false')
            else:
                self.get_logger().info('pose reset succeeded')
        except Exception as exc:
            self.get_logger().error(f'set_pose call failed: {exc}')
        finally:
            self.publish_zero_burst(self.post_zero_publish_count)
            self.is_busy = False
            self.pose_future = None

    def on_reset_trigger(self, _: Empty) -> None:
        if self.is_busy:
            self.get_logger().warning('reset already in progress; ignoring trigger')
            return

        self.is_busy = True
        self.get_logger().info('reset trigger received')
        self.publish_zero_burst(self.pre_zero_publish_count)
        self.call_set_pose()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ResetRosbotServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
