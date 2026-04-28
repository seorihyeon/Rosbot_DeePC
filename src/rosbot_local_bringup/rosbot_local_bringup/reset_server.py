#!/usr/bin/env python3
import math
import threading
import time
from typing import Optional

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Twist, Pose
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from rosbot_interfaces.srv import ResetPose


class ResetRosbotServer(Node):
    def __init__(self) -> None:
        super().__init__('reset_rosbot_server')

        self.declare_parameter('reset_service', '/reset_rosbot')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('set_pose_service', '/world/flat_empty/set_pose')
        self.declare_parameter('entity_name', 'rosbot')
        self.declare_parameter('entity_type', int(Entity.MODEL))
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('target_z', 0.1)
        self.declare_parameter('pre_zero_publish_count', 3)
        self.declare_parameter('post_zero_publish_count', 5)
        self.declare_parameter('zero_publish_period', 0.03)
        self.declare_parameter('service_timeout_sec', 1.0)

        self.reset_service = self.get_parameter('reset_service').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.set_pose_service = self.get_parameter('set_pose_service').value
        self.entity_name = self.get_parameter('entity_name').value
        self.entity_type = self.get_parameter('entity_type').value
        self.frame_id = self.get_parameter('frame_id').value
        self.pre_zero_publish_count = self.get_parameter('pre_zero_publish_count').value
        self.post_zero_publish_count = self.get_parameter('post_zero_publish_count').value
        self.zero_publish_period = self.get_parameter('zero_publish_period').value
        self.service_timeout_sec = self.get_parameter('service_timeout_sec').value

        cmd_vel_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.reset_srv_group = ReentrantCallbackGroup()
        self.pose_client_group = ReentrantCallbackGroup()

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, cmd_vel_qos)
        self.reset_srv = self.create_service(
            ResetPose,
            self.reset_service,
            self.on_reset_request,
            callback_group=self.reset_srv_group,
        )
        self.pose_client = self.create_client(
            SetEntityPose,
            self.set_pose_service,
            callback_group=self.pose_client_group,
        )

        self.is_busy = False

        self.get_logger().info(
            'ready: reset_service=%s, cmd_vel=%s, set_pose=%s, entity=%s'
            % (self.reset_service, self.cmd_vel_topic, self.set_pose_service, self.entity_name)
        )

    def make_zero_twist(self) -> Twist:
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        return msg

    def publish_zero_burst(self, count: int) -> None:
        for _ in range(max(count, 0)):
            self.cmd_pub.publish(self.make_zero_twist())
            if self.zero_publish_period > 0.0:
                time.sleep(self.zero_publish_period)

    def read_target_pose(
        self,
        request: ResetPose.Request,
    ) -> tuple[float, float, float, float]:
        x = float(request.x)
        y = float(request.y)
        z = float(self.get_parameter('target_z').value)
        yaw = float(request.yaw)

        values = {
            'x': x,
            'y': y,
            'target_z': z,
            'yaw': yaw,
        }
        invalid = [name for name, value in values.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError('non-finite reset pose parameter(s): ' + ', '.join(invalid))

        return x, y, z, yaw

    def make_pose_request(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        target_yaw: float,
    ) -> SetEntityPose.Request:
        req = SetEntityPose.Request()
        req.entity.name = self.entity_name
        req.entity.type = self.entity_type

        req.pose = Pose()
        req.pose.position.x = target_x
        req.pose.position.y = target_y
        req.pose.position.z = target_z

        half_yaw = target_yaw * 0.5
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = math.sin(half_yaw)
        req.pose.orientation.w = math.cos(half_yaw)
        return req

    def call_set_pose(self, request: SetEntityPose.Request):
        future = self.pose_client.call_async(request)
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())

        if not done_event.wait(timeout=self.service_timeout_sec):
            future.cancel()
            raise TimeoutError(
                f'set_pose did not respond within {self.service_timeout_sec:.1f} sec'
            )

        return future.result()

    def perform_reset(self, request: ResetPose.Request) -> tuple[bool, str]:
        if self.is_busy:
            msg = 'reset already in progress'
            self.get_logger().warning(msg)
            return False, msg

        self.is_busy = True

        try:
            target_x, target_y, target_z, target_yaw = self.read_target_pose(request)
        except ValueError as exc:
            msg = f'invalid reset pose: {exc}'
            self.get_logger().error(msg)
            self.is_busy = False
            return False, msg

        self.get_logger().info(
            'reset request received: '
            f'x={target_x:+.3f}, y={target_y:+.3f}, '
            f'z={target_z:+.3f}, yaw={target_yaw:+.3f}'
        )
        self.publish_zero_burst(self.pre_zero_publish_count)

        if not self.pose_client.wait_for_service(timeout_sec=self.service_timeout_sec):
            msg = (
                f'service unavailable: {self.set_pose_service} '
                '(start ros_gz_bridge for /world/<world>/set_pose first)'
            )
            self.get_logger().error(msg)
            self.publish_zero_burst(self.post_zero_publish_count)
            self.is_busy = False
            return False, msg

        try:
            resp = self.call_set_pose(
                self.make_pose_request(target_x, target_y, target_z, target_yaw)
            )
            if resp is None or not resp.success:
                msg = 'set_pose returned success=false'
                self.get_logger().error(msg)
                return False, msg
        except Exception as exc:
            msg = f'set_pose call failed: {exc}'
            self.get_logger().error(msg)
            return False, msg
        finally:
            self.publish_zero_burst(self.post_zero_publish_count)
            self.is_busy = False

        msg = (
            'pose reset succeeded: '
            f'x={target_x:+.3f}, y={target_y:+.3f}, yaw={target_yaw:+.3f}'
        )
        self.get_logger().info(msg)
        return True, msg

    def on_reset_request(
        self,
        request: ResetPose.Request,
        response: ResetPose.Response,
    ) -> ResetPose.Response:
        success, message = self.perform_reset(request)
        response.success = bool(success)
        response.message = str(message)
        return response


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ResetRosbotServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
