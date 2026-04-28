#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

def quat_to_yaw(x,y,z,w):
    # Calculate yaw from quaternion
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    return wrap_to_pi(math.atan2(siny_cosp, cosy_cosp))

def wrap_to_pi(angle):
    # Map yaw to [-pi, pi].
    raw = float(angle)
    wrapped = (raw + math.pi) % (2.0*math.pi) - math.pi
    if math.isclose(wrapped, -math.pi, abs_tol=1.0e-12) and raw > 0.0:
        return math.pi
    return wrapped

def signed_angle_diff(angle):
    # Map angle differences to [-pi, pi].
    return wrap_to_pi(angle)

class CircleTest(Node):
    def __init__(self):
        super().__init__('circle_test')

        # parameters
        self.declare_parameter('v', 0.3) # m/s
        self.declare_parameter('omega', 0.3) # rad/s
        self.declare_parameter('wheelbase', 0.20) # m
        self.declare_parameter('cmd_hz', 20.0)
        self.declare_parameter('print_hz', 2.0)
        self.declare_parameter('odom_topic', '/model/rosbot/odometry_gt')
        self.declare_parameter('unwrap_yaw', True) # yaw unwrap 사용 여부

        self.v = float(self.get_parameter('v').value)
        self.omega = float(self.get_parameter('omega').value)
        self.L = float(self.get_parameter('wheelbase').value)
        self.cmd_hz = float(self.get_parameter('cmd_hz').value)
        self.print_hz = float(self.get_parameter('print_hz').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.unwrap_yaw = bool(self.get_parameter('unwrap_yaw').value)

        # publish/subscript
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)
        
        # Latest output
        self.last_odom = None
        self.prev_yaw_raw = None
        self.yaw_unwrapped = 0.0

        # timer
        self.cmd_timer = self.create_timer(1.0 / self.cmd_hz, self.publish_cmd)
        self.print_timer = self.create_timer(1.0 / self.print_hz, self.print_status)

        r = float('inf') if abs(self.omega) < 1e-9 else (self.v / self.omega)
        self.get_logger().info(f"Circle cmd: v={self.v:.3f} m/s, omega = {self.omega:.3f} rad/s, radius~{r:.3f} m")
        self.get_logger().info(f"Odom topic: {self.odom_topic}")
        self.get_logger().info(f"unwrap_yaw: {self.unwrap_yaw}")

    def publish_cmd(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = self.v
        msg.twist.angular.z = self.omega
        self.pub.publish(msg)

    def on_odom(self, odom: Odometry):
        self.last_odom = odom

    def _update_unwrap(self, yaw_raw):
        # Update self.yaw_unrawpped using latest raw yaw in [-pi, pi].
        if self.prev_yaw_raw is None:
            self.prev_yaw_raw = yaw_raw
            self.yaw_unwrapped = yaw_raw
            return self.yaw_unwrapped

        dyaw = signed_angle_diff(yaw_raw - self.prev_yaw_raw)
        self.yaw_unwrapped += dyaw
        self.prev_yaw_raw = yaw_raw
        return self.yaw_unwrapped

    def print_status(self):
        v_cmd = self.v
        w_cmd = self.omega
        
        if self.last_odom is None:
            print(f"[IN ] v_cmd={v_cmd:.3f} omega_cmd={w_cmd:.3f} | [OUT] (waiting odometry...)")
            return

        odom = self.last_odom
        t = odom.header.stamp.sec + odom.header.stamp.nanosec*1e-9

        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        
        yaw_raw = quat_to_yaw(q.x, q.y, q.z, q.w)

        yaw_out = self._update_unwrap(yaw_raw) if self.unwrap_yaw else yaw_raw

        v = odom.twist.twist.linear.x
        omega = odom.twist.twist.angular.z

        if self.unwrap_yaw:
            print(
                f"t={t:10.3f} | "
                f"[IN ] v_cmd={v_cmd:+.3f} omega_cmd={w_cmd:+.3f} |"
                f"[OUT] x={x:+.3f} y={y:+.3f} yaw_raw={yaw_raw:+.3f} yaw_unwrap={yaw_out:+.3f} v={v:+.3f} omega={omega:+.3f}"
            )
        else:
            print(
                f"t={t:10.3f} | "
                f"[IN ] v_cmd={v_cmd:+.3f} omega_cmd={w_cmd:+.3f} |"
                f"[OUT] x={x:+.3f} y={y:+.3f} yaw_raw={yaw_raw:+.3f} v={v:+.3f} omega={omega:+.3f}"
            )

    def stop(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        for _ in range(5):
            self.pub.publish(msg)

def main():
    rclpy.init()
    node = CircleTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
