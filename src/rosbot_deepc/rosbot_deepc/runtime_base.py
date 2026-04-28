import math
import threading
from typing import Callable, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosbot_interfaces.srv import ResetPose

from .utils import (
    normalize_yaw_representation,
    quat_to_yaw,
    signed_angle_diff,
    unwrap_angle,
    yaw_representation_uses_unwrapped_scalar,
)


class RuntimeBase(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._declare_runtime_parameters()
        self._load_runtime_parameters()
        self._create_runtime_interfaces()
        self._init_runtime_state()

    def _declare_runtime_parameters(self) -> None:
        self.declare_parameter("cmd_topic", '/cmd_vel')
        self.declare_parameter("odom_topic", 'odometry/wheels')

        self.declare_parameter("yaw_representation", 'wrap')

        self.declare_parameter("reset_before_start", False)
        self.declare_parameter("reset_service", '/reset_rosbot')
        self.declare_parameter("reset_timeout_sec", 10.0)
        self.declare_parameter("reset_x", 0.0)
        self.declare_parameter("reset_y", 0.0)
        self.declare_parameter("reset_yaw", 0.0)

        self.declare_parameter("sample_time", 0.1)

        self.declare_parameter("v_min", 0.0)
        self.declare_parameter("v_max", 0.7)
        self.declare_parameter("w_min", -2.0)
        self.declare_parameter("w_max", 2.0)

        self.declare_parameter("sanitize_odom_twist", True)
        self.declare_parameter("max_abs_measured_w", 10.0)

    def _load_runtime_parameters(self) -> None:
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.yaw_representation = normalize_yaw_representation(
            self.get_parameter("yaw_representation").value
        )
        self.uses_unwrapped_yaw = yaw_representation_uses_unwrapped_scalar(
            self.yaw_representation
        )

        self.reset_before_start = bool(self.get_parameter("reset_before_start").value)
        self.reset_service = str(self.get_parameter("reset_service").value)
        self.reset_timeout_sec = float(self.get_parameter("reset_timeout_sec").value)
        self.reset_x = float(self.get_parameter("reset_x").value)
        self.reset_y = float(self.get_parameter("reset_y").value)
        self.reset_yaw = float(self.get_parameter("reset_yaw").value)

        self.dt = float(self.get_parameter("sample_time").value)

        self.v_min = float(self.get_parameter("v_min").value)
        self.v_max = float(self.get_parameter("v_max").value)
        self.w_min = float(self.get_parameter("w_min").value)
        self.w_max = float(self.get_parameter("w_max").value)

        self.sanitize_odom_twist = bool(self.get_parameter("sanitize_odom_twist").value)
        self.max_abs_measured_w = float(self.get_parameter("max_abs_measured_w").value)

    def _create_runtime_interfaces(self) -> None:
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)

        self.reset_cli = None
        if self.reset_service:
            self.reset_cli = self.create_client(ResetPose, self.reset_service)

    def _init_runtime_state(self) -> None:
        self.finished = False
        self._shutdown_requested = False
        self.last_odom: Optional[Odometry] = None
        self.odom_lock = threading.Lock()
        self.last_odom_stamp_ns = -1
        self.new_odom_event = threading.Event()

        self.waiting_for_reset = False
        self.waiting_for_post_reset_odom = False
        self.reset_future = None
        self.reset_timeout_timer = None
        self.startup_timer = None
        self._startup_fn: Optional[Callable[[], None]] = None

        self._reset_yaw_tracking_state()

    def _reset_yaw_tracking_state(self) -> None:
        self.last_raw_yaw: Optional[float] = None
        self.current_yaw: Optional[float] = None
        self.last_yaw_rate_stamp_ns = -1
        self.current_measured_w: Optional[float] = None
        self.last_valid_measured_w: Optional[float] = None
        self.measured_w_outlier_count = 0

    def _clear_odom_state(self, *, reset_yaw: bool = False) -> None:
        with self.odom_lock:
            self.last_odom = None
            self.last_odom_stamp_ns = -1
            self.new_odom_event.clear()

        if reset_yaw:
            self._reset_yaw_tracking_state()

    def on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        raw_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        raw_w = float(msg.twist.twist.angular.z)
        warn_msg = None

        with self.odom_lock:
            prev_raw_yaw = self.last_raw_yaw
            prev_stamp_ns = self.last_yaw_rate_stamp_ns

            self.last_odom = msg
            self.last_odom_stamp_ns = stamp_ns
            self.new_odom_event.set()

            if self.uses_unwrapped_yaw:
                self.current_yaw = unwrap_angle(
                    new_angle=raw_yaw,
                    prev_angle=prev_raw_yaw,
                    prev_unwrapped=self.current_yaw,
                )
            else:
                self.current_yaw = raw_yaw

            w_meas = raw_w
            if self.sanitize_odom_twist:
                w_est = None
                if prev_raw_yaw is not None and stamp_ns > prev_stamp_ns:
                    dt_sec = (stamp_ns - prev_stamp_ns) * 1.0e-9
                    if dt_sec > 0.0:
                        w_est = signed_angle_diff(raw_yaw - prev_raw_yaw) / dt_sec

                outlier = (
                    (not math.isfinite(raw_w))
                    or (
                        self.max_abs_measured_w > 0.0
                        and abs(raw_w) > self.max_abs_measured_w
                    )
                )
                if outlier:
                    if (
                        w_est is not None
                        and math.isfinite(w_est)
                        and (
                            self.max_abs_measured_w <= 0.0
                            or abs(w_est) <= self.max_abs_measured_w
                        )
                    ):
                        w_meas = w_est
                    elif self.last_valid_measured_w is not None:
                        w_meas = self.last_valid_measured_w
                    else:
                        w_meas = 0.0

                    self.measured_w_outlier_count += 1
                    if (
                        math.isfinite(w_meas)
                        and (
                            self.max_abs_measured_w <= 0.0
                            or abs(w_meas) <= self.max_abs_measured_w
                        )
                    ):
                        self.last_valid_measured_w = w_meas
                    if (
                        self.measured_w_outlier_count <= 5
                        or self.measured_w_outlier_count % 50 == 0
                    ):
                        warn_msg = (
                            "Sanitized odom angular velocity outlier: "
                            f"raw_w={raw_w:+.3f}, used_w={w_meas:+.3f}, "
                            f"count={self.measured_w_outlier_count}"
                        )
                else:
                    self.last_valid_measured_w = raw_w

            self.current_measured_w = w_meas
            self.last_raw_yaw = raw_yaw
            self.last_yaw_rate_stamp_ns = stamp_ns

            waiting_for_post_reset_odom = self.waiting_for_post_reset_odom
            if waiting_for_post_reset_odom:
                self.waiting_for_post_reset_odom = False

        if warn_msg:
            self.get_logger().warn(warn_msg)

        if waiting_for_post_reset_odom:
            self.get_logger().info("fresh odom received after reset")
            self.on_ready_after_reset()

    def on_ready_after_reset(self) -> None:
        pass

    def schedule_startup(self, startup_fn: Callable[[], None], delay_sec: float = 0.1) -> None:
        self._cancel_startup_timer()
        self._startup_fn = startup_fn
        self.startup_timer = self.create_timer(delay_sec, self._run_startup)

    def _run_startup(self) -> None:
        self._cancel_startup_timer()

        startup_fn = self._startup_fn
        self._startup_fn = None
        if startup_fn is not None:
            startup_fn()

    def _cancel_startup_timer(self) -> None:
        if self.startup_timer is None:
            return
        self.startup_timer.cancel()
        self.startup_timer = None

    def _cancel_reset_timeout_timer(self) -> None:
        if self.reset_timeout_timer is None:
            return
        self.reset_timeout_timer.cancel()
        self.reset_timeout_timer = None

    def cancel_timer(self, timer_name: str) -> None:
        timer = getattr(self, timer_name, None)
        if timer is None:
            return
        try:
            timer.cancel()
        finally:
            setattr(self, timer_name, None)

    def request_shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True

        def _shutdown() -> None:
            if rclpy.ok():
                rclpy.shutdown()

        thread = threading.Thread(target=_shutdown, daemon=True)
        thread.start()

    def cancel_common_timers(self) -> None:
        self._cancel_reset_timeout_timer()
        self._cancel_startup_timer()
        self.cancel_timer("timer")

    def _arm_reset_timeout_timer(self) -> None:
        self._cancel_reset_timeout_timer()
        self.reset_timeout_timer = self.create_timer(
            self.reset_timeout_sec,
            self._on_reset_timeout,
        )

    def _shutdown_due_to_reset_failure(self, msg: str) -> None:
        self.get_logger().error(msg)
        self.finished = True
        self._cancel_reset_timeout_timer()
        self._cancel_startup_timer()
        self.request_shutdown()

    def _on_reset_timeout(self) -> None:
        self._cancel_reset_timeout_timer()
        if not self.waiting_for_reset:
            return

        self.waiting_for_reset = False
        self.reset_future = None
        self._shutdown_due_to_reset_failure(
            f"reset service did not respond within {self.reset_timeout_sec:.1f} sec"
        )

    def on_reset_response(self, future) -> None:
        if not self.waiting_for_reset:
            return

        self._cancel_reset_timeout_timer()
        self.waiting_for_reset = False
        self.reset_future = None

        try:
            response = future.result()
        except Exception as exc:
            self._shutdown_due_to_reset_failure(f"reset service call failed: {exc}")
            return

        if response is None or not response.success:
            message = "" if response is None else str(response.message)
            if message:
                self._shutdown_due_to_reset_failure(
                    f"reset service reported failure: {message}"
                )
            else:
                self._shutdown_due_to_reset_failure("reset service reported failure")
            return

        self.waiting_for_post_reset_odom = True
        self._clear_odom_state(reset_yaw=True)
        self.get_logger().info("reset service completed, waiting for fresh odom")

    def make_reset_request(
        self,
        reset_pose: Optional[Tuple[float, float, float]] = None,
    ) -> ResetPose.Request:
        if reset_pose is None:
            x = self.reset_x
            y = self.reset_y
            yaw = self.reset_yaw
        else:
            x, y, yaw = reset_pose

        values = {
            "x": float(x),
            "y": float(y),
            "yaw": float(yaw),
        }
        invalid = [name for name, value in values.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError("non-finite reset pose value(s): " + ", ".join(invalid))

        req = ResetPose.Request()
        req.x = values["x"]
        req.y = values["y"]
        req.yaw = values["yaw"]
        return req

    def request_reset(
        self,
        reset_pose: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if self.waiting_for_reset:
            self.get_logger().warning(
                "reset request ignored because another reset is already pending"
            )
            return

        if self.reset_cli is None:
            self.on_ready_after_reset()
            return

        try:
            request = self.make_reset_request(reset_pose)
        except ValueError as exc:
            self._shutdown_due_to_reset_failure(f"invalid reset pose: {exc}")
            return

        self.publish_cmd(0.0, 0.0)
        if not self.reset_cli.wait_for_service(timeout_sec=self.reset_timeout_sec):
            self._shutdown_due_to_reset_failure(
                f"reset service unavailable: {self.reset_service} "
                f"(waited {self.reset_timeout_sec:.1f} sec)"
            )
            return

        self.waiting_for_reset = True
        self.reset_future = self.reset_cli.call_async(request)
        self.reset_future.add_done_callback(self.on_reset_response)
        self._arm_reset_timeout_timer()
        self.get_logger().info(
            "Reset request is sent: "
            f"x={request.x:+.3f}, y={request.y:+.3f}, yaw={request.yaw:+.3f}"
        )

    def start_with_optional_reset(
        self,
        *,
        reset_pose: Optional[Tuple[float, float, float]] = None,
        wait_for_first_odom_on_skip_reset: bool = False,
        skip_reset_log: Optional[str] = None,
    ) -> None:
        if self.reset_before_start and self.reset_cli is not None:
            self.request_reset(reset_pose)
            return

        if self.reset_before_start and self.reset_cli is None:
            self.get_logger().info(
                "reset_before_start is true but reset service is disabled; "
                "proceeding without reset"
            )

        if wait_for_first_odom_on_skip_reset:
            self.waiting_for_post_reset_odom = True
            self._clear_odom_state(reset_yaw=True)
            if skip_reset_log:
                self.get_logger().info(skip_reset_log)
            return

        self.on_ready_after_reset()

    def publish_cmd(self, v: float, w: float) -> None:
        if not rclpy.ok():
            return

        msg = Twist()
        # msg.header.stamp = self.get_clock().now().to_msg()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def publish_stop_commands(self, repeat: int = 5) -> None:
        for _ in range(repeat):
            self.publish_cmd(0.0, 0.0)

    def current_measured_state(self) -> Tuple[float, float, float, float, float]:
        with self.odom_lock:
            odom = self.last_odom
            yaw = self.current_yaw
            w = self.current_measured_w

        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        if yaw is None:
            yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        v = odom.twist.twist.linear.x
        if w is None:
            w = odom.twist.twist.angular.z
        return x, y, yaw, v, w

    def get_last_odom_stamp_ns(self) -> int:
        with self.odom_lock:
            return self.last_odom_stamp_ns
