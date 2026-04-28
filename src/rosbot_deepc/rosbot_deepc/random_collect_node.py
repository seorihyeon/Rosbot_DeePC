import math
import random

import rclpy

from .collect_base import CollectBase
from .utils import clamp


class RandomCollectNode(CollectBase):
    def __init__(self):
        super().__init__('random_collect_node')

        self._declare_random_parameters()
        self._load_random_parameters()
        self._init_random_state()

        if self.random_seed >= 0:
            random.seed(self.random_seed)

        self.timer = None
        self.schedule_startup(self.begin_collection)

    def begin_collection(self):
        if (
            self.randomize_initial_pose
            and self.reset_before_start
            and self.reset_cli is not None
        ):
            reset_pose = self.sample_random_initial_pose()
            self.initial_reset_pose = reset_pose
            x, y, yaw = reset_pose
            self.get_logger().info(
                "Using random initial pose: "
                f"x={x:+.3f}, y={y:+.3f}, yaw={yaw:+.3f}"
            )
            self.start_collection_reset_sequence(reset_pose=reset_pose)
            return

        if self.randomize_initial_pose:
            self.get_logger().warning(
                "randomize_initial_pose is true, but "
                "reset_before_start/reset_service "
                "is disabled; starting without random initial pose reset"
            )

        self.start_collection_reset_sequence()

    def start_collection_reset_sequence(self, *, reset_pose=None):
        self.start_with_optional_reset(reset_pose=reset_pose)
        if self.timer is None:
            self.timer = self.create_timer(self.dt, self.on_timer)

    def _declare_random_parameters(self):
        self.declare_parameter('random_dataset_steps', 1200)
        self.declare_parameter('progress_log_interval_steps', 20)
        self.declare_parameter('random_seed', -1)
        self.declare_parameter('random_cmd_dv_min', -0.2)
        self.declare_parameter('random_cmd_dv_max', 0.2)
        self.declare_parameter('random_cmd_dw_min', -0.4)
        self.declare_parameter('random_cmd_dw_max', 0.4)
        self.declare_parameter('random_hold_steps_min', 4)
        self.declare_parameter('random_hold_steps_max', 10)
        self.declare_parameter('random_min_command_norm', 0.03)
        self.declare_parameter('randomize_initial_pose', True)
        self.declare_parameter('random_initial_pose_center_x', 0.0)
        self.declare_parameter('random_initial_pose_center_y', 0.0)
        self.declare_parameter('random_initial_pose_radius', 1.0)
        self.declare_parameter('random_initial_yaw_min', -math.pi)
        self.declare_parameter('random_initial_yaw_max', math.pi)

    def _load_random_parameters(self):
        self.random_dataset_steps = int(
            self.get_parameter('random_dataset_steps').value
        )
        self.progress_log_interval_steps = int(
            self.get_parameter('progress_log_interval_steps').value
        )
        if self.progress_log_interval_steps <= 0:
            raise ValueError("progress_log_interval_steps must be positive")

        self.random_seed = int(self.get_parameter('random_seed').value)

        self.random_cmd_dv_min = float(
            self.get_parameter('random_cmd_dv_min').value
        )
        self.random_cmd_dv_max = float(
            self.get_parameter('random_cmd_dv_max').value
        )
        self.random_cmd_dw_min = float(
            self.get_parameter('random_cmd_dw_min').value
        )
        self.random_cmd_dw_max = float(
            self.get_parameter('random_cmd_dw_max').value
        )
        self.random_hold_steps_min = int(
            self.get_parameter('random_hold_steps_min').value
        )
        self.random_hold_steps_max = int(
            self.get_parameter('random_hold_steps_max').value
        )
        self.random_min_command_norm = float(
            self.get_parameter('random_min_command_norm').value
        )
        self.randomize_initial_pose = bool(
            self.get_parameter('randomize_initial_pose').value
        )
        self.random_initial_pose_center_x = float(
            self.get_parameter('random_initial_pose_center_x').value
        )
        self.random_initial_pose_center_y = float(
            self.get_parameter('random_initial_pose_center_y').value
        )
        self.random_initial_pose_radius = float(
            self.get_parameter('random_initial_pose_radius').value
        )
        self.random_initial_yaw_min = float(
            self.get_parameter('random_initial_yaw_min').value
        )
        self.random_initial_yaw_max = float(
            self.get_parameter('random_initial_yaw_max').value
        )

        if self.random_dataset_steps <= 0:
            raise ValueError("random_dataset_steps must be positive")
        if (
            self.random_hold_steps_min <= 0
            or self.random_hold_steps_max < self.random_hold_steps_min
        ):
            raise ValueError("Invalid random hold-step range")
        if self.random_cmd_dv_min > self.random_cmd_dv_max:
            raise ValueError("random_cmd_dv_min must be <= random_cmd_dv_max")
        if self.random_cmd_dw_min > self.random_cmd_dw_max:
            raise ValueError("random_cmd_dw_min must be <= random_cmd_dw_max")
        if self.random_min_command_norm < 0.0:
            raise ValueError("random_min_command_norm must be nonnegative")
        if not math.isfinite(self.random_initial_pose_center_x):
            raise ValueError("random_initial_pose_center_x must be finite")
        if not math.isfinite(self.random_initial_pose_center_y):
            raise ValueError("random_initial_pose_center_y must be finite")
        if (
            not math.isfinite(self.random_initial_pose_radius)
            or self.random_initial_pose_radius < 0.0
        ):
            raise ValueError(
                "random_initial_pose_radius must be finite and nonnegative"
            )
        if not math.isfinite(self.random_initial_yaw_min):
            raise ValueError("random_initial_yaw_min must be finite")
        if not math.isfinite(self.random_initial_yaw_max):
            raise ValueError("random_initial_yaw_max must be finite")
        if self.random_initial_yaw_min > self.random_initial_yaw_max:
            raise ValueError(
                "random_initial_yaw_min must be <= random_initial_yaw_max"
            )

        max_dv = max(abs(self.random_cmd_dv_min), abs(self.random_cmd_dv_max))
        max_dw = max(abs(self.random_cmd_dw_min), abs(self.random_cmd_dw_max))
        max_norm = math.sqrt(max_dv * max_dv + 0.1 * max_dw * max_dw)
        if self.random_min_command_norm > max_norm + 1.0e-12:
            raise ValueError(
                "random_min_command_norm is too large for the configured "
                "random command ranges"
            )

    def _init_random_state(self):
        self.current_cmd_v = 0.0
        self.current_cmd_w = 0.0
        self.hold_count = 0
        self.initial_reset_pose = None

    def sample_random_initial_pose(self):
        theta = random.uniform(0.0, 2.0 * math.pi)
        radius = self.random_initial_pose_radius * math.sqrt(random.random())
        x = self.random_initial_pose_center_x + radius * math.cos(theta)
        y = self.random_initial_pose_center_y + radius * math.sin(theta)
        yaw = random.uniform(
            self.random_initial_yaw_min,
            self.random_initial_yaw_max,
        )
        return x, y, yaw

    def on_ready_after_reset(self):
        self.open_output_csv(
            stem="random",
            header=[
                'step', 'sim_time_sec',
                'x', 'y', 'yaw',
                'cmd_v', 'cmd_w', 'v_meas', 'w_meas',
                'cmd_source'
            ]
        )
        self.get_logger().info("Started random-input collection")
        self.get_logger().info(f"dataset_steps : {self.random_dataset_steps}")
        self.get_logger().info(f"warmup_steps  : {self.warmup_steps}")
        self.get_logger().info(f"output_file   : {self.csv_path}")

    def log_progress(
        self,
        cmd_v: float,
        cmd_w: float,
        v_meas: float,
        w_meas: float,
        cmd_source: str,
    ) -> None:
        done = self.step_idx + 1
        total = self.random_dataset_steps
        percent = 100.0 * done / total
        remaining_steps = total - done
        remaining_sim_time = remaining_steps * self.dt

        self.get_logger().info(
            f"progress={done:4d}/{total} ({percent:5.1f}%) "
            f"remaining={remaining_steps:4d} steps "
            f"(~{remaining_sim_time:6.2f} s sim) "
            f"src={cmd_source} "
            f"cmd=({cmd_v:+.3f}, {cmd_w:+.3f}) "
            f"meas=({v_meas:+.3f}, {w_meas:+.3f})"
        )

    def sample_random_command(self):
        while True:
            dv = random.uniform(self.random_cmd_dv_min, self.random_cmd_dv_max)
            dw = random.uniform(self.random_cmd_dw_min, self.random_cmd_dw_max)
            norm = math.sqrt(dv * dv + 0.1 * dw * dw)
            if norm >= self.random_min_command_norm:
                self.current_cmd_v = clamp(
                    self.current_cmd_v + dv,
                    self.v_min,
                    self.v_max,
                )
                self.current_cmd_w = clamp(
                    self.current_cmd_w + dw,
                    self.w_min,
                    self.w_max,
                )
                break

        self.hold_count = random.randint(
            self.random_hold_steps_min,
            self.random_hold_steps_max
        )

    def on_timer(self):
        if self.finished or self.waiting_for_reset:
            return
        if self.last_odom is None:
            self.publish_cmd(0.0, 0.0)
            return

        if self.step_idx >= self.random_dataset_steps:
            self.finish_and_shutdown()
            return

        x, y, yaw, v_meas, w_meas = self.current_measured_state()

        if self.step_idx < self.warmup_steps:
            cmd_v = 0.0
            cmd_w = 0.0
            cmd_source = 'warmup'
        else:
            if self.hold_count <= 0:
                self.sample_random_command()
            self.hold_count -= 1
            cmd_v = self.current_cmd_v
            cmd_w = self.current_cmd_w
            cmd_source = 'random_hold'

        self.publish_cmd(cmd_v, cmd_w)

        sim_time = self.get_clock().now().nanoseconds * 1e-9
        self.write_csv_row([
            self.step_idx, f'{sim_time:.6f}',
            f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}',
            f'{cmd_v:.6f}', f'{cmd_w:.6f}', f'{v_meas:.6f}', f'{w_meas:.6f}',
            cmd_source,
        ])

        is_last = (self.step_idx + 1) >= self.random_dataset_steps
        should_log = (
            self.step_idx == 0
            or ((self.step_idx + 1) % self.progress_log_interval_steps == 0)
            or is_last
        )
        if should_log:
            self.log_progress(cmd_v, cmd_w, v_meas, w_meas, cmd_source)

        self.step_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = RandomCollectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish_and_shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
