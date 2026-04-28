#!/usr/bin/env python3
import math
import random
from dataclasses import dataclass

import rclpy

from .collect_base import CollectBase
from .utils import body_frame_pose_error, clamp, unicycle_tracking_law


@dataclass
class PoseTarget:
    x: float
    y: float
    yaw: float


class PRBSCollectNode(CollectBase):
    """
    Collect PRBS-excited stabilizer data.

    1) Hold the robot around a fixed operating point with a stabilizing controller.
    2) Add PRBS excitation to the controller output.
    3) Log the resulting input/output data for DeePC Hankel construction.

    This mirrors the collection procedure used in Elokda et al. (2021), adapted from
    quadcopter inputs/outputs to Rosbot inputs/outputs.
    """

    def __init__(self):
        super().__init__("prbs_collect_node")

        self._declare_prbs_parameters()
        self._load_prbs_parameters()
        self._init_prbs_state()

        self.timer = None
        self.schedule_startup(self.begin_collection)

    def begin_collection(self):
        self.start_with_optional_reset(
            wait_for_first_odom_on_skip_reset=True,
            skip_reset_log=(
                "reset_before_start is false; waiting for first odom before starting collection"
            ),
        )
        self.timer = self.create_timer(self.dt, self.on_timer)

    def _declare_prbs_parameters(self) -> None:
        self.declare_parameter("dataset_steps", 1200)
        self.declare_parameter("progress_log_interval_steps", 20)

        # Operating point to stabilize around.
        # If false, the first fresh odom after reset becomes the target pose.
        self.declare_parameter("use_explicit_target_pose", False)
        self.declare_parameter("target_x", 0.0)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_yaw", 0.0)

        # Same structure as the fallback controller in deepc_node, but with zero nominal motion.
        self.declare_parameter("kx", 0.8)
        self.declare_parameter("ky", 1.8)
        self.declare_parameter("kpsi", 2.0)

        # PRBS excitation amplitudes added on top of the stabilizing command.
        self.declare_parameter("prbs_v_amplitude", 0.08)
        self.declare_parameter("prbs_w_amplitude", 0.35)

        # Update period of each PRBS channel in control steps.
        self.declare_parameter("prbs_v_switch_steps", 6)
        self.declare_parameter("prbs_w_switch_steps", 8)

        # Separate non-zero seeds for the two LFSRs.
        self.declare_parameter("prbs_seed_v", 0xACE1)
        self.declare_parameter("prbs_seed_w", 0xBEEF)

        # Safety / recovery guard.
        self.declare_parameter("guard_pos_err", 0.35)
        self.declare_parameter("guard_yaw_err", 0.50)
        self.declare_parameter("guard_recovery_steps", 10)
        self.declare_parameter("disable_prbs_during_recovery", True)

        # Randomize operating point
        self.declare_parameter("randomize_operating_point", True)
        self.declare_parameter("switch_interval_steps", 200)

        self.declare_parameter("op_x_min", -0.8)
        self.declare_parameter("op_x_max", 0.8)
        self.declare_parameter("op_y_min", -0.8)
        self.declare_parameter("op_y_max", 0.8)
        self.declare_parameter("op_yaw_min", -3.14)
        self.declare_parameter("op_yaw_max", 3.14)

        self.declare_parameter("op_sample_mode", "global")   # global | around_start
        self.declare_parameter("op_dx", 0.5)                 # around_start일 때 x 범위
        self.declare_parameter("op_dy", 0.5)
        self.declare_parameter("op_dyaw", 1.57)

    def _load_prbs_parameters(self) -> None:
        self.dataset_steps = int(self.get_parameter("dataset_steps").value)
        self.progress_log_interval_steps = int(self.get_parameter("progress_log_interval_steps").value)
        if self.progress_log_interval_steps <= 0:
            raise ValueError("progress_log_interval_steps must be positive")

        self.use_explicit_target_pose = bool(self.get_parameter("use_explicit_target_pose").value)
        self.target_x = float(self.get_parameter("target_x").value)
        self.target_y = float(self.get_parameter("target_y").value)
        self.target_yaw = float(self.get_parameter("target_yaw").value)

        self.kx = float(self.get_parameter("kx").value)
        self.ky = float(self.get_parameter("ky").value)
        self.kpsi = float(self.get_parameter("kpsi").value)

        self.prbs_v_amplitude = float(self.get_parameter("prbs_v_amplitude").value)
        self.prbs_w_amplitude = float(self.get_parameter("prbs_w_amplitude").value)
        self.prbs_v_switch_steps = int(self.get_parameter("prbs_v_switch_steps").value)
        self.prbs_w_switch_steps = int(self.get_parameter("prbs_w_switch_steps").value)

        self.prbs_seed_v = int(self.get_parameter("prbs_seed_v").value)
        self.prbs_seed_w = int(self.get_parameter("prbs_seed_w").value)

        self.guard_pos_err = float(self.get_parameter("guard_pos_err").value)
        self.guard_yaw_err = float(self.get_parameter("guard_yaw_err").value)
        self.guard_recovery_steps = int(self.get_parameter("guard_recovery_steps").value)
        self.disable_prbs_during_recovery = bool(self.get_parameter("disable_prbs_during_recovery").value)

        self.randomize_operating_point = bool(self.get_parameter("randomize_operating_point").value)
        self.switch_interval_steps = int(self.get_parameter("switch_interval_steps").value)

        self.op_x_min = float(self.get_parameter("op_x_min").value)
        self.op_x_max = float(self.get_parameter("op_x_max").value)
        self.op_y_min = float(self.get_parameter("op_y_min").value)
        self.op_y_max = float(self.get_parameter("op_y_max").value)
        self.op_yaw_min = float(self.get_parameter("op_yaw_min").value)
        self.op_yaw_max = float(self.get_parameter("op_yaw_max").value)

        self.op_sample_mode = str(self.get_parameter("op_sample_mode").value)
        self.op_dx = float(self.get_parameter("op_dx").value)
        self.op_dy = float(self.get_parameter("op_dy").value)
        self.op_dyaw = float(self.get_parameter("op_dyaw").value)

        if self.dataset_steps <= 0:
            raise ValueError("dataset_steps must be positive")
        if self.prbs_v_switch_steps <= 0 or self.prbs_w_switch_steps <= 0:
            raise ValueError("prbs_*_switch_steps must be positive")
        if self.guard_recovery_steps < 0:
            raise ValueError("guard_recovery_steps must be >= 0")
        if self.switch_interval_steps < 0:
            raise ValueError("switch_interval_steps must be >= 0")
        if self.op_sample_mode not in ("global", "around_start"):
            raise ValueError("op_sample_mode must be 'global' or 'around_start'")

    def _init_prbs_state(self) -> None:
        self.target_pose: PoseTarget | None = None

        self._lfsr_v = self._sanitize_seed(self.prbs_seed_v)
        self._lfsr_w = self._sanitize_seed(self.prbs_seed_w)

        self.delta_v = 0.0
        self.delta_w = 0.0
        self.v_hold_count = 0
        self.w_hold_count = 0

        self.recovery_countdown = 0

    @staticmethod
    def _sanitize_seed(seed: int) -> int:
        seed = int(seed) & 0xFFFF
        return seed if seed != 0 else 0x1D0F

    @staticmethod
    def _advance_lfsr(state: int) -> int:
        # 16-bit Fibonacci LFSR with taps [16,14,13,11].
        # Period 65535 for any non-zero seed.
        bit = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
        state = ((state >> 1) | (bit << 15)) & 0xFFFF
        return state if state != 0 else 0x1D0F

    def _next_prbs_sign_v(self) -> float:
        self._lfsr_v = self._advance_lfsr(self._lfsr_v)
        return 1.0 if (self._lfsr_v & 1) else -1.0

    def _next_prbs_sign_w(self) -> float:
        self._lfsr_w = self._advance_lfsr(self._lfsr_w)
        return 1.0 if (self._lfsr_w & 1) else -1.0

    def on_ready_after_reset(self) -> None:
        x, y, yaw, _, _ = self.current_measured_state()

        self.start_x = x
        self.start_y = y
        self.start_yaw = yaw

        if self.use_explicit_target_pose:
            self.target_pose = PoseTarget(self.target_x, self.target_y, self.target_yaw)
        else:
            self.target_pose = PoseTarget(x, y, yaw)

        if self.randomize_operating_point:
            self.target_pose = self.sample_random_target_pose()
        elif self.use_explicit_target_pose:
            self.target_pose = PoseTarget(self.target_x, self.target_y, self.target_yaw)
        else:
            self.target_pose = PoseTarget(x, y, yaw)

        self.open_output_csv(
            stem="prbs",
            header=[
                "step", "sim_time_sec",
                "target_x", "target_y", "target_yaw",
                "x", "y", "yaw",
                "e_x", "e_y", "e_psi",
                "base_cmd_v", "base_cmd_w",
                "delta_v", "delta_w",
                "cmd_v", "cmd_w",
                "v_meas", "w_meas",
                "cmd_source",
            ],
        )

        self.get_logger().info("Started PRBS-on-stabilizer collection")
        self.get_logger().info(f"dataset_steps : {self.dataset_steps}")
        self.get_logger().info(f"warmup_steps  : {self.warmup_steps}")
        self.get_logger().info(f"output_file   : {self.csv_path}")
        self.get_logger().info(
            f"target_pose   : ({self.target_pose.x:+.3f}, {self.target_pose.y:+.3f}, {self.target_pose.yaw:+.3f})"
        )
        self.get_logger().info(
            f"prbs_amp      : dv={self.prbs_v_amplitude:+.3f}, dw={self.prbs_w_amplitude:+.3f}"
        )
        self.get_logger().info(
            f"prbs_switch   : v={self.prbs_v_switch_steps} steps, w={self.prbs_w_switch_steps} steps"
        )

    def sample_random_target_pose(self):
        if self.op_sample_mode == "around_start":
            cx = self.start_x
            cy = self.start_y
            cyaw = self.start_yaw

            tx = random.uniform(cx - self.op_dx, cx + self.op_dx)
            ty = random.uniform(cy - self.op_dy, cy + self.op_dy)
            tyaw = random.uniform(cyaw - self.op_dyaw, cyaw + self.op_dyaw)
        else:
            tx = random.uniform(self.op_x_min, self.op_x_max)
            ty = random.uniform(self.op_y_min, self.op_y_max)
            tyaw = random.uniform(self.op_yaw_min, self.op_yaw_max)

        return PoseTarget(tx, ty, tyaw)

    def compute_body_frame_error(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        assert self.target_pose is not None
        return body_frame_pose_error(
            self.target_pose.x,
            self.target_pose.y,
            self.target_pose.yaw,
            x,
            y,
            yaw,
        )

    def stabilizing_law(self, e_x: float, e_y: float, e_psi: float) -> tuple[float, float]:
        return unicycle_tracking_law(
            0.0,
            0.0,
            e_x,
            e_y,
            e_psi,
            kx=self.kx,
            ky=self.ky,
            kpsi=self.kpsi,
            v_min=self.v_min,
            v_max=self.v_max,
            w_min=self.w_min,
            w_max=self.w_max,
        )

    def update_prbs(self) -> None:
        if self.v_hold_count <= 0:
            self.delta_v = self.prbs_v_amplitude * self._next_prbs_sign_v()
            self.v_hold_count = self.prbs_v_switch_steps
        if self.w_hold_count <= 0:
            self.delta_w = self.prbs_w_amplitude * self._next_prbs_sign_w()
            self.w_hold_count = self.prbs_w_switch_steps

        self.v_hold_count -= 1
        self.w_hold_count -= 1

    def log_progress(
        self,
        cmd_v: float,
        cmd_w: float,
        v_meas: float,
        w_meas: float,
        e_x: float,
        e_y: float,
        e_psi: float,
        cmd_source: str,
    ) -> None:
        done = self.step_idx + 1
        total = self.dataset_steps
        percent = 100.0 * done / total
        remaining_steps = total - done
        remaining_sim_time = remaining_steps * self.dt
        pos_err = math.sqrt(e_x * e_x + e_y * e_y)

        self.get_logger().info(
            f"progress={done:4d}/{total} ({percent:5.1f}%) "
            f"remaining={remaining_steps:4d} steps (~{remaining_sim_time:6.2f} s sim) "
            f"src={cmd_source} "
            f"err=(pos:{pos_err:.3f}, yaw:{abs(e_psi):.3f}) "
            f"cmd=({cmd_v:+.3f}, {cmd_w:+.3f}) "
            f"meas=({v_meas:+.3f}, {w_meas:+.3f})"
        )

    def on_timer(self) -> None:
        if self.finished or self.waiting_for_reset:
            return
        if self.last_odom is None or self.target_pose is None:
            self.publish_cmd(0.0, 0.0)
            return

        if self.step_idx >= self.dataset_steps:
            self.finish_and_shutdown()
            return

        if (
            self.randomize_operating_point
            and self.step_idx >= self.warmup_steps
            and self.switch_interval_steps > 0
            and self.step_idx % self.switch_interval_steps == 0
        ):
            self.target_pose = self.sample_random_target_pose()
            self.get_logger().info(
                f"new operating point: "
                f"x={self.target_pose.x:+.3f}, "
                f"y={self.target_pose.y:+.3f}, "
                f"yaw={self.target_pose.yaw:+.3f}"
            )

        x, y, yaw, v_meas, w_meas = self.current_measured_state()
        e_x, e_y, e_psi = self.compute_body_frame_error(x, y, yaw)

        base_cmd_v, base_cmd_w = self.stabilizing_law(e_x, e_y, e_psi)

        pos_err = math.sqrt(e_x * e_x + e_y * e_y)
        out_of_guard = (pos_err > self.guard_pos_err) or (abs(e_psi) > self.guard_yaw_err)
        if out_of_guard:
            self.recovery_countdown = self.guard_recovery_steps
        elif self.recovery_countdown > 0:
            self.recovery_countdown -= 1

        if self.step_idx < self.warmup_steps:
            delta_v = 0.0
            delta_w = 0.0
            cmd_source = "warmup"
        elif self.disable_prbs_during_recovery and self.recovery_countdown > 0:
            delta_v = 0.0
            delta_w = 0.0
            cmd_source = "recovery"
        else:
            self.update_prbs()
            delta_v = self.delta_v
            delta_w = self.delta_w
            cmd_source = "baseline_plus_prbs"

        cmd_v = clamp(base_cmd_v + delta_v, self.v_min, self.v_max)
        cmd_w = clamp(base_cmd_w + delta_w, self.w_min, self.w_max)
        self.publish_cmd(cmd_v, cmd_w)

        sim_time = self.get_clock().now().nanoseconds * 1e-9
        self.write_csv_row([
            self.step_idx, f"{sim_time:.6f}",
            f"{self.target_pose.x:.6f}", f"{self.target_pose.y:.6f}", f"{self.target_pose.yaw:.6f}",
            f"{x:.6f}", f"{y:.6f}", f"{yaw:.6f}",
            f"{e_x:.6f}", f"{e_y:.6f}", f"{e_psi:.6f}",
            f"{base_cmd_v:.6f}", f"{base_cmd_w:.6f}",
            f"{delta_v:.6f}", f"{delta_w:.6f}",
            f"{cmd_v:.6f}", f"{cmd_w:.6f}",
            f"{v_meas:.6f}", f"{w_meas:.6f}",
            cmd_source,
        ])

        is_last = (self.step_idx + 1) >= self.dataset_steps
        should_log = (
            self.step_idx == 0
            or ((self.step_idx + 1) % self.progress_log_interval_steps == 0)
            or is_last
        )
        if should_log:
            self.log_progress(cmd_v, cmd_w, v_meas, w_meas, e_x, e_y, e_psi, cmd_source)

        self.step_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = PRBSCollectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish_and_shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
