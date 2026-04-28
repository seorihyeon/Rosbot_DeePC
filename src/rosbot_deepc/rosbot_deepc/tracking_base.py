import csv
import math
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from nav_msgs.msg import Path
from ros_gz_interfaces.srv import ControlWorld

from .runtime_base import RuntimeBase
from .utils import (
    RefPoint,
    body_frame_pose_error,
    build_path_msg,
    load_reference_csv,
    make_pose_stamped,
    signed_angle_diff,
    wrap_to_pi,
)


class TrackingBase(RuntimeBase):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._declare_tracking_parameters()
        self._load_tracking_parameters()
        self._create_tracking_interfaces()
        self._init_tracking_state()

    def _declare_tracking_parameters(self) -> None:
        self.declare_parameter("path_topic", "/deepc/reference_path")
        self.declare_parameter("actual_path_topic", "/deepc/actual_path")

        self.declare_parameter("reference_csv", "")
        self.declare_parameter("append_final_stop_steps", 20)

        self.declare_parameter("goal_pos_tol", 0.08)
        self.declare_parameter("goal_yaw_tol", 0.15)
        self.declare_parameter("final_hold_steps", 10)
        self.declare_parameter("max_final_stop_steps", 200)

        self.declare_parameter("tracking_preview_steps", 6)
        self.declare_parameter("ref_back_search", 3)
        self.declare_parameter("ref_forward_search", 25)
        self.declare_parameter("ref_progress_eps", 0.02)

        self.declare_parameter("abort_if_far_from_reference", True)
        self.declare_parameter("abort_max_pos_err", 0.35)
        self.declare_parameter("abort_far_steps", 5)
        self.declare_parameter("abort_ignore_first_steps", 20)

        self.declare_parameter("output_dir", "/ws/results")
        self.declare_parameter("file_prefix", "deepc_run")

        self.declare_parameter("stepped_mode", False)
        self.declare_parameter("world_control_service", "/world/flat_empty/control")
        self.declare_parameter("physics_dt", 0.001)
        self.declare_parameter("step_service_timeout_sec", 2.0)
        self.declare_parameter("new_odom_timeout_sec", 2.0)

    def _load_tracking_parameters(self) -> None:
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.actual_path_topic = str(self.get_parameter("actual_path_topic").value)

        self.reference_csv = str(self.get_parameter("reference_csv").value)
        self.append_final_stop_steps = int(
            self.get_parameter("append_final_stop_steps").value
        )

        self.goal_pos_tol = float(self.get_parameter("goal_pos_tol").value)
        self.goal_yaw_tol = float(self.get_parameter("goal_yaw_tol").value)
        self.final_hold_steps = int(self.get_parameter("final_hold_steps").value)
        self.max_final_stop_steps = int(
            self.get_parameter("max_final_stop_steps").value
        )

        self.tracking_preview_steps = int(
            self.get_parameter("tracking_preview_steps").value
        )
        self.ref_back_search = int(self.get_parameter("ref_back_search").value)
        self.ref_forward_search = int(self.get_parameter("ref_forward_search").value)
        self.ref_progress_eps = float(self.get_parameter("ref_progress_eps").value)

        self.abort_if_far_from_reference = bool(
            self.get_parameter("abort_if_far_from_reference").value
        )
        self.abort_max_pos_err = float(
            self.get_parameter("abort_max_pos_err").value
        )
        self.abort_far_steps = int(self.get_parameter("abort_far_steps").value)
        self.abort_ignore_first_steps = int(
            self.get_parameter("abort_ignore_first_steps").value
        )

        self.output_dir = str(self.get_parameter("output_dir").value)
        self.file_prefix = str(self.get_parameter("file_prefix").value)

        self.stepped_mode = bool(self.get_parameter("stepped_mode").value)
        self.world_control_service = str(
            self.get_parameter("world_control_service").value
        )
        self.physics_dt = float(self.get_parameter("physics_dt").value)
        self.step_service_timeout_sec = float(
            self.get_parameter("step_service_timeout_sec").value
        )
        self.new_odom_timeout_sec = float(
            self.get_parameter("new_odom_timeout_sec").value
        )

        self.multi_step_count = max(1, int(round(self.dt / self.physics_dt)))

    def _create_tracking_interfaces(self) -> None:
        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        self.actual_path_pub = self.create_publisher(Path, self.actual_path_topic, 1)

    def _init_tracking_state(self) -> None:
        self.step_idx = 0
        self.ref_idx = 0

        self.final_reached_count = 0
        self.final_stop_count = 0
        self.abort_far_count = 0

        self.run_rows = []
        self.current_run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.ref_traj = []
        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"
        self.actual_path_msg = Path()
        self.actual_path_msg.header.frame_id = "odom"

        self.control_thread = None
        self.world_control_cli = None
        self.timer = None
        self.runtime_started = False

    def begin_runtime(self) -> None:
        if self.runtime_started:
            return

        self.runtime_started = True
        if self.stepped_mode:
            self.world_control_cli = self.create_client(
                ControlWorld,
                self.world_control_service,
            )
            self.control_thread = threading.Thread(
                target=self.control_loop,
                daemon=True,
            )
            self.control_thread.start()
        else:
            self.timer = self.create_timer(self.dt, self.on_timer)

    def begin_tracking(self) -> None:
        self.start_with_optional_reset()

    def on_ready_after_reset(self) -> None:
        self.begin_runtime()

    def _load_reference(self) -> None:
        self.ref_traj = load_reference_csv(
            self.reference_csv,
            self.dt,
            self.append_final_stop_steps,
        )
        if not self.uses_unwrapped_yaw:
            self._wrap_reference_yaw_in_place()
        self.path_msg = self.build_reference_path_msg()

    def _wrap_reference_yaw_in_place(self) -> None:
        for ref in self.ref_traj:
            ref.yaw = wrap_to_pi(ref.yaw)

        last_idx = len(self.ref_traj) - 1
        for idx, ref in enumerate(self.ref_traj):
            if idx < last_idx:
                next_yaw = self.ref_traj[idx + 1].yaw
                ref.w = signed_angle_diff(next_yaw - ref.yaw) / max(self.dt, 1.0e-9)
            else:
                ref.w = 0.0

    def build_reference_path_msg(self) -> Path:
        return build_path_msg(self.ref_traj, frame_id="odom")

    def wait_for_new_odom(self, prev_stamp_ns: int) -> bool:
        deadline = time.monotonic() + self.new_odom_timeout_sec

        while time.monotonic() < deadline and rclpy.ok():
            with self.odom_lock:
                if self.last_odom_stamp_ns > prev_stamp_ns:
                    return True

            self.new_odom_event.wait(timeout=0.01)
            self.new_odom_event.clear()

        return False

    def compute_body_frame_error(
        self,
        ref: RefPoint,
        x: float,
        y: float,
        yaw: float,
    ) -> Tuple[float, float, float]:
        return body_frame_pose_error(
            ref.x,
            ref.y,
            ref.yaw,
            x,
            y,
            yaw,
            wrap_yaw_error=not self.uses_unwrapped_yaw,
        )

    def get_tracking_ref(self) -> RefPoint:
        idx = min(self.ref_idx + self.tracking_preview_steps, len(self.ref_traj) - 1)
        return self.ref_traj[idx]

    def find_nearest_reference_point(
        self,
        x: float,
        y: float,
        *,
        allow_backtrack: bool = True,
    ) -> Tuple[int, float]:
        start = (
            max(0, self.ref_idx - self.ref_back_search)
            if allow_backtrack
            else self.ref_idx
        )
        stop = min(len(self.ref_traj), self.ref_idx + self.ref_forward_search + 1)

        best_idx = self.ref_idx
        best_dist2 = float("inf")

        for idx in range(start, stop):
            rp = self.ref_traj[idx]
            dx = rp.x - x
            dy = rp.y - y
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_idx = idx

        return best_idx, math.sqrt(best_dist2)

    def find_progress_reference_point(self, x: float, y: float) -> Tuple[int, float]:
        start = self.ref_idx
        stop = min(len(self.ref_traj), self.ref_idx + self.ref_forward_search + 1)

        best_idx = self.ref_idx
        best_dist = float("inf")
        furthest_close_idx = self.ref_idx
        furthest_close_dist = float("inf")

        for idx in range(start, stop):
            rp = self.ref_traj[idx]
            dist = math.hypot(rp.x - x, rp.y - y)

            if dist <= best_dist:
                best_dist = dist
                best_idx = idx

            if dist <= self.ref_progress_eps:
                furthest_close_idx = idx
                furthest_close_dist = dist

        if furthest_close_idx > self.ref_idx:
            return furthest_close_idx, furthest_close_dist

        return best_idx, best_dist

    def update_reference_index(self, x: float, y: float) -> None:
        if self.ref_idx >= len(self.ref_traj) - 1:
            return

        next_idx, _ = self.find_progress_reference_point(x, y)
        if next_idx > self.ref_idx:
            self.ref_idx = next_idx

    def append_actual_path(self, x: float, y: float, yaw: float) -> None:
        ps = make_pose_stamped(
            x,
            y,
            yaw,
            frame_id="odom",
            stamp=self.get_clock().now().to_msg(),
        )

        self.actual_path_msg.poses.append(ps)
        self.actual_path_msg.header.stamp = ps.header.stamp
        self.actual_path_pub.publish(self.actual_path_msg)

    def log_step(
        self,
        mode: str,
        ref: RefPoint,
        x: float,
        y: float,
        yaw: float,
        v_meas: float,
        w_meas: float,
        e_x: float,
        e_y: float,
        e_psi: float,
        cmd_v: float,
        cmd_w: float,
    ) -> None:
        sim_time = self.get_clock().now().nanoseconds * 1e-9

        self.run_rows.append({
            "step": self.step_idx,
            "mode": mode,
            "sim_time_sec": sim_time,
            "ref_idx": self.ref_idx,
            "ref_x": ref.x,
            "ref_y": ref.y,
            "ref_yaw": ref.yaw,
            "ref_v": ref.v,
            "ref_w": ref.w,
            "x": x,
            "y": y,
            "yaw": yaw,
            "v_meas": v_meas,
            "w_meas": w_meas,
            "e_x": e_x,
            "e_y": e_y,
            "e_psi": e_psi,
            "cmd_v": cmd_v,
            "cmd_w": cmd_w,
        })

    def save_run_csv(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(
            self.output_dir,
            f"{self.file_prefix}_run_{self.current_run_stamp}.csv",
        )

        fieldnames = [
            "step", "mode", "sim_time_sec", "ref_idx",
            "ref_x", "ref_y", "ref_yaw", "ref_v", "ref_w",
            "x", "y", "yaw", "v_meas", "w_meas",
            "e_x", "e_y", "e_psi",
            "cmd_v", "cmd_w",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.run_rows)

        self._safe_info(f"Saved run log to: {csv_path}")

    def save_additional_tracking_outputs(self) -> None:
        pass

    def finish(self) -> None:
        if self.finished:
            return

        self.finished = True
        self.cancel_common_timers()

        self.get_logger().info(
            "Reference tracking finished. Sending zero command and saving results..."
        )

        self.publish_stop_commands()
        self.save_run_csv()
        self.save_additional_tracking_outputs()
        self.request_shutdown()

    def check_finish_condition(self, e_x: float, e_y: float, e_psi: float) -> None:
        pos_err = math.sqrt(e_x * e_x + e_y * e_y)
        yaw_err = abs(e_psi)

        is_last_ref = self.ref_idx >= len(self.ref_traj) - 1

        if is_last_ref:
            self.final_stop_count += 1

            if pos_err <= self.goal_pos_tol and yaw_err <= self.goal_yaw_tol:
                self.final_reached_count += 1
            else:
                self.final_reached_count = 0

            if self.final_reached_count >= self.final_hold_steps:
                self.get_logger().info("Goal tolerance satisfied at final reference.")
                self.finish()
                return

            if self.final_stop_count >= self.max_final_stop_steps:
                self.get_logger().warn(
                    "Final stop timeout reached. Finishing run anyway."
                )
                self.finish()
                return
        else:
            self.final_stop_count = 0
            self.final_reached_count = 0

    def check_abort_condition(self, x: float, y: float) -> None:
        if not self.abort_if_far_from_reference:
            return

        if self.step_idx < self.abort_ignore_first_steps:
            return

        nearest_idx, pos_err = self.find_nearest_reference_point(x, y)
        too_far = pos_err > self.abort_max_pos_err

        if too_far:
            self.abort_far_count += 1
        else:
            self.abort_far_count = 0

        if self.abort_far_count >= self.abort_far_steps:
            self.get_logger().warn(
                "Abort: robot diverged from reference. "
                f"nearest_ref_idx={nearest_idx}, "
                f"pos_err={pos_err:.3f} m, "
                f"thresholds=({self.abort_max_pos_err:.3f} m)"
            )
            self.finish()

    def _safe_info(self, msg: str) -> None:
        try:
            if rclpy.ok():
                self.get_logger().info(msg)
            else:
                print(msg, flush=True)
        except Exception:
            print(msg, flush=True)

    def control_loop(self) -> None:
        self.get_logger().info(
            f"Stepped mode enabled. service = {self.world_control_service}, "
            f"sample_time={self.dt:.4f}, physics_dt={self.physics_dt:.4f}, "
            f"multi_step={self.multi_step_count}"
        )

        deadline = time.monotonic() + self.new_odom_timeout_sec
        while rclpy.ok() and self.last_odom is None and time.monotonic() < deadline:
            time.sleep(0.01)

        if not rclpy.ok():
            return

        if self.last_odom is None:
            self.get_logger().error(
                f"No odom received within {self.new_odom_timeout_sec:.1f} sec "
                "before stepped control start."
            )
            self.finish()
            return

        if not self.call_world_control(pause=True):
            self.get_logger().error(
                "Failed to pause Gazebo world before stepped control."
            )
            self.finish()
            return

        while rclpy.ok() and not self.finished:
            if self.last_odom is None:
                time.sleep(0.001)
                continue

            prev_stamp_ns = self.get_last_odom_stamp_ns()

            self.control_once()
            if self.finished:
                break

            ok = self.call_world_control(multi_step=self.multi_step_count)
            if not ok:
                self.get_logger().error("Failed to call multi_step on Gazebo world.")
                self.finish()
                break

            if not self.wait_for_new_odom(prev_stamp_ns):
                self.get_logger().warn(
                    f"No new odom received after multi_step={self.multi_step_count}."
                )
                self.finish()
                break

    def call_world_control(
        self,
        pause: Optional[bool] = None,
        multi_step: Optional[int] = None,
    ) -> bool:
        if self.world_control_cli is None:
            return False

        if not self.world_control_cli.wait_for_service(
            timeout_sec=self.step_service_timeout_sec
        ):
            self.get_logger().error(
                "World control service not available within "
                f"{self.step_service_timeout_sec:.1f} sec: "
                f"{self.world_control_service}"
            )
            return False

        req = ControlWorld.Request()

        if pause is not None:
            req.world_control.pause = bool(pause)

        if multi_step is not None:
            req.world_control.multi_step = int(multi_step)

        future = None
        try:
            future = self.world_control_cli.call_async(req)
            deadline = time.monotonic() + self.step_service_timeout_sec

            while rclpy.ok() and time.monotonic() < deadline:
                if future.done():
                    resp = future.result()
                    return bool(resp.success)
                time.sleep(0.001)
        except Exception as exc:
            self.get_logger().error(f"World control call failed: {exc}")
            return False

        if future is not None:
            future.cancel()

        self.get_logger().error(
            f"World control call timed out after {self.step_service_timeout_sec:.1f} sec"
        )
        return False

    def on_timer(self) -> None:
        self.control_once()

    def control_once(self) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        self.cancel_common_timers()

        try:
            if self.control_thread is not None and self.control_thread.is_alive():
                self.control_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            self.publish_stop_commands()
        except Exception as exc:
            print(f"cleanup publish failed: {exc}", file=sys.stderr, flush=True)

        print(
            "cleanup: "
            f"run_rows={len(self.run_rows)}, "
            f"pred_rows={len(getattr(self, 'pred_rows', []))}, "
            f"finished={self.finished}",
            flush=True,
        )

        try:
            if self.run_rows and not self.finished:
                self.save_run_csv()
        except Exception as exc:
            print(f"save_run_csv failed: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()

        try:
            if not self.finished:
                self.save_additional_tracking_outputs()
        except Exception as exc:
            print(
                f"save_additional_tracking_outputs failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
