import os
import random
from pathlib import Path as FilePath
from typing import List

import rclpy
from rclpy.parameter import Parameter
from nav_msgs.msg import Path

from .collect_base import CollectBase
from .utils import (
    RefPoint,
    align_reference_yaw_to_reset_branch,
    body_frame_pose_error,
    build_path_msg,
    clamp,
    load_reference_csv,
    signed_angle_diff,
    unicycle_tracking_law,
    wrap_to_pi,
)


class ReferenceCollectNode(CollectBase):
    def __init__(self):
        super().__init__('reference_collect_node')

        self._declare_reference_parameters()
        self._load_reference_parameters()
        self._create_reference_interfaces()
        self._init_reference_state()

        self.timer = None
        self.schedule_startup(self.begin_collection)

    def begin_collection(self):
        self.start_next_reference()
        self.timer = self.create_timer(self.dt, self.on_timer)

    def _declare_reference_parameters(self):
        self.declare_parameter('path_topic', '/deepc/reference_path')
        self.declare_parameter('append_final_stop_steps', 20)
        self.declare_parameter('reference_csv_list', Parameter.Type.STRING_ARRAY)

        self.declare_parameter('kx', 0.8)
        self.declare_parameter('ky', 1.8)
        self.declare_parameter('kpsi', 2.0)

        # perturbation parameters
        self.declare_parameter('perturb_enable', False)
        self.declare_parameter('perturb_seed', -1)
        self.declare_parameter('perturb_v_min', -0.03)
        self.declare_parameter('perturb_v_max', 0.03)
        self.declare_parameter('perturb_w_min', -0.15)
        self.declare_parameter('perturb_w_max', 0.15)
        self.declare_parameter('perturb_hold_steps_min', 2)
        self.declare_parameter('perturb_hold_steps_max', 5)

    def _load_reference_parameters(self):
        self.path_topic = str(self.get_parameter('path_topic').value)
        self.append_final_stop_steps = int(self.get_parameter('append_final_stop_steps').value)
        self.reference_csv_list = list(self.get_parameter('reference_csv_list').value)

        self.kx = float(self.get_parameter('kx').value)
        self.ky = float(self.get_parameter('ky').value)
        self.kpsi = float(self.get_parameter('kpsi').value)

        self.perturb_enable = bool(self.get_parameter('perturb_enable').value)
        self.perturb_seed = int(self.get_parameter('perturb_seed').value)
        self.perturb_v_min = float(self.get_parameter('perturb_v_min').value)
        self.perturb_v_max = float(self.get_parameter('perturb_v_max').value)
        self.perturb_w_min = float(self.get_parameter('perturb_w_min').value)
        self.perturb_w_max = float(self.get_parameter('perturb_w_max').value)
        self.perturb_hold_steps_min = int(self.get_parameter('perturb_hold_steps_min').value)
        self.perturb_hold_steps_max = int(self.get_parameter('perturb_hold_steps_max').value)

        if (
            self.perturb_hold_steps_min <= 0
            or self.perturb_hold_steps_max < self.perturb_hold_steps_min
        ):
            raise ValueError("Invalid perturb hold-step range")

    def _create_reference_interfaces(self):
        self.path_pub = self.create_publisher(Path, self.path_topic, 1)

    def _init_reference_state(self):
        self.reference_paths = self.resolve_reference_paths()
        self.reference_file_index = -1
        self.current_reference_path = ''
        self.current_reference_name = ''
        self.ref_traj: List[RefPoint] = []
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

        self.rng = random.Random()
        if self.perturb_seed >= 0:
            self.rng.seed(self.perturb_seed)

        self.current_perturb_v = 0.0
        self.current_perturb_w = 0.0
        self.perturb_hold_count = 0

    def resolve_reference_paths(self) -> List[str]:
        paths = [str(p) for p in self.reference_csv_list if str(p).strip()]
        if not paths:
            raise RuntimeError("No reference CSVs were provided")
        unique = []
        seen = set()
        for p in paths:
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                unique.append(ap)
        return unique

    def start_next_reference(self):
        self.reference_file_index += 1
        if self.reference_file_index >= len(self.reference_paths):
            self.get_logger().info("All references completed")
            self.finish_and_shutdown()
            return

        self.current_reference_path = self.reference_paths[self.reference_file_index]
        self.current_reference_name = FilePath(self.current_reference_path).stem
        self.ref_traj = load_reference_csv(
            self.current_reference_path, self.dt, self.append_final_stop_steps
        )
        if self.uses_unwrapped_yaw:
            align_reference_yaw_to_reset_branch(self.ref_traj)
        else:
            self.wrap_reference_yaw_in_place()
        self.publish_reference_path()

        self.step_idx = 0
        self._clear_odom_state(reset_yaw=self.reset_before_start)
        self.current_perturb_v = 0.0
        self.current_perturb_w = 0.0
        self.perturb_hold_count = 0

        self.start_with_optional_reset(reset_pose=self.current_reference_reset_pose())
        self.get_logger().info(f"Started reference '{self.current_reference_name}' with {len(self.ref_traj)} steps")

    def on_ready_after_reset(self):
        self.open_output_csv(
            stem=f"reference_{self.reference_file_index:02d}_{self.current_reference_name}",
            header=[
                'step', 'sim_time_sec',
                'ref_x', 'ref_y', 'ref_yaw', 'ref_v', 'ref_w',
                'cmd_v', 'cmd_w',
                'x', 'y', 'yaw', 'v_meas', 'w_meas',
            ]
        )

    def publish_reference_path(self):
        self.path_msg = build_path_msg(self.ref_traj, frame_id='odom')

    def current_reference_reset_pose(self):
        if not self.ref_traj:
            raise RuntimeError("Cannot reset to an empty reference trajectory")
        ref = self.ref_traj[0]
        return ref.x, ref.y, wrap_to_pi(ref.yaw)

    def wrap_reference_yaw_in_place(self):
        for ref in self.ref_traj:
            ref.yaw = wrap_to_pi(ref.yaw)

        last_idx = len(self.ref_traj) - 1
        for idx, ref in enumerate(self.ref_traj):
            if idx < last_idx:
                next_yaw = self.ref_traj[idx + 1].yaw
                ref.w = signed_angle_diff(next_yaw - ref.yaw) / max(self.dt, 1.0e-9)
            else:
                ref.w = 0.0

    def compute_body_frame_error(self, ref, x, y, yaw):
        return body_frame_pose_error(ref.x, ref.y, ref.yaw, x, y, yaw)

    def baseline_tracking_law(self, ref, e_x, e_y, e_psi):
        return unicycle_tracking_law(
            ref.v,
            ref.w,
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

    def update_perturbation(self):
        if not self.perturb_enable:
            self.current_perturb_v = 0.0
            self.current_perturb_w = 0.0
            return

        if self.perturb_hold_count <= 0:
            self.current_perturb_v = self.rng.uniform(self.perturb_v_min, self.perturb_v_max)
            self.current_perturb_w = self.rng.uniform(self.perturb_w_min, self.perturb_w_max)
            self.perturb_hold_count = self.rng.randint(
                self.perturb_hold_steps_min,
                self.perturb_hold_steps_max
            )

        self.perturb_hold_count -= 1

    def apply_perturbation(self, cmd_v_nominal, cmd_w_nominal):
        self.update_perturbation()
        cmd_v = clamp(cmd_v_nominal + self.current_perturb_v, self.v_min, self.v_max)
        cmd_w = clamp(cmd_w_nominal + self.current_perturb_w, self.w_min, self.w_max)
        return cmd_v, cmd_w

    def on_timer(self):
        if self.finished or self.waiting_for_reset:
            return

        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.path_msg)

        if self.last_odom is None:
            self.publish_cmd(0.0, 0.0)
            return

        if self.writer is None:
            self.publish_cmd(0.0, 0.0)
            return

        if self.step_idx >= len(self.ref_traj):
            self.get_logger().info(f"Finished reference '{self.current_reference_name}'")
            self.close_output_csv()
            self.start_next_reference()
            return

        ref = self.ref_traj[self.step_idx]
        x, y, yaw, v_meas, w_meas = self.current_measured_state()
        e_x, e_y, e_psi = self.compute_body_frame_error(ref, x, y, yaw)

        cmd_v_nominal, cmd_w_nominal = self.baseline_tracking_law(ref, e_x, e_y, e_psi)
        cmd_v, cmd_w = self.apply_perturbation(cmd_v_nominal, cmd_w_nominal)

        self.publish_cmd(cmd_v, cmd_w)

        sim_time = self.get_clock().now().nanoseconds * 1e-9
        self.write_csv_row([
            self.step_idx, f'{sim_time:.6f}',
            f'{ref.x:.6f}', f'{ref.y:.6f}', f'{ref.yaw:.6f}', f'{ref.v:.6f}', f'{ref.w:.6f}',
            f'{cmd_v:.6f}', f'{cmd_w:.6f}',
            f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}', f'{v_meas:.6f}', f'{w_meas:.6f}'
        ])
        if self.step_idx % 50 == 0:
            self.get_logger().info(f"{self.step_idx}/{len(self.ref_traj)} complete")
        self.step_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = ReferenceCollectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish_and_shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
