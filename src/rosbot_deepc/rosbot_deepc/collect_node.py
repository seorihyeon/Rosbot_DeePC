#!/usr/bin/env python3
import csv, math, os, random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry, Path

from .utils import wrap_to_pi, quat_to_yaw, clamp

@dataclass
class RefPoint:
    t: float
    x: float
    y: float
    yaw: float
    v: float
    w: float
    segment: str


class CollectNode(Node):
    def __init__(self) -> None:
        super().__init__('collect_node')

        # Parameter delcaration
        ## topic
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/model/rosbot/odometry_gt')
        self.declare_parameter('path_topic', '/deepc/reference_path')

        ## Sample time
        self.declare_parameter('sample_time', 0.05)

        ## Path definition
        self.declare_parameter('straight_1_length', 1.5)
        self.declare_parameter('straight_2_length', 1.0)
        self.declare_parameter('straight_3_length', 1.5)
        self.declare_parameter('straight_speed', 0.25)

        self.declare_parameter('turn_radius', 0.8)
        self.declare_parameter('turn_angle_deg', 90.0)
        self.declare_parameter('turn_speed', 0.20)

        ## Baseline tracker gain
        self.declare_parameter('kx', 0.8)
        self.declare_parameter('ky', 1.8)
        self.declare_parameter('kpsi', 2.0)

        ## Command limits
        self.declare_parameter('v_min', -0.2)
        self.declare_parameter('v_max', 0.45)
        self.declare_parameter('w_min', -1.2)
        self.declare_parameter('w_max', 1.2)

        ## Perturbation setting
        self.declare_parameter('enable_perturbation', True)
        self.declare_parameter('perturb_after_warmup_only', True)

        self.declare_parameter('dv_perturb_min', -0.06)
        self.declare_parameter('dv_perturb_max', 0.06)
        self.declare_parameter('dw_perturb_min', -0.35)
        self.declare_parameter('dw_perturb_max', 0.35)

        self.declare_parameter('perturb_hold_steps_min', 4)
        self.declare_parameter('perturb_hold_steps_max', 10)

        self.declare_parameter('min_perturb_norm', 0.03)

        ## Output
        self.declare_parameter('output_dir', '/ws/src/rosbot_deepc/datasets')
        self.declare_parameter('file_prefix', 'collect')
        self.declare_parameter('warmup_steps', 20)

        # Define parameter variance
        self.cmd_topic = str(self.get_parameter('cmd_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.path_topic = str(self.get_parameter('path_topic').value)

        self.dt = float(self.get_parameter('sample_time').value)

        self.straight_1_length = float(self.get_parameter('straight_1_length').value)
        self.straight_2_length = float(self.get_parameter('straight_2_length').value)
        self.straight_3_length = float(self.get_parameter('straight_3_length').value)
        self.straight_speed = float(self.get_parameter('straight_speed').value)

        self.turn_radius = float(self.get_parameter('turn_radius').value)
        self.turn_angle_deg = float(self.get_parameter('turn_angle_deg').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)

        self.kx = float(self.get_parameter('kx').value)
        self.ky = float(self.get_parameter('ky').value)
        self.kpsi = float(self.get_parameter('kpsi').value)

        self.v_min = float(self.get_parameter('v_min').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.w_min = float(self.get_parameter('w_min').value)
        self.w_max = float(self.get_parameter('w_max').value)

        self.enable_perturbation = bool(self.get_parameter('enable_perturbation').value)
        self.perturb_after_warmup_only = bool(self.get_parameter('perturb_after_warmup_only').value)

        self.dv_perturb_min = float(self.get_parameter('dv_perturb_min').value)
        self.dv_perturb_max = float(self.get_parameter('dv_perturb_max').value)
        self.dw_perturb_min = float(self.get_parameter('dw_perturb_min').value)
        self.dw_perturb_max = float(self.get_parameter('dw_perturb_max').value)

        self.perturb_hold_steps_min = int(self.get_parameter('perturb_hold_steps_min').value)
        self.perturb_hold_steps_max = int(self.get_parameter('perturb_hold_steps_max').value)

        self.min_perturb_norm = float(self.get_parameter('min_perturb_norm').value)

        self.output_dir = str(self.get_parameter('output_dir').value)
        self.file_prefix = str(self.get_parameter('file_prefix').value)
        self.warmup_steps = int(self.get_parameter('warmup_steps').value)

        # ROS interfaces
        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)
        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)

        # State
        self.last_odom: Optional[Odometry] = None
        self.step_idx = 0
        self.finished = False

        self.current_dv_pert = 0.0
        self.current_dw_pert = 0.0
        self.perturb_hold_count = 0

        # Build reference trajectory
        self.ref_traj: List[RefPoint] = self.build_reference_trajectory()
        self.publish_reference_path()

        # CSV
        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.output_dir, f'{self.file_prefix}_{stamp}.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            'step', 'segment', 'sim_time_sec',
            'ref_x', 'ref_y', 'ref_yaw', 'ref_v', 'ref_w',
            'x', 'y', 'yaw', 'v_meas', 'w_meas',
            'e_x', 'e_y', 'e_psi',
            'cmd_v', 'cmd_w',
            'cmd_v_nom', 'cmd_w_nom', 'dv_pert', 'dw_pert'
        ])
        self.csv_file.flush()

        self.timer = self.create_timer(self.dt, self.on_timer)

        self.get_logger().info('Data collection node started.')
        self.get_logger().info(f'odom_topic  : {self.odom_topic}')
        self.get_logger().info(f'cmd_topic   : {self.cmd_topic}')
        self.get_logger().info(f'sample_time : {self.dt:.3f} s')
        self.get_logger().info(f'ref_steps   : {len(self.ref_traj)}')
        self.get_logger().info(f'output_file : {self.csv_path}')

    def build_reference_trajectory(self) -> List[RefPoint]:
        traj: List[RefPoint] = []

        t = 0.0
        x = 0.0
        y = 0.0
        yaw = 0.0
        turn_angle = math.radians(self.turn_angle_deg)

        def append_straight(length: float, v: float, name: str):
            nonlocal x, y, yaw, t, traj
            steps = max(1, int(round(length / max(abs(v), 1e-6) / self.dt)))
            for _ in range(steps):
                traj.append(RefPoint(t=t, x=x, y=y, yaw=yaw, v=v, w=0.0, segment=name))
                x += v * math.cos(yaw) * self.dt
                y += v * math.sin(yaw) * self.dt
                t += self.dt

        def append_turn(radius: float, angle: float, v: float, left: bool, name: str):
            nonlocal x, y, yaw, t, traj
            w = abs(v) / max(radius, 1e-6)
            if not left:
                w = -w

            steps = max(1, int(round(abs(angle) / max(abs(w), 1e-6) / self.dt)))
            for _ in range(steps):
                traj.append(RefPoint(t=t, x=x, y=y, yaw=yaw, v=v, w=w, segment=name))
                x += v * math.cos(yaw) * self.dt
                y += v * math.sin(yaw) * self.dt
                yaw += w * self.dt
                t += self.dt

        append_straight(self.straight_1_length, self.straight_speed, 'straight_1')
        append_turn(self.turn_radius, turn_angle, self.turn_speed, True, 'left_turn')
        append_straight(self.straight_2_length, self.straight_speed, 'straight_2')
        append_turn(self.turn_radius, turn_angle, self.turn_speed, False, 'right_turn')
        append_straight(self.straight_3_length, self.straight_speed, 'straight_3')

        for _ in range(20):
            traj.append(RefPoint(t=t, x=x, y=y, yaw=yaw, v=0.0, w=0.0, segment='final_stop'))
            t += self.dt

        return traj

    def publish_reference_path(self) -> None:
        msg = Path()
        msg.header.frame_id = 'odom'

        for p in self.ref_traj:
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x = p.x
            ps.pose.position.y = p.y
            ps.pose.position.z = 0.0
            qz = math.sin(p.yaw / 2.0)
            qw = math.cos(p.yaw / 2.0)
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            msg.poses.append(ps)

        self.path_msg = msg

    def on_odom(self, msg: Odometry) -> None:
        self.last_odom = msg

    def publish_cmd(self, v: float, w: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def current_measured_state(self):
        odom = self.last_odom
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        v = odom.twist.twist.linear.x
        w = odom.twist.twist.angular.z
        return x, y, yaw, v, w

    def compute_body_frame_error(self, ref: RefPoint, x: float, y: float, yaw: float):
        dx = ref.x - x
        dy = ref.y - y
        e_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        e_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        e_psi = wrap_to_pi(ref.yaw - yaw)
        return e_x, e_y, e_psi

    def baseline_tracking_law(self, ref: RefPoint, e_x: float, e_y: float, e_psi: float):
        cmd_v = ref.v * math.cos(e_psi) + self.kx * e_x
        cmd_w = ref.w + self.ky * e_y + self.kpsi * e_psi

        cmd_v = clamp(cmd_v, self.v_min, self.v_max)
        cmd_w = clamp(cmd_w, self.w_min, self.w_max)
        return cmd_v, cmd_w

    def change_perturbation(self) -> None:
        while True:
            dv = random.uniform(self.dv_perturb_min, self.dv_perturb_max)
            dw = random.uniform(self.dw_perturb_min, self.dw_perturb_max)

            pert_norm = math.sqrt(dv*dv + 0.1*dw*dw)
            if pert_norm >= self.min_perturb_norm:
                self.current_dv_pert = dv
                self.current_dw_pert = dw
                break
        
        self.perturb_hold_count = random.randint(self.perturb_hold_steps_min, self.perturb_hold_steps_max)

    def write_row(
        self, 
        ref: RefPoint,
        x: float,
        y: float,
        yaw: float,
        v_meas: float,
        w_meas: float,
        e_x: float,
        e_y: float,
        e_psi: float,
        cmd_v_nom: float,
        cmd_w_nom: float,
        dv_pert: float,
        dw_pert: float,
        cmd_v: float,
        cmd_w: float
    ) -> None:
        sim_time = self.get_clock().now().nanoseconds * 1e-9

        self.writer.writerow([
            self.step_idx, ref.segment, f'{sim_time:.6f}',
            f'{ref.x:.6f}', f'{ref.y:.6f}', f'{ref.yaw:.6f}', f'{ref.v:.6f}', f'{ref.w:.6f}',
            f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}', f'{v_meas:.6f}', f'{w_meas:.6f}',
            f'{e_x:.6f}', f'{e_y:.6f}', f'{e_psi:.6f}',
            f'{cmd_v:.6f}', f'{cmd_w:.6f}',
            f'{cmd_v_nom:.6f}', f'{cmd_w_nom:.6f}', f'{dv_pert:.6f}', f'{dw_pert:.6f}',
        ])

        if self.step_idx % 50 == 0:
            self.csv_file.flush()

    def finish(self) -> None:
        if self.finished:
            return

        self.finished = True
        self.get_logger().info('Collection finished. Sending zero command...')
        for _ in range(5):
            self.publish_cmd(0.0, 0.0)

        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info(f'Saved dataset to: {self.csv_path}')
        rclpy.shutdown()

    def on_timer(self) -> None:
        if self.finished:
            return

        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.path_msg)

        if self.last_odom is None:
            self.publish_cmd(0.0, 0.0)
            return

        ref_idx = max(0, self.step_idx - self.warmup_steps)
        if ref_idx >= len(self.ref_traj):
            self.finish()
            return

        ref = self.ref_traj[ref_idx]

        is_warmup = self.step_idx < self.warmup_steps

        if is_warmup:
            cmd_v_nom = 0.0; dv_pert = 0.0; cmd_v = 0.0
            cmd_w_nom = 0.0; dw_pert = 0.0; cmd_w = 0.0
            x, y, yaw, v_meas, w_meas = self.current_measured_state()
            e_x, e_y, e_psi = self.compute_body_frame_error(ref, x, y, yaw)
            self.write_row(ref, x, y, yaw, v_meas, w_meas, e_x, e_y, e_psi, 
            cmd_v_nom, cmd_w_nom, dv_pert, dw_pert, cmd_v, cmd_w)
            self.publish_cmd(cmd_v, cmd_w)
            self.step_idx += 1
            return

        x, y, yaw, v_meas, w_meas = self.current_measured_state()
        e_x, e_y, e_psi = self.compute_body_frame_error(ref, x, y, yaw)
        cmd_v_nom, cmd_w_nom = self.baseline_tracking_law(ref, e_x, e_y, e_psi)

        dv_pert = 0.0; dw_pert = 0.0

        perturbation_allowed = self.enable_perturbation
        if self.perturb_after_warmup_only and is_warmup:
            perturbation_allowed = False
        
        if perturbation_allowed and ref.segment != 'final_stop':
            if self.perturb_hold_count <= 0:
                self.change_perturbation()
            self.perturb_hold_count -= 1
            dv_pert = self.current_dv_pert
            dw_pert = self.current_dw_pert
        
        cmd_v = clamp(cmd_v_nom + dv_pert, self.v_min, self.v_max)
        cmd_w = clamp(cmd_w_nom + dw_pert, self.w_min, self.w_max)

        self.publish_cmd(cmd_v, cmd_w)
        self.write_row(ref, x, y, yaw, v_meas, w_meas, e_x, e_y, e_psi,
         cmd_v_nom, cmd_w_nom, dv_pert, dw_pert, cmd_v, cmd_w)

        if self.step_idx % 20 == 0:
            self.get_logger().info(
                f"step={self.step_idx:4d}/{len(self.ref_traj)} "
                f"seg={ref.segment:>10s} "
                f"e=({e_x:+.3f}, {e_y:+.3f}, {e_psi:+.3f}) "
                f"cmd=({cmd_v:+.3f}, {cmd_w:+.3f})"
            )

        self.step_idx += 1

    def cleanup(self) -> None:
        try:
            for _ in range(5):
                self.publish_cmd(0.0, 0.0)
        except Exception:
            pass

        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass


def main():
    rclpy.init()
    node = CollectNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()