#!usr/bin/env python3
import csv, glob, math, os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Optional, Tuple

import numpy as np
try:
    import cvxpy as cp
except ImportError:
    cp = None

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
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

def block_hankel(data: np.ndarray, L: int) -> np.ndarray:
    """
    data: shape (dim, T)
    return: shape (dim*L, T-L+1)
    """

    dim, T = data.shape
    if T < L:
        raise ValueError(f"Not enough data for Hankel. T = {T}, L={L}")

    n_col = T - L + 1
    H = np.zeros((dim*L, n_col), dtype=np.float64)
    for i in range(n_col):
        H[:,i] = data[:, i:i+L].reshape(-1, order="F")
    return H

def load_dataset_csv(csv_path: str, drop_initial_rows: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if drop_initial_rows > 0:
        rows = rows[drop_initial_rows:]

    if len(rows) == 0:
        raise RuntimeError("Dataset CSV is empty after dropping initial rows.")

    u_list = []; y_list = []

    required = ["cmd_v", "cmd_w", "e_x", "e_y", "e_psi"]
    for key in required:
        if key not in rows[0]:
            if key not in rows[0]:
                raise KeyError(f"CSV column '{key}' is missing.")
    
    for row in rows:
        u_list.append([float(row["cmd_v"]), float(row["cmd_w"])])
        y_list.append([float(row["e_x"]), float(row["e_y"]), float(row["e_psi"])])

    u_data = np.asarray(u_list, dtype=np.float64).T
    y_data = np.asarray(y_list, dtype=np.float64).T
    return u_data, y_data

class DeePCSolver:
    def __init__(self, u_data: np.ndarray, y_data: np.ndarray, Tini: int, N: int, 
    Q_diag: List[float], R_diag: List[float], lambda_g: float, lambda_y: float, solver_name: str = "OSQP") -> None:
        if cp is None:
            raise RuntimeError("cvxpy is not installed")

        self.u_dim = u_data.shape[0]
        self.y_dim = y_data.shape[0]
        self.Tini = Tini; self.N = N; self.L = Tini + N; self.solver_name = solver_name

        Hu = block_hankel(u_data, self.L); Hy = block_hankel(y_data, self.L)

        self.Up = Hu[:self.u_dim*Tini, :]; self.Uf = Hu[self.u_dim*Tini:, :]
        self.Yp = Hy[:self.y_dim*Tini, :]; self.Yf = Hy[self.y_dim*Tini:, :]

        self.n_col = self.Up.shape[1]

        Q = np.diag(np.asarray(Q_diag, dtype=np.float64)); R = np.diag(np.asarray(R_diag, dtype=np.float64))
        self.Q_blk = np.kron(np.eye(N), Q); self.R_blk = np.kron(np.eye(N), R)

        self.u_ini_p = cp.Parameter(self.u_dim*Tini)
        self.y_ini_p = cp.Parameter(self.y_dim*Tini)
        self.u_ref_p = cp.Parameter(self.u_dim*N)
        self.y_ref_p = cp.Parameter(self.y_dim*N)
        self.u_min_p = cp.Parameter(self.u_dim)
        self.u_max_p = cp.Parameter(self.u_dim)

        self.g = cp.Variable(self.n_col)
        self.sigma_y = cp.Variable(self.y_dim*Tini)
        self.u_f = cp.Variable(self.u_dim*N)
        self.y_f = cp.Variable(self.y_dim*N)
        
        cost = (
            cp.quad_form(self.y_f - self.y_ref_p, self.Q_blk) +
            cp.quad_form(self.u_f - self.u_ref_p, self.R_blk) +
            lambda_g*cp.sum_squares(self.g) + lambda_y* cp.sum_squares(self.sigma_y)
        )

        constraints = [
            self.Up @ self.g == self.u_ini_p,
            self.Yp @ self.g == self.y_ini_p + self.sigma_y,
            self.Uf @ self.g == self.u_f,
            self.Yf @ self.g == self.y_f
        ]

        for k in range(N):
            uk = self.u_f[k*self.u_dim:(k+1)*self.u_dim]
            constraints += [
                uk >= self.u_min_p,
                uk <= self.u_max_p
            ]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, u_ini: np.ndarray, y_ini: np.ndarray, 
    u_ref: np.ndarray, y_ref: np.ndarray, 
    u_min: np.ndarray, u_max: np.ndarray) -> Tuple[np.ndarray, : np.ndarray]:
        self.u_ini_p.value = u_ini
        self.y_ini_p.value = y_ini
        self.u_ref_p.value = u_ref
        self.y_ref_p.value = y_ref
        self.u_min_p.value = u_min
        self.u_max_p.value = u_max

        solver = getattr(cp, self.solver_name, None)
        if solver is None:
            self.problem.solve(warm_start = True, verbose = False)
        else:
            self.problem.solve(solver = solver, warm_start = True, verbose = False)
        
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"DeePC solve failed: {self.problem.status}")

        u0 = np.asarray(self.u_f.value[: self.u_dim], dtype=np.float64).copy()
        y_pred = np.asarray(self.y_f.value, dtype=np.float64).copy()
        return u0, y_pred

class DeePCNode(Node):
    def __init__(self) -> Node:
        super().__init__("deepc_node")

        # ROS topics
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/model/rosbot/odometry_gt")
        self.declare_parameter("path_topic", "/deepc/reference_path")

        # timing
        self.declare_parameter("sample_time", 0.05)

        # reference trajectory
        self.declare_parameter("straight_1_length", 1.5)
        self.declare_parameter("straight_2_length", 1.0)
        self.declare_parameter("straight_3_length", 1.5)
        self.declare_parameter("straight_speed", 0.25)

        self.declare_parameter("turn_radius", 0.8)
        self.declare_parameter("turn_angle_deg", 90.0)
        self.declare_parameter("turn_speed", 0.20)

        # finish
        self.declare_parameter("goal_pos_tol", 0.08)
        self.declare_parameter("goal_yaw_tol", 0.15)
        self.declare_parameter("final_hold_steps", 10)
        self.declare_parameter("max_final_stop_steps", 200)

        # fallback baseline controller
        self.declare_parameter("kx", 0.8)
        self.declare_parameter("ky", 1.8)
        self.declare_parameter("kpsi", 2.0)

        # input limits
        self.declare_parameter("v_min", -0.2)
        self.declare_parameter("v_max", 0.45)
        self.declare_parameter("w_min", -1.2)
        self.declare_parameter("w_max", 1.2)

        # dataset / DeePC
        self.declare_parameter("dataset_csv", "")
        self.declare_parameter("dataset_dir", "/ws/datasets")
        self.declare_parameter("drop_initial_rows", 20)

        self.declare_parameter("Tini", 12)
        self.declare_parameter("horizon", 15)
        self.declare_parameter("Q_diag", [30.0, 45.0, 12.0])  # e_x, e_y, e_psi
        self.declare_parameter("R_diag", [1.0, 0.2])          # v, w
        self.declare_parameter("lambda_g", 1.0)
        self.declare_parameter("lambda_y", 10000.0)
        self.declare_parameter("solver_name", "OSQP")

        # reference search
        self.declare_parameter("ref_back_search", 3)
        self.declare_parameter("ref_forward_search", 25)

        # Logging
        self.declare_parameter("actual_path_topic", "/deepc/actual_path")
        self.declare_parameter("output_dir", "/ws/results")
        self.declare_parameter("file_prefix", "deepc_run")

        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)

        self.dt = float(self.get_parameter("sample_time").value)

        self.straight_1_length = float(self.get_parameter("straight_1_length").value)
        self.straight_2_length = float(self.get_parameter("straight_2_length").value)
        self.straight_3_length = float(self.get_parameter("straight_3_length").value)
        self.straight_speed = float(self.get_parameter("straight_speed").value)

        self.turn_radius = float(self.get_parameter("turn_radius").value)
        self.turn_angle_deg = float(self.get_parameter("turn_angle_deg").value)
        self.turn_speed = float(self.get_parameter("turn_speed").value)

        self.goal_pos_tol = float(self.get_parameter("goal_pos_tol").value)
        self.goal_yaw_tol = float(self.get_parameter("goal_yaw_tol").value)
        self.final_hold_steps = int(self.get_parameter("final_hold_steps").value)
        self.max_final_stop_steps = int(self.get_parameter("max_final_stop_steps").value)

        self.kx = float(self.get_parameter("kx").value)
        self.ky = float(self.get_parameter("ky").value)
        self.kpsi = float(self.get_parameter("kpsi").value)

        self.v_min = float(self.get_parameter("v_min").value)
        self.v_max = float(self.get_parameter("v_max").value)
        self.w_min = float(self.get_parameter("w_min").value)
        self.w_max = float(self.get_parameter("w_max").value)

        self.dataset_csv = str(self.get_parameter("dataset_csv").value)
        self.dataset_dir = str(self.get_parameter("dataset_dir").value)
        self.drop_initial_rows = int(self.get_parameter("drop_initial_rows").value)

        self.Tini = int(self.get_parameter("Tini").value)
        self.N = int(self.get_parameter("horizon").value)
        self.Q_diag = list(self.get_parameter("Q_diag").value)
        self.R_diag = list(self.get_parameter("R_diag").value)
        self.lambda_g = float(self.get_parameter("lambda_g").value)
        self.lambda_y = float(self.get_parameter("lambda_y").value)
        self.solver_name = str(self.get_parameter("solver_name").value)

        self.ref_back_search = int(self.get_parameter("ref_back_search").value)
        self.ref_forward_search = int(self.get_parameter("ref_forward_search").value)

        self.actual_path_topic = str(self.get_parameter("actual_path_topic").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.file_prefix = str(self.get_parameter("file_prefix").value)

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)
        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        self.actual_path_pub = self.create_publisher(Path, self.actual_path_topic, 1)

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)

        self.last_odom: Optional[Odometry] = None
        self.step_idx = 0; self.ref_idx = 0
        self.warned_solve = False

        self.u_hist: Deque[np.ndarray] = deque(maxlen = self.Tini)
        self.y_hist: Deque[np.ndarray] = deque(maxlen = self.Tini)

        self.ref_traj = self.build_reference_trajectory()
        self.path_msg = self.build_reference_path_msg()

        self.finished = False
        self.final_reached_count = 0
        self.final_stop_count = 0

        self.run_rows = []
        self.actual_path_msg = Path()
        self.actual_path_msg.header.frame_id = "odom"

        csv_path = self.resolve_dataset_path()
        u_data, y_data = load_dataset_csv(csv_path, self.drop_initial_rows)
        self.deepc = DeePCSolver(
            u_data = u_data, y_data = y_data,
            Tini = self.Tini, N = self.N,
            Q_diag = self.Q_diag, R_diag = self.R_diag,
            lambda_g = self.lambda_g, lambda_y = self.lambda_y,
            solver_name = self.solver_name
        )

        self.timer = self.create_timer(self.dt, self.on_timer)

        self.get_logger().info("DeePC tracking node started.")
        self.get_logger().info(f"dataset_csv : {csv_path}")
        self.get_logger().info(f"Tini = {self.Tini}, N = {self.N}")
        self.get_logger().info(f"cmd_topic    : {self.cmd_topic}")
        self.get_logger().info(f"odom_topic   : {self.odom_topic}")

    def resolve_dataset_path(self) -> str:
        if self.dataset_csv:
            return self.dataset_csv

        pattern = os.path.join(self.dataset_dir, "*.csv")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No CSV files found in dataset_dir: {self.dataset_dir}")
        # Get most recent dataset
        files.sort(key=os.path.getmtime)
        return files[-1]
    
    def build_reference_trajectory(self) -> List[RefPoint]:
        traj: List[RefPoint] = []

        t = 0.0; x = 0.0; y = 0.0; yaw = 0.0
        turn_angle = math.radians(self.turn_angle_deg)

        def append_straight(length: float, v: float, name: str):
            nonlocal x, y, yaw, t, traj
            steps = max(1, int(round(length/max(abs(v), 1e-6)/self.dt)))
            for _ in range(steps):
                traj.append(RefPoint(t=t, x=x, y=y, yaw=yaw, v=v, w=0.0, segment=name))
                x += v*math.cos(yaw)*self.dt
                y += v*math.sin(yaw)*self.dt
                t += self.dt

        def append_turn(radius: float, angle: float, v: float, left: bool, name: str):
            nonlocal x, y, yaw, t, traj
            w = abs(v)/max(radius, 1e-6)
            if not left:
                w = -w

            steps = max(1, int(round(abs(angle)/max(abs(w), 1e-6)/self.dt)))
            for _ in range(steps):
                traj.append(RefPoint(t=t, x=x, y=y, yaw=yaw, v=v, w=w, segment=name))
                x += v*math.cos(yaw)*self.dt
                y += v*math.sin(yaw)*self.dt
                yaw += w*self.dt
                t += self.dt

        append_straight(self.straight_1_length, self.straight_speed, "straight_1")
        append_turn(self.turn_radius, turn_angle, self.turn_speed, True, "left_turn")
        append_straight(self.straight_2_length, self.straight_speed, "straight_2")
        append_turn(self.turn_radius, turn_angle, self.turn_speed, False, "right_turn")
        append_straight(self.straight_3_length, self.straight_speed, "straight_3")

        for _ in range(20):
            traj.append(RefPoint(t=t, x=x, y=y, yaw=yaw, v=0.0, w=0.0, segment="final_stop"))
            t += self.dt

        return traj

    def build_reference_path_msg(self) -> Path:
        msg = Path()
        msg.header.frame_id = "odom"

        for p in self.ref_traj:
            ps = PoseStamped()
            ps.header.frame_id = "odom"
            ps.pose.position.x = p.x
            ps.pose.position.y = p.y
            ps.pose.position.z = 0.0
            ps.pose.orientation.z = math.sin(p.yaw / 2.0)
            ps.pose.orientation.w = math.cos(p.yaw / 2.0)
            msg.poses.append(ps)

        return msg

    def on_odom(self, msg: Odometry) -> None:
        self.last_odom = msg

    def publish_cmd(self, v: float, w: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def current_measured_state(self) -> Tuple[float, float, float, float, float]:
        odom = self.last_odom
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        v = odom.twist.twist.linear.x
        w = odom.twist.twist.angular.z
        return x, y, yaw, v, w

    def compute_body_frame_error(self, ref: RefPoint, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        dx = ref.x - x
        dy = ref.y - y
        e_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        e_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        e_psi = wrap_to_pi(ref.yaw - yaw)
        return e_x, e_y, e_psi

    def baseline_tracking_law(self, ref: RefPoint, e_x: float, e_y: float, e_psi: float) -> Tuple[float, float]:
        cmd_v = ref.v * math.cos(e_psi) + self.kx * e_x
        cmd_w = ref.w + self.ky * e_y + self.kpsi * e_psi

        cmd_v = clamp(cmd_v, self.v_min, self.v_max)
        cmd_w = clamp(cmd_w, self.w_min, self.w_max)
        return cmd_v, cmd_w

    def update_reference_index(self, x: float, y: float) -> None:
        start = max(0, self.ref_idx - self.ref_back_search)
        stop = min(len(self.ref_traj), self.ref_idx + self.ref_forward_search)

        best_idx = self.ref_idx
        best_dist2 = float("inf")

        for idx in range(start, stop):
            dx = self.ref_traj[idx].x - x
            dy = self.ref_traj[idx].y - y
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_idx = idx

        self.ref_idx = best_idx

    def build_future_reference(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        y_ref: zero error tracking
        u_ref: feedforward (ref_v, ref_w)
        """
        u_ref = []; y_ref = []

        for k in range(self.N):
            idx = min(self.ref_idx + k, len(self.ref_traj) - 1)
            rp = self.ref_traj[idx]
            u_ref.extend([rp.v, rp.w])
            y_ref.extend([0.0, 0.0, 0.0])

        return (
            np.asarray(u_ref, dtype=np.float64),
            np.asarray(y_ref, dtype=np.float64),
        )

    def append_actual_path(self, x: float, y: float, yaw: float) -> None:
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "odom"
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)

        self.actual_path_msg.poses.append(ps)
        self.actual_path_msg.header.stamp = ps.header.stamp
        self.actual_path_pub.publish(self.actual_path_msg)

    def log_step(self, mode: str, ref: RefPoint,
    x: float, y: float, yaw: float, v_meas: float, w_meas: float,
    e_x: float, e_y: float, e_psi: float,
    cmd_v: float, cmd_w: float) -> None:
        sim_time = self.get_clock().now().nanoseconds * 1e-9

        self.run_rows.append({
            "step": self.step_idx, "mode": mode, "sim_time_sec": sim_time,
            "ref_idx": self.ref_idx, "segment": ref.segment,

            "ref_x": ref.x, "ref_y": ref.y, "ref_yaw": ref.yaw, "ref_v": ref.v, "ref_w": ref.w,

            "x": x, "y": y, "yaw": yaw, "v_meas": v_meas, "w_meas": w_meas,

            "e_x": e_x, "e_y": e_y, "e_psi": e_psi,

            "cmd_v": cmd_v, "cmd_w": cmd_w,
        })

    def save_run_csv(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"{self.file_prefix}_{stamp}.csv")

        fieldnames = [
            "step", "mode", "sim_time_sec", "ref_idx", "segment",
            "ref_x", "ref_y", "ref_yaw", "ref_v", "ref_w",
            "x", "y", "yaw", "v_meas", "w_meas",
            "e_x", "e_y", "e_psi",
            "cmd_v", "cmd_w",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.run_rows)

        self.get_logger().info(f"Saved run log to: {csv_path}")

    def finish(self) -> None:
        if self.finished:
            return

        self.finished = True
        self.get_logger().info("Reference tracking finished. Sending zero command and saving results...")

        for _ in range(5):
            self.publish_cmd(0.0, 0.0)

        self.save_run_csv()
        rclpy.shutdown()

    def check_finish_condition(self, ref: RefPoint, e_x: float, e_y: float, e_psi: float) -> None:
        pos_err = math.sqrt(e_x * e_x + e_y * e_y)
        yaw_err = abs(e_psi)

        if ref.segment == "final_stop":
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
                self.get_logger().warn("Final stop timeout reached. Finishing run anyway.")
                self.finish()
                return
        else:
            self.final_stop_count = 0
            self.final_reached_count = 0

    def on_timer(self) -> None:
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.path_msg)

        if self.last_odom is None:
            self.publish_cmd(0.0, 0.0)
            return

        x, y, yaw, v_meas, w_meas = self.current_measured_state()
        self.append_actual_path(x, y, yaw)
        self.update_reference_index(x, y)
        ref = self.ref_traj[self.ref_idx]

        e_x, e_y, e_psi = self.compute_body_frame_error(ref, x, y, yaw)
        y_now = np.asarray([e_x, e_y, e_psi], dtype=np.float64)

        cmd_v_base, cmd_w_base = self.baseline_tracking_law(ref, e_x, e_y, e_psi)

        cmd_v = cmd_v_base
        cmd_w = cmd_w_base
        mode = "baseline"

        if len(self.u_hist) >= self.Tini and len(self.y_hist) >= self.Tini:
            u_ini = np.concatenate(list(self.u_hist), axis=0)
            y_ini = np.concatenate(list(self.y_hist), axis=0)
            u_ref, y_ref = self.build_future_reference()

            try:
                u0, _ = self.deepc.solve(
                    u_ini=u_ini,
                    y_ini=y_ini,
                    u_ref=u_ref,
                    y_ref=y_ref,
                    u_min=np.asarray([self.v_min, self.w_min], dtype=np.float64),
                    u_max=np.asarray([self.v_max, self.w_max], dtype=np.float64),
                )
                cmd_v = clamp(float(u0[0]), self.v_min, self.v_max)
                cmd_w = clamp(float(u0[1]), self.w_min, self.w_max)
                mode = "deepc"
            except Exception as exc:
                cmd_v = cmd_v_base
                cmd_w = cmd_w_base
                if (not self.warned_solve) or (self.step_idx % 100 == 0):
                    self.get_logger().warn(f"DeePC solver fallback to baseline: {exc}")
                    self.warned_solve = True
        
        self.log_step(mode=mode, ref=ref,
             x=x, y=y, yaw=yaw, v_meas=v_meas, w_meas=w_meas,
            e_x=e_x, e_y=e_y, e_psi=e_psi, cmd_v=cmd_v, cmd_w=cmd_w,)

        self.publish_cmd(cmd_v, cmd_w)

        # history stores (u_k, y_k) pairs consistent with collected dataset
        self.u_hist.append(np.asarray([cmd_v, cmd_w], dtype=np.float64))
        self.y_hist.append(y_now)

        self.check_finish_condition(ref, e_x, e_y, e_psi)
        if self.finished:
            return

        if self.step_idx % 20 == 0:
            self.get_logger().info(
                f"[{mode}] step={self.step_idx:4d} ref_idx={self.ref_idx:4d} "
                f"e=({e_x:+.3f}, {e_y:+.3f}, {e_psi:+.3f}) "
                f"u=({cmd_v:+.3f}, {cmd_w:+.3f}) "
                f"meas=({v_meas:+.3f}, {w_meas:+.3f})"
            )

        self.step_idx += 1

    def cleanup(self) -> None:
        try:
            for _ in range(5):
                self.publish_cmd(0.0, 0.0)
        except Exception:
            pass

        try:
            if self.run_rows and not self.finished:
                self.save_run_csv()
        except Exception:
            pass


def main() -> None:
    rclpy.init()
    node = DeePCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

    


